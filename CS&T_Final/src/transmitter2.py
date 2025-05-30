import numpy as np
import os
from PIL import Image
from scipy.io import wavfile
from scipy import signal
from src.modulation import BPSKModulator
from src.channel_model import AWGNChannel
from src.audio_processing import AudioProcessor
from src.image_processing import ImageProcessor


class ImprovedTransmitter:
    #Better transmitter-- handles compression, but tries to keep audio understandable
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000, quantization_bits=16):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.quantization_bits = quantization_bits
        
        # Initialize core components
        self.modulator = BPSKModulator(carrier_freq, sample_rate, symbol_rate)
        self.channel = AWGNChannel()
        self.audio_processor = AudioProcessor(quantization_bits)
        self.image_processor = ImageProcessor()
        
        print(f"Transmitter initialized - Carrier: {carrier_freq}Hz, Sample Rate: {sample_rate}Hz")
    
    def audio_compression(self, audio_data, original_sr, target_bits_max=50000):
        print(f"Audio compressed: {len(audio_data)} samples at {original_sr} Hz")
        
        #Frequency range used in telephony for intelligible speech (300 Hz - 3400 Hz) --> https://pubs.aip.org/asa/jel/article/4/7/075202/3304664/Speech-intelligibility-and-talker-identification
        nyquist = original_sr / 2
        low_cutoff = 300 / nyquist
        high_cutoff = min(3400 / nyquist, 0.95)  # Don't go too close to Nyquist
        
        #Butterworth bandpass filter (should pass signals using cutoffs defined above. b and a are the filter coefficients)
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band') #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        #filtfilt applies the filter in both directions, which should cancel out any distortions 
        filtered_audio = signal.filtfilt(b, a, audio_data) #https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
        
        print(f"Applied speech filter: 300-3400 Hz")
        
        #determine downsample factor (choosing between frequency or number of bits, taking least agressive)
        #3kHz is max in filter, so from Nyquist we need at least 6kHz
        min_sample_rate = 6000
        max_downsample = max(1, original_sr // min_sample_rate)
        
        #factor in filesize
        current_bits = len(audio_data) * 16  # 16-bit quantization
        target_downsample = max(1, int(np.sqrt(current_bits / target_bits_max))) #sqrt makes this less agressive downsampling
        #Take the smaller factor (less agressive), max is 4
        downsample_factor = min(max_downsample, target_downsample, 4) 
        
        print(f"Downsampling factor: {downsample_factor}x (max was {max_downsample})")
        
        #apply downsampling factor
        if downsample_factor > 1:
            #decimate applies anti-alias filter first -->: prevents distortions
            downsampled_audio = signal.decimate(filtered_audio, downsample_factor, ftype='fir')
            effective_sr = original_sr // downsample_factor
        else:
            downsampled_audio = filtered_audio
            effective_sr = original_sr
        
        print(f"Result: {len(downsampled_audio)} samples at {effective_sr} Hz")
        
        #Normalization
        rms = np.sqrt(np.mean(downsampled_audio**2)) #average volume
        if rms > 0:
            normalized_audio = downsampled_audio / (rms * 3)  # Boost quiet parts, reduce loud parts
            normalized_audio = np.tanh(normalized_audio)  #clip to fit into -1, +1, (tanh is for "soft clipping", meaning that there isn't a hard cutoff at -1 or +1)
        else:
            normalized_audio = downsampled_audio
        
        print(f"Applied dynamic range compression")
        
        return normalized_audio, downsample_factor, effective_sr
    
    def create_packet(self, image_path=None, audio_path=None, image_quality=4):
        print("-"*60)
        print("Creating Transmission Packet")
        
        #improved metadata (beyond data sizes from original transmitter)
        packet_bits = []
        metadata = {
            'has_image': False,
            'has_audio': False,
            'image_width': 0,
            'image_height': 0,
            'image_pixels': 0,
            'audio_samples': 0,
            'audio_downsample': 1,
            'audio_sample_rate': 8000
        }
        
        #compress image (reduce size) and create packet of bits
        if image_path and os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            
            img = Image.open(image_path)
            if img.mode != 'L': #grayscale
                img = img.convert('L')
            
            new_width = img.size[0] // image_quality
            new_height = img.size[1] // image_quality
            new_width = max(new_width, 16)
            new_height = max(new_height, 16)
            
            compressed_img = img.resize((new_width, new_height), Image.LANCZOS)
            pixels = list(compressed_img.getdata())
            reduced_pixels = [p // 16 for p in pixels]  # 8 -> 4 bits per pixel
            
            metadata['has_image'] = True
            metadata['image_width'] = new_width
            metadata['image_height'] = new_height
            metadata['image_pixels'] = len(reduced_pixels)
            
            #Convert to bits
            for pixel in reduced_pixels:
                for i in range(3, -1, -1):
                    packet_bits.append((pixel >> i) & 1)
            
            print(f"Image: {img.size} -> {compressed_img.size} = {len(reduced_pixels)*4} bits")
        
        #compress audio (using filter and downsampling from above) and create packet of bits
        if audio_path and os.path.exists(audio_path):
            print(f"Processing audio: {audio_path}")
            
            audio_data, original_sr = self.audio_processor.load_audio_file(audio_path)
            if audio_data is not None:
                compressed_audio, downsample_factor, effective_sr = self.audio_compression(
                    audio_data, original_sr, target_bits_max=100000  # Allow more bits for better quality
                )
                
                #Convert to bits 
                audio_8bit = np.clip(compressed_audio, -1.0, 1.0)
                quantized_audio = np.round(audio_8bit * 127).astype(np.int16)
                
                #Convert to unsigned (0-255)
                unsigned_audio = np.clip(quantized_audio + 128, 0, 255).astype(np.uint8)
                
                #Convert each sample to binary
                audio_bits = []
                for sample in unsigned_audio:
                    for i in range(7, -1, -1):
                        audio_bits.append((sample >> i) & 1)
                
                metadata['has_audio'] = True
                metadata['audio_samples'] = len(compressed_audio)
                metadata['audio_downsample'] = downsample_factor
                metadata['audio_sample_rate'] = effective_sr
                
                packet_bits.extend(audio_bits)
                
                print(f"Audio: {len(audio_data)} -> {len(compressed_audio)} samples")
                print(f"Sample rate: {original_sr} -> {effective_sr} Hz")
                print(f"Audio bits: {len(audio_bits)} (8-bit quantization)")
        
        #Create header for metadata
        header_bits = []
        
        #has image and/or audio
        header_bits.append(1 if metadata['has_image'] else 0)
        header_bits.append(1 if metadata['has_audio'] else 0)
        
        # Image dimensions
        img_w = min(metadata['image_width'], 1023)
        img_h = min(metadata['image_height'], 1023)
        
        for i in range(9, -1, -1):
            header_bits.append((img_w >> i) & 1)
        for i in range(9, -1, -1):
            header_bits.append((img_h >> i) & 1)
        
        #num of audio samples divided by 10
        audio_scaled = min(metadata['audio_samples'] // 10, 4095)
        for i in range(11, -1, -1):
            header_bits.append((audio_scaled >> i) & 1)
        
        #Audio downsample factor
        downsample = min(metadata['audio_downsample'], 15)
        for i in range(3, -1, -1):
            header_bits.append((downsample >> i) & 1)
        
        #Audio sample rate
        sr_scaled = min(metadata['audio_sample_rate'] // 100, 1023)
        for i in range(9, -1, -1):
            header_bits.append((sr_scaled >> i) & 1)
        
        full_packet = header_bits + packet_bits
        metadata['total_bits'] = len(full_packet)
        metadata['header_bits'] = len(header_bits)
        
        print(f"Final packet: {len(full_packet)} bits")
        print(f"Header: {len(header_bits)} bits, Data: {len(packet_bits)} bits")
        
        return np.array(full_packet, dtype=int), metadata
    
    def transmit_data(self, data_bits, snr_db=15, output_audio_file=None):
        print(f"\nTransmitting {len(data_bits)} bits at SNR = {snr_db} dB")
        
        #Estimate transmission time
        transmission_time = len(data_bits) * (self.sample_rate // self.symbol_rate) / self.sample_rate
        print(f"Estimated transmission time: {transmission_time:.1f} seconds")
        
        #apply bpsk modulation
        print("Modulating data with BPSK...")
        modulated_signal = self.modulator.modulate(data_bits)
        print(f"Generated modulated signal with {len(modulated_signal)} samples")
        
        #add AWGN noise
        print("Adding AWGN noise...")
        noisy_signal, noise = self.channel.add_noise(modulated_signal, snr_db)
        
        #actual vs target snr
        actual_snr = self.channel.calculate_snr(modulated_signal, noise)
        print(f"Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.2f} dB")
        
        print("Converting to audio signal for transmission...")
        max_amplitude = np.max(np.abs(noisy_signal))
        if max_amplitude > 0:
            audio_signal = noisy_signal / max_amplitude * 0.8  # 80% of max to prevent clipping
        else:
            audio_signal = noisy_signal
        
        if output_audio_file:
            print(f"Saving transmitted audio to {output_audio_file}")
            self.audio_processor.save_audio_file(audio_signal, output_audio_file, self.sample_rate)
        
        return audio_signal, modulated_signal, noise, actual_snr
    
    def transmit_file(self, image_path=None, audio_path=None, snr_db=15, image_quality=4, output_file=None):
        #Putting everything together to do the transmission

        packet_bits, metadata = self.create_packet(image_path, audio_path, image_quality)

        transmission_time = len(packet_bits) * (self.sample_rate // self.symbol_rate) / self.sample_rate
        print(f"Estimated transmission time: {transmission_time}")

        actual_snr = max(snr_db, 15)
        
        if output_file is None:
            output_file = "data/transmitted_signal.wav"
        
        #Transmit
        audio_signal, clean_signal, noise, measured_snr = self.transmit_data(
            packet_bits, snr_db=actual_snr, output_audio_file=output_file
        )
        
        return output_file, metadata


# def test_transmitter():
# #debug
#     print("="*60)
#     print("TESTING IMPROVED TRANSMITTER")
#     print("="*60)
    
#     transmitter = ImprovedTransmitter(
#         carrier_freq=1000, 
#         sample_rate=8000, 
#         symbol_rate=2000,
#         quantization_bits=16
#     )

#     image_path = "data/matlab_logo.png" if os.path.exists("data/matlab_logo.png") else None
#     audio_path = "data/audio_recording.wav" if os.path.exists("data/audio_recording.wav") else None
    
#     if not audio_path and not image_path:
#         print("No test files found")
#         return
    
#     try:
#         output_file, metadata = transmitter.transmit_file(
#             image_path=image_path,
#             audio_path=audio_path,
#             snr_db=18,
#             image_quality=4,
#             output_file="data/test_transmission.wav"
#         )
        
#         if output_file:
#             print("Transmission successful!")
#             print(f"Output file: {output_file}")
#             if metadata:
#                 print(f"Total bits transmitted: {metadata['total_bits']}")
#         else:
#             print("Transmission failed")
            
#     except Exception as e:
#         print(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     test_transmitter()