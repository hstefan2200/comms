import numpy as np
from modulation import BPSKModulator
from channel_model import AWGNChannel
from audio_processing import AudioProcessor
from image_processing import ImageProcessor
import matplotlib.pyplot as plt

class Transmitter:
    def __init__(self, carrier_freq=1000, sample_rate=8000, quantization_bits=8):
        self.modulator = BPSKModulator(carrier_freq, sample_rate)
        self.channel = AWGNChannel()
        self.audio_processor = AudioProcessor(quantization_bits)
        self.image_processor = ImageProcessor()
        
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        
    def prepare_data_packet(self, image_path=None, audio_path=None):
        packet_bits = []
        metadata = {
            'image_length': 0,
            'audio_length': 0,
            'total_length': 0
        }
        
        #process image data
        if image_path:
            print(f"Processing image: {image_path}")
            try:
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    image_bits = self.image_processor.image_to_bits(image_path)
                    metadata['image_length'] = len(image_bits)
                    packet_bits.extend(image_bits)
                    print(f"Image converted to {len(image_bits)} bits")
                else:
                    print("Failed to load image")
                    metadata['image_length'] = 0
            except Exception as e:
                print(f"Error processing image {e}")
                metadata['image_length'] = 0
        
        #Process audio data        
        if audio_path:
            print(f"Processing audio: {audio_path}")
            try:
                audio_data, sample_rate = self.audio_processor.load_audio_file(audio_path)
                if audio_data is not None:
                    audio_bits, _ = self.audio_processor.audio_to_bits(audio_data)
                    metadata['audio_length'] = len(audio_bits)
                    packet_bits.extend(audio_bits)
                    print(f"Audio converted to {len(audio_bits)} bits")
                else:
                    print("Failed to load audio file")
                    metadata['audio_length'] = 0
            except Exception as e:
                print(f"Error processing audio {e}")
                metadata['audio_length'] = 0
                
        header_bits = []
        for length in [metadata['image_length'], metadata['audio_length']]:
            length_bits = [(length >> i) & 1 for i in range(15, -1, -1)]
            header_bits.extend(length_bits)
            
        #combine header and data arrays
        full_packet = header_bits + packet_bits 
        metadata['total_length'] = len(full_packet)
        
        print(f"Total packet size: {len(full_packet)} bits")
        
        return np.array(full_packet, dtype=int), metadata
    
    def transmit_data(self, data_bits, snr_db=10, output_audio_file=None):
        print(f"\nTransmitting {len(data_bits)} bits at SNR = {snr_db}")
        
        print("MOdulating Data")
        modulated_signal = self.modulator.modulate(data_bits)
        print(f"Generated modulated signal with {len(modulated_signal)} samples")
        
        print("Adding AWGN noise")
        noisy_signal, noise = self.channel.add_noise(modulated_signal, snr_db)
        
        actual_snr = self.channel.calculate_snr(modulated_signal, noise)
        print(f"Target SNR: {snr_db} dB, Actual SNR: {actual_snr} dB")
        
        #convert to audio format (ie: according to directions this needs to be an rf signal)
        print("Converting to sound for transmission")
        max_amplitude = np.max(np.abs(noisy_signal))
        if max_amplitude > 0:
            audio_signal = noisy_signal / max_amplitude * 0.8 #80% of max to prevent clipping
        else:
            audio_signal = noisy_signal
            
        if output_audio_file:
            print(f"saving transmitted audio to {output_audio_file}")
            self.audio_processor.save_audio_file(audio_signal, output_audio_file, self.sample_rate)
            
        return audio_signal, modulated_signal, noise, actual_snr
    
    
    def plot_transmission_analysis(self, original_signal, noisy_signal, data_bits):
        """Plot analysis of the transmission"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original vs Noisy Signal (first 1000 samples)
        plt.subplot(2, 3, 1)
        samples_to_plot = min(1000, len(original_signal))
        t = np.arange(samples_to_plot) / self.sample_rate
        plt.plot(t, original_signal[:samples_to_plot], 'b-', label='Original', linewidth=1)
        plt.plot(t, noisy_signal[:samples_to_plot], 'r--', label='Noisy', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Signal spectrum
        plt.subplot(2, 3, 2)
        freqs = np.fft.fftfreq(len(original_signal), 1/self.sample_rate)
        signal_fft = np.abs(np.fft.fft(original_signal))
        plt.plot(freqs[:len(freqs)//2], signal_fft[:len(freqs)//2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Signal Spectrum')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Data bits (first 100)
        plt.subplot(2, 3, 3)
        bits_to_plot = min(100, len(data_bits))
        plt.stem(range(bits_to_plot), data_bits[:bits_to_plot], basefmt=' ')
        plt.xlabel('Bit Index')
        plt.ylabel('Bit Value')
        plt.title(f'Data Bits (first {bits_to_plot})')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Signal constellation (BPSK)
        plt.subplot(2, 3, 4)
        samples_per_symbol = self.sample_rate // 1000
        num_symbols = len(data_bits)
        symbol_samples = []
        
        for i in range(min(100, num_symbols)):  # Plot first 100 symbols
            start_idx = i * samples_per_symbol
            end_idx = (i + 1) * samples_per_symbol
            if end_idx <= len(original_signal):
                symbol_avg = np.mean(original_signal[start_idx:end_idx])
                symbol_samples.append(symbol_avg)
        
        plt.scatter(symbol_samples, np.zeros(len(symbol_samples)), alpha=0.6)
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.title('BPSK Constellation')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Power spectral density
        plt.subplot(2, 3, 5)
        f, psd = plt.psd(original_signal, NFFT=1024, Fs=self.sample_rate)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.title('Power Spectral Density')
        
        # Plot 6: Statistics
        plt.subplot(2, 3, 6)
        stats_text = f"""Transmission Statistics:
        
Data Bits: {len(data_bits)}
Signal Samples: {len(original_signal)}
Sample Rate: {self.sample_rate} Hz
Carrier Freq: {self.carrier_freq} Hz
Duration: {len(original_signal)/self.sample_rate:.2f} seconds

Signal Power: {np.mean(original_signal**2):.6f}
Peak Amplitude: {np.max(np.abs(original_signal)):.3f}
RMS: {np.sqrt(np.mean(original_signal**2)):.3f}"""
        
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    

def test_transmitter():
    print("-" * 50)
    print("Transmission Test")
    
    transmitter = Transmitter(carrier_freq=1000, sample_rate=8000, quantization_bits=16)
    
    #Image test
    print("Testing image transmission")
    print("-"*50)
    try:
        image_bits, image_metadata = transmitter.prepare_data_packet(image_path="data/matlab_logo.png")
        audio_signal, clean_signal, noise, actual_snr = transmitter.transmit_data(image_bits, snr_db=10, output_audio_file="data/transmitted_image.wav")
        # transmitter.plot_transmission_analysis(clean_signal, audio_signal, image_bits)
    except Exception as e:
        print(f"Image test failed: {e}")
    
    
    #Audio Test
    print("-"*50)
    print("Audio test")
    
    try:
        audio_bits, audio_metadata = transmitter.prepare_data_packet(audio_path="data/audio_recording.wav")
        audio_signal2, clean_signal2, noise2, actual_snr2 = transmitter.transmit_data(audio_bits, snr_db=10, output_audio_file="data/transmitted_audio.wav")
        # transmitter.plot_transmission_analysis(clean_signal2, audio_signal2, audio_bits)
    except Exception as e:
        print(f"Audio test failed: {e}")
        
    #combined test
    print("-"*50)
    print("Testing combined transmission")
    try:
        combined_bits, combined_metadata = transmitter.prepare_data_packet(image_path="data/matlab_logo.png", audio_path="data/audio_recording.wav")
        audio_signal3, clean_signal3, noise3, actual_snr3 = transmitter.transmit_data(combined_bits, snr_db=10, output_audio_file="data/transmitted_combined.wav")
        transmitter.plot_transmission_analysis(clean_signal3, audio_signal3, combined_bits)
    except Exception as e:
        print(f"Combined test failed: {e}")
        
            
if __name__ == "__main__":
    test_transmitter()

    
        
        