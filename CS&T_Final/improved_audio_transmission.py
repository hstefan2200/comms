import numpy as np
import os
from PIL import Image
from scipy.io import wavfile
from scipy import signal

class ImprovedAudioTransmitter:
    """Enhanced transmitter with better audio quality preservation"""
    
    def __init__(self, carrier_freq=1000, sample_rate=8000, quantization_bits=16):
        from src.transmitter import Transmitter
        from src.receiver import Receiver
        from src.image_processing import ImageProcessor
        from src.audio_processing import AudioProcessor
        
        self.base_transmitter = Transmitter(carrier_freq, sample_rate, quantization_bits)
        self.base_receiver = Receiver(carrier_freq, sample_rate, quantization_bits)
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor(quantization_bits)
        
    def smart_audio_compression(self, audio_data, original_sr, target_bits_max=50000):
        """Intelligently compress audio while preserving speech intelligibility"""
        print(f"Smart audio compression: {len(audio_data)} samples at {original_sr} Hz")
        
        # Step 1: Apply speech-optimized filtering (300 Hz - 3400 Hz)
        # This is the frequency range used in telephony for intelligible speech
        nyquist = original_sr / 2
        low_cutoff = 300 / nyquist
        high_cutoff = min(3400 / nyquist, 0.95)  # Don't go too close to Nyquist
        
        # Design a bandpass filter for speech
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        print(f"Applied speech filter: 300-3400 Hz")
        
        # Step 2: Determine optimal downsampling factor
        # We want at least 3kHz bandwidth, so minimum 6kHz sample rate
        min_sample_rate = 6000
        max_downsample = max(1, original_sr // min_sample_rate)
        
        # But also consider the target bit count
        current_bits = len(audio_data) * 16  # 16-bit quantization
        target_downsample = max(1, int(np.sqrt(current_bits / target_bits_max)))
        
        # Use the more conservative (smaller) downsampling factor
        downsample_factor = min(max_downsample, target_downsample, 4)  # Cap at 4x
        
        print(f"Downsampling factor: {downsample_factor}x (max was {max_downsample})")
        
        # Step 3: Apply downsampling
        if downsample_factor > 1:
            # Use scipy's decimate for better quality than simple indexing
            downsampled_audio = signal.decimate(filtered_audio, downsample_factor, ftype='fir')
            effective_sr = original_sr // downsample_factor
        else:
            downsampled_audio = filtered_audio
            effective_sr = original_sr
        
        print(f"Result: {len(downsampled_audio)} samples at {effective_sr} Hz")
        
        # Step 4: Dynamic range compression (make quiet sounds louder)
        # This helps with speech intelligibility
        rms = np.sqrt(np.mean(downsampled_audio**2))
        if rms > 0:
            # Normalize to use more of the dynamic range
            normalized_audio = downsampled_audio / (rms * 3)  # Boost quiet parts
            # But don't clip loud parts
            normalized_audio = np.tanh(normalized_audio)  # Soft clipping
        else:
            normalized_audio = downsampled_audio
        
        print(f"Applied dynamic range compression")
        
        return normalized_audio, downsample_factor, effective_sr
    
    def create_improved_packet(self, image_path=None, audio_path=None, image_quality=4):
        """Create packet with improved audio and simple image compression"""
        print("="*60)
        print("CREATING IMPROVED PACKET")
        print("="*60)
        
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
        
        # Process image (same as before)
        if image_path and os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            
            # Reduce resolution
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
            
            # Convert to bits
            for pixel in reduced_pixels:
                for i in range(3, -1, -1):
                    packet_bits.append((pixel >> i) & 1)
            
            print(f"Image: {img.size} -> {compressed_img.size} = {len(reduced_pixels)*4} bits")
        
        # Process audio with improved compression
        if audio_path and os.path.exists(audio_path):
            print(f"Processing audio: {audio_path}")
            
            audio_data, original_sr = self.audio_processor.load_audio_file(audio_path)
            if audio_data is not None:
                # Apply smart compression
                compressed_audio, downsample_factor, effective_sr = self.smart_audio_compression(
                    audio_data, original_sr, target_bits_max=100000  # Allow more bits for better quality
                )
                
                # Convert to bits using 8-bit quantization for smaller size
                # but apply it to the already-compressed audio
                audio_8bit = np.clip(compressed_audio, -1.0, 1.0)
                quantized_audio = np.round(audio_8bit * 127).astype(np.int16)  # FIXED: Use int16 first
                
                # Convert to unsigned for bit conversion, ensuring no overflow
                unsigned_audio = np.clip(quantized_audio + 128, 0, 255).astype(np.uint8)  # FIXED: Clip to valid range
                
                # Convert each sample to 8 bits
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
        
        # Create header (48 bits for more metadata)
        header_bits = []
        
        # Flags (2 bits)
        header_bits.append(1 if metadata['has_image'] else 0)
        header_bits.append(1 if metadata['has_audio'] else 0)
        
        # Image dimensions (10 bits each)
        img_w = min(metadata['image_width'], 1023)
        img_h = min(metadata['image_height'], 1023)
        
        for i in range(9, -1, -1):
            header_bits.append((img_w >> i) & 1)
        for i in range(9, -1, -1):
            header_bits.append((img_h >> i) & 1)
        
        # Audio metadata (26 bits)
        # Audio samples (12 bits, scaled by 10)
        audio_scaled = min(metadata['audio_samples'] // 10, 4095)
        for i in range(11, -1, -1):
            header_bits.append((audio_scaled >> i) & 1)
        
        # Audio downsample factor (4 bits, max 15)
        downsample = min(metadata['audio_downsample'], 15)
        for i in range(3, -1, -1):
            header_bits.append((downsample >> i) & 1)
        
        # Audio sample rate (10 bits, divided by 100)
        sr_scaled = min(metadata['audio_sample_rate'] // 100, 1023)
        for i in range(9, -1, -1):
            header_bits.append((sr_scaled >> i) & 1)
        
        # Combine header and data
        full_packet = header_bits + packet_bits
        metadata['total_bits'] = len(full_packet)
        metadata['header_bits'] = len(header_bits)
        
        print(f"Final packet: {len(full_packet)} bits")
        print(f"Header: {len(header_bits)} bits, Data: {len(packet_bits)} bits")
        
        return np.array(full_packet, dtype=int), metadata
    
    def transmit_improved_file(self, image_path=None, audio_path=None, snr_db=15, image_quality=4):
        """Transmit with improved audio quality"""
        
        packet_bits, metadata = self.create_improved_packet(image_path, audio_path, image_quality)
        
        transmission_time = len(packet_bits) * 4 / 8000
        print(f"Estimated transmission time: {transmission_time:.1f} seconds")
        
        if transmission_time > 45:
            print("⚠️  WARNING: Transmission will take over 45 seconds!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                return None, None
        
        # Use high SNR for reliability
        actual_snr = max(snr_db, 18)
        print(f"Transmitting with SNR = {actual_snr} dB...")
        
        output_file = "data/improved_transmission.wav"
        audio_signal, clean_signal, noise, measured_snr = self.base_transmitter.transmit_data(
            packet_bits, snr_db=actual_snr, output_audio_file=output_file
        )
        
        return output_file, metadata
    
    def receive_improved_file(self, audio_file_path, output_dir="data/received/"):
        """Receive with improved audio reconstruction"""
        print("="*60)
        print("RECEIVING IMPROVED FILE")
        print("="*60)
        
        received_signal, sample_rate = self.base_receiver.load_received_signal(audio_file_path)
        if received_signal is None:
            return None
        
        recovered_bits = self.base_receiver.demodulate_signal(received_signal)
        
        # Parse header (48 bits)
        if len(recovered_bits) < 48:
            print("❌ Not enough bits for header")
            return None
        
        header = recovered_bits[:48]
        data_bits = recovered_bits[48:]
        
        # Parse header
        has_image = bool(header[0])
        has_audio = bool(header[1])
        
        # Image dimensions
        img_w = sum(bit * (2**(9-i)) for i, bit in enumerate(header[2:12]))
        img_h = sum(bit * (2**(9-i)) for i, bit in enumerate(header[12:22]))
        
        # Audio metadata
        audio_samples_scaled = sum(bit * (2**(11-i)) for i, bit in enumerate(header[22:34]))
        audio_samples = audio_samples_scaled * 10
        
        downsample_factor = sum(bit * (2**(3-i)) for i, bit in enumerate(header[34:38]))
        
        audio_sr_scaled = sum(bit * (2**(9-i)) for i, bit in enumerate(header[38:48]))
        audio_sample_rate = audio_sr_scaled * 100
        
        print(f"Parsed header:")
        print(f"  Image: {has_image} ({img_w}x{img_h})")
        print(f"  Audio: {has_audio} ({audio_samples} samples at {audio_sample_rate} Hz)")
        print(f"  Downsample factor: {downsample_factor}")
        
        current_pos = 0
        
        # Reconstruct image (same as before)
        if has_image and img_w > 0 and img_h > 0:
            print(f"\nReconstructing image...")
            bits_needed = img_w * img_h * 4
            
            if current_pos + bits_needed <= len(data_bits):
                image_bits = data_bits[current_pos:current_pos + bits_needed]
                current_pos += bits_needed
                
                pixels = []
                for i in range(0, len(image_bits), 4):
                    if i + 4 <= len(image_bits):
                        pixel_bits = image_bits[i:i+4]
                        pixel_value = sum(bit * (2**(3-j)) for j, bit in enumerate(pixel_bits))
                        pixels.append(pixel_value * 16)  # Restore to 8-bit
                
                if len(pixels) == img_w * img_h:
                    img_array = np.array(pixels, dtype=np.uint8).reshape(img_h, img_w)
                    img = Image.fromarray(img_array, mode='L')
                    
                    image_path = f"{output_dir}improved_received_image.png"
                    img.save(image_path)
                    print(f"✅ Image saved to {image_path}")
        
        # Reconstruct audio with improved quality
        if has_audio and audio_samples > 0:
            print(f"\nReconstructing audio...")
            audio_bits_needed = audio_samples * 8  # 8-bit quantization
            
            if current_pos + audio_bits_needed <= len(data_bits):
                audio_data_bits = data_bits[current_pos:current_pos + audio_bits_needed]
                
                # Convert bits back to samples
                audio_samples_reconstructed = []
                for i in range(0, len(audio_data_bits), 8):
                    if i + 8 <= len(audio_data_bits):
                        sample_bits = audio_data_bits[i:i+8]
                        sample_value = sum(bit * (2**(7-j)) for j, bit in enumerate(sample_bits))
                        # Convert from unsigned 8-bit to signed, then to float
                        # FIXED: Proper conversion from uint8 to float range [-1, 1]
                        signed_sample = (sample_value - 128) / 127.0
                        signed_sample = np.clip(signed_sample, -1.0, 1.0)  # FIXED: Ensure valid range
                        audio_samples_reconstructed.append(signed_sample)
                
                if len(audio_samples_reconstructed) > 0:
                    audio_array = np.array(audio_samples_reconstructed, dtype=np.float32)
                    
                    # Save audio with correct sample rate
                    audio_path = f"{output_dir}improved_received_audio.wav"
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    
                    try:
                        wavfile.write(audio_path, audio_sample_rate, audio_int16)
                        print(f"✅ Audio saved to {audio_path}")
                        print(f"   Sample rate: {audio_sample_rate} Hz")
                        print(f"   Duration: {len(audio_array) / audio_sample_rate:.2f} seconds")
                        print(f"   Downsample factor was: {downsample_factor}")
                    except Exception as e:
                        print(f"❌ Audio save failed: {e}")
                else:
                    print("❌ No audio samples reconstructed")
            else:
                print(f"❌ Not enough bits for audio data")
        
        return True

def test_improved_system():
    """Test the improved audio quality system"""
    
    print("="*60)
    print("TESTING IMPROVED AUDIO QUALITY TRANSMISSION")
    print("="*60)
    
    transmitter = ImprovedAudioTransmitter(
        carrier_freq=1000, sample_rate=8000, quantization_bits=16
    )
    
    os.makedirs("data/received", exist_ok=True)
    
    # Test files
    image_path = "data/matlab_logo.png" if os.path.exists("data/matlab_logo.png") else None
    audio_path = "data/audio_recording.wav" if os.path.exists("data/audio_recording.wav") else None
    
    if not audio_path:
        print("❌ No audio file found for testing")
        return
    
    # Show original specs
    if audio_path:
        sr, audio = wavfile.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        print(f"Original audio: {len(audio)} samples at {sr} Hz")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
    
    # Test transmission
    try:
        output_file, metadata = transmitter.transmit_improved_file(
            image_path=image_path,
            audio_path=audio_path,
            snr_db=18,
            image_quality=4
        )
        
        if output_file:
            success = transmitter.receive_improved_file(output_file, "data/received/")
            
            if success:
                print(f"\n🎉 IMPROVED TRANSMISSION SUCCESSFUL!")
                
                # Compare audio quality
                if os.path.exists("data/received/improved_received_audio.wav"):
                    recv_sr, recv_audio = wavfile.read("data/received/improved_received_audio.wav")
                    print(f"\nAudio quality comparison:")
                    print(f"  Original: {sr} Hz, {len(audio)} samples")
                    print(f"  Received: {recv_sr} Hz, {len(recv_audio)} samples")
                    print(f"  Quality factor: {sr/recv_sr:.1f}x downsample")
                    
                    if recv_sr >= 3000:
                        print("✅ Audio should be intelligible (≥3kHz)")
                    else:
                        print("⚠️  Audio may be muffled (<3kHz)")
                
            else:
                print("❌ Reception failed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_system()