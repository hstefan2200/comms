import enum
import numpy as np
from modulation import BPSKDemodulator
from audio_processing import AudioProcessor
from image_processing import ImageProcessor
import matplotlib.pyplot as plt
from scipy.io import wavfile

class Receiver:
    def __init__(self, carrier_freq=1000, sample_rate=8000, quantization_bits=16):
        self.demodulator = BPSKDemodulator(carrier_freq, sample_rate)
        self.audio_processor = AudioProcessor(quantization_bits)
        self.image_processor = ImageProcessor()
        
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        
    def load_received_signal(self, audio_file_path):
        print(f"Loading received signal from {audio_file_path}")
        try:
            sample_rate, audio_data = wavfile.read(audio_file_path)
            
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            print(f"loaded {len(audio_data)} samples at {sample_rate}")
            
            return audio_data, sample_rate

        except Exception as e:
            print(f"Error loading received signal: {e}")
            return None, None
        
    def demodulate_signal(self, received_signal):
        print("-"*50)
        print("Demodulating received signal")
        
        max_amplitude = np.max(np.abs(received_signal))
        if max_amplitude > 0:
            normalized_signal = received_signal / max_amplitude
        else:
            normalized_signal = received_signal
            
        #using BPSKDemodulator
        recovered_bits = self.demodulator.demodulate(normalized_signal)
        
        print(f"Recovered {len(recovered_bits)} bits from signal")
        
        return recovered_bits
    
    def parse_packet_header(self, bits):
        #extract the metadata to assist with reconstruction
        if len(bits) < 32:
            print("WARNING: Less than 32 bits, header is first 32 bits")
            return None, None
        
        #Extract image length if it exists (first 16 bits)
        image_length_bits = bits[:16]
        image_length = sum(bit * (2**(15-i)) for i, bit in enumerate(image_length_bits))
        
        #Extract audio length if it exists (second 16 bits)
        audio_length_bits = bits[16:32]
        audio_length = sum(bit * (2**(15-i)) for i, bit in enumerate(audio_length_bits))
        
        #reconstruct metadata
        metadata = {
            'image_length': image_length,
            'audio_length': audio_length,
            'header_length': 32,
            'total_expected': 32 + image_length + audio_length
        }
        
        print(f"Packet metadata - Image: {image_length} bits, Audio: {audio_length} bits")
        
        return metadata, bits[32:]
    
    def extract_data(self, data_bits, metadata):
        image_bits = None
        audio_bits = None
        
        current_position = 0
                
        # Extract image bits if present
        if metadata['image_length'] > 0:
            end_position = current_position + metadata['image_length']
            if end_position <= len(data_bits):
                image_bits = data_bits[current_position:end_position]
                current_position = end_position
                print(f"Extracted {len(image_bits)} image bits")
            else:
                print("Warning: Not enough bits for complete image data")
                
        # Extract audio bits if present
        if metadata['audio_length'] > 0:
            end_position = current_position + metadata['audio_length']
            if end_position <= len(data_bits):
                audio_bits = data_bits[current_position:end_position]
                print(f"Extracted {len(audio_bits)} audio bits")
            else:
                print("Warning: Not enough bits for complete audio data")
        
        return image_bits, audio_bits
    
    def reconstruct_image(self, image_bits, output_path=None):
        if image_bits is None or len(image_bits) == 0:
            print("No image data received")
            return None
        print("Reconstructing image")
        try:
            reconstructed_image = self.image_processor.bits_to_image(image_bits)
            
            if output_path:
                success = self.image_processor.save_image(reconstructed_image, output_path)
                if success:
                    print(f"Image save to {output_path}")
                else:
                    print("Failed to save image")
                    return None
                
            return reconstructed_image
        except Exception as e:
            print(f"Error reconstructing image: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def reconstruct_audio(self, audio_bits, original_metadata=None, output_path=None):
        if audio_bits is None or len(audio_bits) == 0:
            print("No audio data received")
            return None
        
        print("Reconstructing audio data")
        try:
            if original_metadata is None:
                bits_per_sample = self.audio_processor.quantization_bits
                num_samples = len(audio_bits) // bits_per_sample
                
                metadata = {
                    'num_samples': num_samples,
                    'quantization_bits': bits_per_sample,
                    'downsample_factor': 1,
                    'original_range': [-1.0, 1.0]
                }
            else:
                metadata = original_metadata
            
            reconstructed_audio = self.audio_processor.bits_to_audio(audio_bits, metadata)
            
            if output_path:
                sample_rate = self.sample_rate // metadata.get('downsample_factor', 1)
                try:
                    self.audio_processor.save_audio_file(reconstructed_audio, output_path, sample_rate)
                    print(f"Audio saved to {output_path}")
                except Exception as e:
                    print(f"Failed to save audio: {e}")
                    return None
                
            return reconstructed_audio
        
        except Exception as e:
            print(f"Error reconstructing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def calculate_bit_error_rate(self, original_bits, received_bits):
        if len(original_bits) != len(received_bits):
            min_len = min(len(original_bits), len(received_bits))
            original_bits = original_bits[:min_len]
            received_bits = received_bits[:min_len]
            print(f"WARNGIN: bit length mismatch, truncating to {min_len} bits")
            
        errors = np.sum(original_bits != received_bits)
        ber = errors / len(original_bits)
        
        print(f"BER: {ber:.6f} ({errors}/{len(original_bits)})")
        
        return ber, errors
            
            
    def receive_and_process(self, audio_file_path, original_bits=None, output_dir="data/received/"):
        #putting everything together here
        print("-"*50)
        print("Starting data reception")
        
        #load received signal
        received_signal, sample_rate = self.load_received_signal(audio_file_path)
        if received_signal is None:
            print("Error loading signal")
            return None

        #demodulate signal
        recovered_bits = self.demodulate_signal(received_signal)
        
        #parse packet header to determine what data types are present
        metadata, data_bits = self.parse_packet_header(recovered_bits)
        if metadata is None:
            print("Failed to parse packet header")
            return None
        
        has_image = metadata['image_length'] > 0
        has_audio = metadata['audio_length'] > 0
        
        print(f"\nImage data detected: {'yes' if has_image else 'no'}")
        print(f"Audio data detected: {'yes' if has_audio else 'no'}")
        
        if not has_image and not has_audio:
            print("WARNING no data detected in packet")
            return None
        
        #extract data
        image_bits, audio_bits = self.extract_data(data_bits, metadata)
        
        #reconstruct image if present
        reconstructed_image = None
        if has_image and image_bits is not None:
            print(f"\nProcessing image data ({len(image_bits)} bits)")
            image_filename = f"{output_dir}received_image.png"
            reconstructed_image = self.reconstruct_image(image_bits, image_filename)
            if reconstructed_image:
                print("Image reconstruction failed")
            else:
                print("Image reconstruction failed")
            
        #reconstruct audio if present
        reconstructed_audio = None
        if has_audio and audio_bits is not None:
            print(f"\nProcessing audio data ({len(audio_bits)} bits)")
            audio_filename = f"{output_dir}received_audio.wav"
            reconstructed_audio = self.reconstruct_audio(audio_bits, output_path=audio_filename)
            if reconstructed_audio:
                print("audio reconstruction failed")
            else:
                print("audio reconstruction failed")
        
        #find BER
        if original_bits is not None:
            ber, errors = self.calculate_bit_error_rate(original_bits, recovered_bits)
            print(f"BER: {ber:.6f}")
            print(f"Success rate: {(1-ber)*100:.4f}%")
            
            
def test_receiver():
    print("Testing Automatic Receiver Implementation")
    print("-"*50)
    
    receiver = Receiver(carrier_freq=1000, sample_rate=8000, quantization_bits=16)
    
    import os
    os.makedirs("data/received", exist_ok=True)
    
    # List of test files
    test_files = [
        "data/transmitted_image.wav",
        "data/transmitted_audio.wav", 
        "data/transmitted_combined.wav"
    ]
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n{i}. Testing Reception: {test_file}")
        print("-" * 50)
        
        try:
            # Check if file exists
            if not os.path.exists(test_file):
                print(f"⚠ Test file {test_file} not found. Skipping...")
                continue
                
            # Automatically detect and process whatever data is in the file
            results = receiver.receive_and_process(
                audio_file_path=test_file,
                output_dir="data/received/"
            )
            
            if results:
                data_types = results['data_types']
                print(f"\nTest Results for {test_file}:")
                
                if data_types['has_image']:
                    status = "SUCCESS" if (results['reconstructed_image'] is not None) else "FAILED"
                    print(f"   Image: {status}")
                    
                if data_types['has_audio']:
                    status = "SUCCESS" if (results['reconstructed_audio'] is not None) else "FAILED"
                    print(f"   Audio: {status}")
                    
                if not data_types['has_image'] and not data_types['has_audio']:
                    print("   No data detected")
                    
            else:
                print("Reception completely failed")
                
        except Exception as e:
            print(f"Test failed with error: {e}")
    
    print(f"\n" + "-"*60)
    print("Receiver testing completed")     
        
        
if __name__ == "__main__":
    test_receiver()