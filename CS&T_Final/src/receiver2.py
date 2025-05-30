import numpy as np
import os
from PIL import Image
from scipy.io import wavfile
from src.modulation import BPSKDemodulator
from src.audio_processing import AudioProcessor
from src.image_processing import ImageProcessor


class ImprovedReceiver:
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000, quantization_bits=16):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.quantization_bits = quantization_bits
        
        self.demodulator = BPSKDemodulator(carrier_freq, sample_rate, symbol_rate)
        self.audio_processor = AudioProcessor(quantization_bits)
        self.image_processor = ImageProcessor()
        
        print(f"Receiver initialized - Carrier: {carrier_freq}Hz, Sample Rate: {sample_rate}Hz")
    
    def load_received_signal(self, audio_file_path):
        print(f"Loading received signal from {audio_file_path}")
        try:
            sample_rate, audio_data = wavfile.read(audio_file_path)
            
            #wavfile.read() returns raw integer data from the .wav file, we need to normalize that back to -1, +1 (ie: this will handlke 16 and 32 bit signed data, and 8 bit unsigned, and reduce the huge ranges to 2)
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            
            #take left channel only if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"Error loading received signal: {e}")
            return None, None
    
    def demodulate_signal(self, received_signal):
        print("-"*50)
        print("Demodulating received signal...")
        
        #normalize
        max_amplitude = np.max(np.abs(received_signal))
        if max_amplitude > 0:
            normalized_signal = received_signal / max_amplitude
        else:
            normalized_signal = received_signal
            print("Warning: Zero amplitude signal detected")

        recovered_bits = self.demodulator.demodulate(normalized_signal)
        
        print(f"Recovered {len(recovered_bits)} bits from signal")
        return recovered_bits
    
    def parse_packet_header(self, bits):
        if len(bits) < 48:
            print(f"Not enough bits for packet header (need 48, got {len(bits)})")
            return None, None
        
        header = bits[:48]
        data_bits = bits[48:]
        
        #extract header metadata
        try:
            has_image = bool(header[0])
            has_audio = bool(header[1])
            img_w = sum(bit * (2**(9-i)) for i, bit in enumerate(header[2:12]))
            img_h = sum(bit * (2**(9-i)) for i, bit in enumerate(header[12:22]))
            audio_samples_scaled = sum(bit * (2**(11-i)) for i, bit in enumerate(header[22:34]))
            audio_samples = audio_samples_scaled * 10
            downsample_factor = sum(bit * (2**(3-i)) for i, bit in enumerate(header[34:38]))
            if downsample_factor == 0:
                downsample_factor = 1
            audio_sr_scaled = sum(bit * (2**(9-i)) for i, bit in enumerate(header[38:48]))
            audio_sample_rate = audio_sr_scaled * 100
            if audio_sample_rate == 0:
                audio_sample_rate = 8000
            
            metadata = {
                'has_image': has_image,
                'has_audio': has_audio,
                'image_width': img_w,
                'image_height': img_h,
                'audio_samples': audio_samples,
                'audio_downsample': downsample_factor,
                'audio_sample_rate': audio_sample_rate,
                'header_length': 48
            }
            
            print(f"Parsed packet header:")
            print(f"  Image: {has_image} ({img_w}x{img_h})")
            print(f"  Audio: {has_audio} ({audio_samples} samples at {audio_sample_rate} Hz)")
            print(f"  Downsample factor: {downsample_factor}")
            print(f"  Data bits available: {len(data_bits)}")
            
            return metadata, data_bits
            
        except Exception as e:
            print(f"Error parsing header: {e}")
            return None, None
    
    def extract_data(self, data_bits, metadata):
        image_bits = None
        audio_bits = None
        current_pos = 0
        
        print(f"Extracting data from {len(data_bits)} available bits...")
        
        #Extract image bits if present
        if metadata['has_image'] and metadata['image_width'] > 0 and metadata['image_height'] > 0:
            bits_needed = metadata['image_width'] * metadata['image_height'] * 4  # 4 bits per pixel
            # print(f"Image needs {bits_needed} bits")
            
            if current_pos + bits_needed <= len(data_bits):
                image_bits = data_bits[current_pos:current_pos + bits_needed]
                current_pos += bits_needed
                print(f"Extracted {len(image_bits)} image bits")
            else:
                print(f"Not enough bits for image: need {bits_needed}, have {len(data_bits) - current_pos}")
        
        # Extract audio bits if present
        if metadata['has_audio'] and metadata['audio_samples'] > 0:
            audio_bits_needed = metadata['audio_samples'] * 8  # 8-bit quantization
            # print(f"Audio needs {audio_bits_needed} bits")
            
            if current_pos + audio_bits_needed <= len(data_bits):
                audio_bits = data_bits[current_pos:current_pos + audio_bits_needed]
                current_pos += audio_bits_needed
                print(f"Extracted {len(audio_bits)} audio bits")
            else:
                print(f"Not enough bits for audio: need {audio_bits_needed}, have {len(data_bits) - current_pos}")
        
        return image_bits, audio_bits
    
    def reconstruct_image(self, image_bits, metadata, output_path=None):
        if image_bits is None or len(image_bits) == 0:
            print("No image data to reconstruct")
            return None
        
        print("Reconstructing image...")
        try:
            img_w = metadata['image_width']
            img_h = metadata['image_height']
            
            expected_bits = img_w * img_h * 4
            if len(image_bits) != expected_bits:
                print(f"Image bit count mismatch: expected {expected_bits}, got {len(image_bits)}")
                # Pad or truncate as needed
                if len(image_bits) < expected_bits:
                    image_bits = np.pad(image_bits, (0, expected_bits - len(image_bits)), 'constant')
                else:
                    image_bits = image_bits[:expected_bits]
            
            #Convert 4-bit pixels back to 8-bit
            pixels = []
            for i in range(0, len(image_bits), 4):
                if i + 4 <= len(image_bits):
                    pixel_bits = image_bits[i:i+4]
                    pixel_value = sum(bit * (2**(3-j)) for j, bit in enumerate(pixel_bits))
                    pixels.append(pixel_value * 16)
            
            if len(pixels) == img_w * img_h:
                img_array = np.array(pixels, dtype=np.uint8).reshape(img_h, img_w)
                reconstructed_image = Image.fromarray(img_array, mode='L')
                
                if output_path:
                    reconstructed_image.save(output_path)
                    print(f"Image saved to {output_path}")
                
                return reconstructed_image
            else:
                print(f"Pixel count mismatch: expected {img_w * img_h}, got {len(pixels)}")
                return None
                
        except Exception as e:
            print(f"Error reconstructing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def reconstruct_audio(self, audio_bits, metadata, output_path=None):
        if audio_bits is None or len(audio_bits) == 0:
            print("No audio data to reconstruct")
            return None
        
        print("Reconstructing audio...")
        try:
            expected_bits = metadata['audio_samples'] * 8
            if len(audio_bits) != expected_bits:
                print(f"Audio bit count mismatch: expected {expected_bits}, got {len(audio_bits)}")
                # Pad or truncate as needed
                if len(audio_bits) < expected_bits:
                    audio_bits = np.pad(audio_bits, (0, expected_bits - len(audio_bits)), 'constant')
                else:
                    audio_bits = audio_bits[:expected_bits]
            
            # Convert bits back to samples
            audio_samples = []
            for i in range(0, len(audio_bits), 8):
                if i + 8 <= len(audio_bits):
                    sample_bits = audio_bits[i:i+8]
                    sample_value = sum(bit * (2**(7-j)) for j, bit in enumerate(sample_bits))
                    signed_sample = (sample_value - 128) / 127.0
                    signed_sample = np.clip(signed_sample, -1.0, 1.0)  # Ensure valid range
                    audio_samples.append(signed_sample)
            
            if len(audio_samples) > 0:
                audio_array = np.array(audio_samples, dtype=np.float32)
                
                if output_path:
                    #Ensure valid sample rate
                    sample_rate = max(metadata['audio_sample_rate'], 1000)  # Minimum 1kHz
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    
                    try:
                        wavfile.write(output_path, sample_rate, audio_int16)
                        print(f"Audio saved to {output_path}")
                        print(f"   Sample rate: {sample_rate} Hz")
                        print(f"   Duration: {len(audio_array) / sample_rate:.2f} seconds")
                        print(f"   Downsample factor was: {metadata['audio_downsample']}")
                            
                    except Exception as e:
                        print(f"Audio save failed: {e}")
                        return None
                
                return audio_array
            else:
                print("No audio samples reconstructed")
                return None
                
        except Exception as e:
            print(f"Error reconstructing audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_bit_error_rate(self, original_bits, received_bits):
        if original_bits is None or received_bits is None:
            print("Cannot calculate BER - missing bit data")
            return None, None
        
        original_bits = np.array(original_bits)
        received_bits = np.array(received_bits)
        
        if len(original_bits) != len(received_bits):
            min_len = min(len(original_bits), len(received_bits))
            original_bits = original_bits[:min_len]
            received_bits = received_bits[:min_len]
            print(f"Bit length mismatch, truncating to {min_len} bits")
        
        if len(original_bits) == 0:
            print("No bits to compare")
            return None, None
        
        errors = np.sum(original_bits != received_bits)
        ber = errors / len(original_bits)
        
        print(f"BER Analysis:")
        print(f"  Bit errors: {errors}/{len(original_bits)}")
        print(f"  BER: {ber:.6f}")
        print(f"  Success rate: {(1-ber)*100:.4f}%")
        
        return ber, errors
    
    def receive_file(self, audio_file_path, original_bits=None, output_dir="data/received/"):
        #Putting everything above together
        print("-"*60)
        print("Receiving file")

        os.makedirs(output_dir, exist_ok=True)
        
        received_signal, sample_rate = self.load_received_signal(audio_file_path)
        if received_signal is None:
            print("Failed to load received signal")
            return None
        
        # Demodulate signal
        recovered_bits = self.demodulate_signal(received_signal)
        if len(recovered_bits) == 0:
            print("No bits recovered from signal")
            return None
        
        # Parse header
        metadata, data_bits = self.parse_packet_header(recovered_bits)
        if metadata is None:
            print("Failed to parse packet header")
            return None
        
        #Extract data
        image_bits, audio_bits = self.extract_data(data_bits, metadata)
        
        #store results
        results = {
            'metadata': metadata,
            'recovered_bits': recovered_bits,
            'reconstructed_image': None,
            'reconstructed_audio': None,
            'ber': None,
            'errors': None
        }
        
        #Reconstruct image if present
        if metadata['has_image'] and image_bits is not None:
            print(f"Processing image data ({len(image_bits)} bits)")
            image_filename = f"{output_dir}received_image.png"
            results['reconstructed_image'] = self.reconstruct_image(
                image_bits, metadata, image_filename
            )
        
        #reconstruct audio if present
        if metadata['has_audio'] and audio_bits is not None:
            print(f"Processing audio data ({len(audio_bits)} bits)")
            audio_filename = f"{output_dir}received_audio.wav"
            results['reconstructed_audio'] = self.reconstruct_audio(
                audio_bits, metadata, audio_filename
            )
        
        # Calculate BER
        if original_bits is not None:
            print(f"Calculating Bit Error Rate...")
            results['ber'], results['errors'] = self.calculate_bit_error_rate(
                original_bits, recovered_bits
            )
        
        return results


# def test_receiver():
#     print("-"*60)
#     print("TESTING IMPROVED RECEIVER")
    
#     receiver = ImprovedReceiver(
#         carrier_freq=1000,
#         sample_rate=8000,
#         symbol_rate=2000,
#         quantization_bits=16
#     )
    
#     os.makedirs("data/received", exist_ok=True)

#     test_files = [
#         "data/transmitted_signal.wav",
#         "data/test_transmission.wav",
#         "data/improved_transmission.wav",
#         "data/large_file_transmission.wav"
#     ]
    
#     successful_tests = 0
#     total_tests = 0
    
#     for i, test_file in enumerate(test_files, 1):
#         print(f"\n{i}. Testing reception: {test_file}")
#         print("-" * 50)
        
#         if not os.path.exists(test_file):
#             print(f"⚠️  Test file {test_file} not found. Skipping...")
#             continue
        
#         total_tests += 1
        
#         try:
#             results = receiver.receive_file(
#                 audio_file_path=test_file,
#                 output_dir="data/received/"
#             )
            
#             if results:
#                 print(f"Reception test {i} completed successfully")
#                 successful_tests += 1
#             else:
#                 print(f"Reception test {i} failed")
                
#         except Exception as e:
#             print(f"Test {i} failed with error: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print("RECEIVER TESTING COMPLETED")
#     print(f"Results: {successful_tests}/{total_tests} tests passed")
#     if total_tests > 0:
#         print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
#     else:
#         print("No test files found - create transmitted files first")


# if __name__ == "__main__":
#     test_receiver()