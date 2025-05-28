import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from scipy.io import wavfile

class AudioProcessor:
    def __init__(self, quantization_bits=8):
        self.quantization_bits = quantization_bits
        self.max_val = 2**(quantization_bits - 1) - 1
        
    def load_audio_file(self, filepath):
        try:
            sample_rate, audio_data = wavfile.read(filepath)
            
            #Handle different audio bit rates and normalize to [-1, +1]
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
             #Only take left channel (if stereo)    
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
                
            print(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
            print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
            print(f"Audio range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")
            
            return audio_data, sample_rate
        
        except Exception as e:
            print(F"Error loading audio file: {e}")
            return None, None
        
    def quantize_audio(self, audio_data):
        #"Normalize" to -1, +1 (handling volume discrepencies)
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        #Quantize to 8 bit --> multiply by 127
        quantized = np.round(audio_clipped * self.max_val).astype(int)
        #Round from float to int
        quantized_float = quantized.astype(np.float32) / self.max_val
        
        return quantized_float, quantized
    
    def audio_to_bits(self, audio_data, downsample_factor=1):
        #downsample if desired (default is no downsampling)
        if downsample_factor > 1:
            audio_data = audio_data[::downsample_factor]
            print(f"Downsampled audio to {len(audio_data)} samples")
            
        quantized_audio, quantized_int = self.quantize_audio(audio_data)
        #Convert to unsigned bits
        if self.quantization_bits == 8:
            unsigned_data = (quantized_int + 128).astype(np.uint8)  # Add 128, not 32768!
            bits_per_sample = 8
        else:  # 16 bits
            unsigned_data = (quantized_int + 32768).astype(np.uint16)
            bits_per_sample = 16
        
        #convert each sample to binary    
        bits_list = []
        for sample in unsigned_data:
            binary_str = format(sample, f'0{bits_per_sample}b')
            
            sample_bits = [int(bit) for bit in binary_str]
            bits_list.extend(sample_bits)
            
        bits = np.array(bits_list, dtype=int)
        
        metadata = {
            'num_samples': len(audio_data),
            'quantization_bits': self.quantization_bits,
            'downsample_factor': downsample_factor,
            'original_range': [np.min(audio_data), np.max(audio_data)]
        }    
        
        return bits, metadata
        
    def bits_to_audio(self, bits, metadata):
        #reverse everything to return bits to signed numpy array representing audio samples
        bits_per_sample = metadata['quantization_bits']
        num_samples = metadata['num_samples']
        
        #convert from single array of bits to nested array of samples (still in binary)
        if len(bits) != num_samples * bits_per_sample:
            print(f"Warning: Expected {num_samples * bits_per_sample} bits, got {len(bits)}")
            expected_bits = num_samples * bits_per_sample
            #handle length differences
            if len(bits) > expected_bits:
                bits = bits[:expected_bits]
            else:
                bits = np.pad(bits, (0, expected_bits - len(bits)), 'constant')
        
        bits_2d = bits.reshape(num_samples, bits_per_sample)
        
        #convert binary back to integers (still unsigned) 0-255
        reconstructed_samples = []
        for sample_bits in bits_2d:
            # Convert binary array to integer
            binary_str = ''.join(sample_bits.astype(str))
            sample_val = int(binary_str, 2)
            reconstructed_samples.append(sample_val)
        
        reconstructed_samples = np.array(reconstructed_samples)
        
        #revert back to signed values -128 - +127
        if bits_per_sample == 8:
            signed_samples = reconstructed_samples.astype(int) - 128
            max_val = 127
        else:  # 16 bits
            signed_samples = reconstructed_samples.astype(int) - 32768
            max_val = 32767
        
        reconstructed_audio = signed_samples.astype(np.float32) / max_val
        
        print(f"Reconstructed {len(reconstructed_audio)} audio samples")
        
        return reconstructed_audio
    
    def save_adio_file(self, audio_data, filepath, sample_rate=8000):
        #convert numpy array of audio samples back into wav file and save
        audio_int16 = (audio_data * 32767).astype(np.int16)
        try:
            wavfile.write(filepath, sample_rate, audio_int16)
            print(f"Successfully saved audio to {filepath}")
        except Exception as e:
            print(f"Error saving audio file: {e}")


def test_audio_processing():
    print("Testing audio processing")
    print("-" * 50)
    
    processor = AudioProcessor(quantization_bits=16)
    
    audio_data, sample_rate = processor.load_audio_file("data/audio_recording.wav")
    
    downsampling_factor = 1
    
    bits, metadata = processor.audio_to_bits(audio_data, downsample_factor=downsampling_factor)
    
    reconstructed_audio = processor.bits_to_audio(bits, metadata)
    
    # debug
    print(f"Audio samples: {len(audio_data)}")
    print(f"Expected bits: {len(audio_data) * 8}")
    print(f"Actual bits: {len(bits)}")
    print(f"Bits per sample: {len(bits) / len(audio_data)}")
    
    if downsampling_factor > 1:
        effective_sample_rate = sample_rate // downsampling_factor
    else:
        effective_sample_rate = sample_rate
    
    processor.save_adio_file(reconstructed_audio, "data/reconstructed_audio2.wav", effective_sample_rate)
    
    print("Audio Processing test completed")
    
if __name__ == "__main__":
    test_audio_processing()