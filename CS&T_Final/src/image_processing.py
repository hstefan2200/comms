import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageProcessor:
    def __init__(self):
        pass
    
    def load_image(self, filepath):
        #return Pillow image object
         try:
             img = Image.open(filepath)
             return img
         except Exception as e:
            print(F"Error loading iamge file: {e}")
            return None
            
    def image_to_bits(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        #Convert to grayscale (uses 66% less data)    
        if image.mode != 'L':
            image = image.convert('L')
        
        #convert to numpy array and flatten(from 2D to 1D)
        img_array = np.array(image)
        height, width = img_array.shape
        img_flat = img_array.flatten()
        
        #convert each pixel in array (0-255 grayscale) to 8 bit binary
        bits = []
        for pixel in img_flat:
            pixel_bits = [(pixel >> i) & 1 for i in range(7, -1, -1)]
            bits.extend(pixel_bits)
        
        #Store height and width metadata as 16 bit binary    
        height_bits = [(height >> i) & 1 for i in range(15, -1, -1)]
        width_bits = [(width >> i) & 1 for i in range(15, -1, -1)]
        all_bits = height_bits + width_bits + bits
        
        return np.array(all_bits, dtype=int)
    
    def bits_to_image(self, bits):
        #Extract height and width metadata from first 32 bits
        height_bits = bits[:16]
        height = int(sum(bit * (2**(15-i)) for i, bit in enumerate(height_bits)))
        width_bits = bits[16:32]
        width = int(sum(bit * (2**(15-i)) for i, bit in enumerate(width_bits)))
        #extract image bits (remaining bits)
        image_bits = bits[32:]
        
        image_bits = np.array(image_bits).flatten()
        num_image_bits = image_bits.shape[0]
        
        #verify number of bits (pad if too few, truncate if too many)
        expected_bits = height * width * 8
        if num_image_bits < expected_bits:
            print(f"Warning: Not enough bits for image reconstruction. Expected {expected_bits}, got {len(image_bits)}")
            padding = expected_bits - len(image_bits)
            image_bits = np.concatenate([image_bits, np.zeros(padding, dtype=int)])
        elif num_image_bits > expected_bits:
            image_bits = image_bits[:expected_bits]
        
        #Convert bits back into pixels (steps of 8 bits, then binary to decimal conversion), still in 1d numpy array   
        pixels = []
        for i in range(0, len(image_bits), 8):
            if i + 8 <= len(image_bits):
                pixel_bits = image_bits[i:i+8]
                pixel_value = sum(bit * (2**(7-j)) for j, bit in enumerate(pixel_bits))
                pixels.append(pixel_value)
        
        #reshape into dimensions using metadata
        img_array = np.array(pixels[:height*width], dtype=np.uint8).reshape(height, width)
        #converts back to PIL image object
        image = Image.fromarray(img_array, mode='L')
        
        return image
    
    def save_image(self, image, filepath):
        try:
            image.save(filepath)
            print(f"Image saved successfully to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to save image to {filepath}: {e}")
            return False
        

def test_image_processor():
    print("Testing Image Processor")
    print("-" * 50)
    
    processor = ImageProcessor()
    
    image = processor.load_image('data/matlab_logo.png')
    print(f"Loaded image: {image.size}, mode: {image.mode}")
    
    print("Converting image to bits")      
    bits = processor.image_to_bits(image)
    print(f"generated {len(bits)} bits")
    print(f"first 50 bits: {bits[:50]}")
    
    reconstructed_image = processor.bits_to_image(bits)
    
    print(f"reconstructed image: {reconstructed_image.size}, mode: {reconstructed_image.mode}")
    
    processor.save_image(reconstructed_image, "data/reconstructed_image.png")
        
if __name__ == "__main__":
    test_image_processor()
    
        
    
    
    

    
    
    