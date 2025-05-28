import numpy as np

class AWGNChannel:
    def __init__(self):
        pass
    
    def add_noise(self, signal, snr_db):
        signal_power = np.mean(signal**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
        noisy_signal = signal + noise
        
        return noisy_signal, noise
    
    def get_noise_power(self, signal, snr_db):
        signal_power = np.mean(signal**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        return noise_power
    
    def calculate_snr(self, signal, noise):
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
    
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db