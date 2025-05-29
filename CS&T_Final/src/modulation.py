import numpy as np

class BPSKModulator:
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000):
        self.fc = carrier_freq
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = self.fs // self.symbol_rate
    
    def modulate(self, bits):
        #Map 1 = +1, and 0 = -1 (bits to symbols)
        symbols = 2 * bits - 1
        
        #Generate time vector
        # samples_per_symbol = self.fs // 1000
        t = np.arange(len(symbols) * self.samples_per_symbol) / self.fs
        
        #Generate modulated signal
        signal = np.repeat(symbols, self.samples_per_symbol) * np.cos(2 * np.pi * self.fc * t)
        
        return signal
    
class BPSKDemodulator:
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000):
        self.fc = carrier_freq
        self.fs = sample_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = self.fs // self.symbol_rate
        
    def demodulate(self, signal):
        # samples_per_symbol = self.fs //1000
        t = np.arange(len(signal)) / self.fs
        
        #Coherent detection using correlation
        reference = np.cos(2 * np.pi * self.fc * t)
        demod_signal = signal * reference
        
        num_symbols = len(signal) // self.samples_per_symbol
        bits = []
        
        for i in range(num_symbols):
            start_index = i * self.samples_per_symbol
            end_index = (i + 1) * self.samples_per_symbol
            symbol_average = np.mean(demod_signal[start_index:end_index])
            bits.append(1 if symbol_average > 0 else 0)
        
        return np.array(bits)