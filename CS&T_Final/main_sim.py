import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from src.channel_model import AWGNChannel
from src.modulation import BPSKModulator, BPSKDemodulator
from src.ber_analysis import calculate_ber, single_ber_test, run_ber_analysis, plot_ber_curve

def test_basic_bpsk():
    print("-" * 50)
    test_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0,])
    print(f"Original bits: {test_bits}")

    modulator = BPSKModulator()
    modulated_signal = modulator.modulate(test_bits)

    demodulator = BPSKDemodulator()
    recovered_bits = demodulator.demodulate(modulated_signal)

    print(f"Recovered bits: {recovered_bits}")

    print(f"Bit errors: {np.sum(test_bits != recovered_bits)}")
    
def test_awgn_channel():
    print("\n" + "-"*50)
    print("AWGN Channel Test")
    
    t = np.linspace(0, .1, 1000)
    test_signal = np.sin(2 * np.pi * 50 * t)
    
    channel = AWGNChannel()
    
    snr_values = [20, 10, 5, 0]
    
    plt.figure(figsize=(12, 8))
    
    for i, snr_db in enumerate(snr_values):
        noisy_signal, noise = channel.add_noise(test_signal, snr_db)
        
        actual_snr = channel.calculate_snr(test_signal, noise)
        
        print(f"SNR {snr_db:2d} dB: Actual = {actual_snr:5.1f} dB")
        
        plt.subplot(2, 2, i+1)
        plt.plot(t[:200], test_signal[:200], 'b-', label='CLean', linewidth=2)
        plt.plot(t[:200], noisy_signal[:200], 'r--', label='Noisy', alpha=0.8)
        plt.title(f'SNR = {snr_db} dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    print("AWGN Test Completed")
    
def test_bpsk_with_noise():
    print("\n" + "-"*50)
    print("BPSK + AWGN Channel Test")
    
    modulator = BPSKModulator(carrier_freq=1000, sample_rate=8000)
    demodulator = BPSKDemodulator(carrier_freq=1000, sample_rate=8000)
    channel = AWGNChannel()
    
    test_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0,])
    snr_values = [15, 10, 5, 0]
    
    print(f"Original bits: {test_bits}")
    print("SNR (dB) | Received Bits  | Errors | BER")
    print("-" * 50)
    
    for snr_db in snr_values:
        modulated_signal = modulator.modulate(test_bits)
        
        noisy_signal, noise = channel.add_noise(modulated_signal, snr_db)
        
        received_bits = demodulator.demodulate(noisy_signal)
        
        errors = np.sum(test_bits != received_bits)
        ber = errors / len(test_bits)
        
        print(f"{snr_db:8d} | {received_bits} | {errors:6d} | {ber:.3f}")
        
    print("BPSK with noise test completed")
    
def simple_ber_analysis():
    print("\n" + "-"*50)
    print("Simple BER Analysis")
    
    modulator = BPSKModulator(carrier_freq=1000, sample_rate=8000)
    demodulator = BPSKDemodulator(carrier_freq=1000, sample_rate=8000)
    
    snr_range = np.arange(0, 16, 2)
    num_bits = 100
    num_trials = 5
    
    print(f"Running BER analysis: {num_bits} bits, {num_trials} trials per SNR")
    
    snr_vals, ber_results = run_ber_analysis(modulator, demodulator, snr_range, num_bits, num_trials)
    
    plot_ber_curve(snr_vals, ber_results, show_theoretical=True)
    
    print("BER Analysis completed")

def main():
    
    test_basic_bpsk()
    test_awgn_channel()
    test_bpsk_with_noise()
    simple_ber_analysis()
    
if __name__ == "__main__":
    main()        
    