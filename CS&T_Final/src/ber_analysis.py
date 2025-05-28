import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import random
from src.channel_model import AWGNChannel

def generate_random_bits(num_bits):
    """Generate random binary numpy array"""
    return np.random.randint(0, 2, num_bits, dtype=int)

def calculate_ber(original_bits, received_bits):
    if len(original_bits) != len(received_bits):
        min_len = min(len(original_bits), len(received_bits))
        original_bits = original_bits[:min_len]
        received_bits = received_bits[:min_len]
        print(f"Warning: Length mismatch. Using {min_len} bits for comparison")
        
    errors = np.sum(original_bits != received_bits)
    ber = errors / len(original_bits)
    
    return ber, errors

def single_ber_test(modulator, demodulator, channel, test_bits, snr_db):
    modulated_signal = modulator.modulate(test_bits)
    noisy_signal, noise = channel.add_noise(modulated_signal, snr_db)
    received_bits = demodulator.demodulate(noisy_signal)
    ber, errors = calculate_ber(test_bits, received_bits)
    
    return ber, received_bits

def run_ber_analysis(modulator, demodulator, snr_range, num_bits=1000, num_trials=10):
    channel = AWGNChannel()
    ber_results = []
    
    print("Running BER analysis...")
    print("SNR (dB) | BER    | Errors")
    print("-" * 30)
    
    for snr_db in snr_range:
        total_errors = 0
        total_bits = 0
        
        for trial in range(num_trials):
            test_bits = generate_random_bits(num_bits)
            
            ber, received_bits = single_ber_test(modulator, demodulator, channel, test_bits, snr_db)
            
            _, errors = calculate_ber(test_bits, received_bits)
            total_errors += errors
            total_bits += len(test_bits)
            
        avg_ber = total_errors / total_bits
        ber_results.append(avg_ber)
        
        print(f"{snr_db:8.1f} | {avg_ber:.6f} | {total_errors}/{total_bits}")
        
    return snr_range, ber_results

def plot_ber_curve(snr_range, ber_results, show_theoretical=True):
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_range, ber_results, 'bo-', label='Simulated BER', 
                markersize=6, linewidth=2)
    
    if show_theoretical:
        snr_linear = 10**(np.array(snr_range) / 10)
        theoretical_ber = 0.5 * erfc(np.sqrt(snr_linear))
        plt.semilogy(snr_range, theoretical_ber, 'r--', 
                    label='Theoretical BPSK', linewidth=2)
    
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BPSK Performance in AWGN Channel')
    plt.legend()
    plt.ylim([1e-5, 1])
    plt.xlim([min(snr_range), max(snr_range)])
    plt.show()