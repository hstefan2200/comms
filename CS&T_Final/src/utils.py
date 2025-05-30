import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import os
from PIL import Image


def estimate_transmission_time(num_bits, symbol_rate=2000, samples_per_symbol=4, sample_rate=8000):
    #Each bit becomes a symbol, each symbol has samples_per_symbol samples
    total_samples = num_bits * samples_per_symbol
    time_seconds = total_samples / sample_rate
    return time_seconds

def calculate_theoretical_ber(snr_db_values):
    #https://www.mathworks.com/matlabcentral/fileexchange/25922-ber-of-bpsk-in-awgn-channel
    snr_linear = 10**(np.array(snr_db_values) / 10)
    
    # for BPSK: BER = Q(sqrt(2*Eb/N0)) = 0.5 * erfc(sqrt(Eb/N0))  
    # where Eb/N0 = SNR for BPSK (since 1 bit per symbol)
    theoretical_ber = 0.5 * erfc(np.sqrt(snr_linear))
    
    # Debug
    if len(snr_db_values) <= 10: 
        for snr_db, snr_lin, ber in zip(snr_db_values, snr_linear, theoretical_ber):
            print(f"  SNR: {snr_db} dB = {snr_lin:.1f} linear → BER: {ber:.2e}")
    
    return theoretical_ber

def plot_signal_analysis(signal, sample_rate=8000, title="Signal Analysis"):
    
    plt.figure(figsize=(15, 10))
    
    # Time domain plot (first 1000 samples)
    plt.subplot(2, 3, 1)
    samples_to_plot = min(1000, len(signal))
    t = np.arange(samples_to_plot) / sample_rate
    plt.plot(t, signal[:samples_to_plot], 'b-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{title} - Time Domain')
    plt.grid(True, alpha=0.3)
    
    # Frequency domain plot
    plt.subplot(2, 3, 2)
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    spectrum = np.abs(np.fft.fft(signal))
    plt.plot(freqs[:len(freqs)//2], spectrum[:len(freqs)//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'{title} - Frequency Domain')
    plt.grid(True, alpha=0.3)
    
    # Power spectral density
    plt.subplot(2, 3, 3)
    f, psd = plt.psd(signal, NFFT=1024, Fs=sample_rate)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title(f'{title} - PSD')
    
    # Histogram
    plt.subplot(2, 3, 4)
    plt.hist(signal, bins=50, alpha=0.7, density=True)
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.title('Amplitude Distribution')
    plt.grid(True, alpha=0.3)
    
    # Signal statistics
    plt.subplot(2, 3, 5)
    stats_text = f"""Signal Statistics:

    Length: {len(signal)} samples
    Duration: {len(signal)/sample_rate:.3f} seconds
    Sample Rate: {sample_rate} Hz

    RMS: {np.sqrt(np.mean(signal**2)):.6f}
    Peak: {np.max(np.abs(signal)):.6f}
    Mean: {np.mean(signal):.6f}
    Std Dev: {np.std(signal):.6f}

    Dynamic Range: {20*np.log10(np.max(np.abs(signal))/np.sqrt(np.mean(signal**2))):.1f} dB
    """
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top', 
             fontfamily='monospace', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Statistics')

def compare_signals(original, received, sample_rate=8000, title="Signal Comparison"):
    #compare input and reconstructed signals
    plt.figure(figsize=(12, 8))
    
    # Align signals by length
    min_len = min(len(original), len(received))
    orig_truncated = original[:min_len]
    recv_truncated = received[:min_len]
    
    # Time domain comparison
    plt.subplot(2, 2, 1)
    samples_to_plot = min(500, min_len)
    t = np.arange(samples_to_plot) / sample_rate
    plt.plot(t, orig_truncated[:samples_to_plot], 'b-', label='Original', linewidth=1)
    plt.plot(t, recv_truncated[:samples_to_plot], 'r--', label='Received', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{title} - Waveform Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Frequency domain comparison
    plt.subplot(2, 2, 2)
    freqs = np.fft.fftfreq(min_len, 1/sample_rate)
    orig_spectrum = np.abs(np.fft.fft(orig_truncated))
    recv_spectrum = np.abs(np.fft.fft(recv_truncated))
    
    plt.plot(freqs[:len(freqs)//2], orig_spectrum[:len(freqs)//2], 'b-', label='Original')
    plt.plot(freqs[:len(freqs)//2], recv_spectrum[:len(freqs)//2], 'r--', label='Received', alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error signal
    plt.subplot(2, 2, 3)
    error = recv_truncated - orig_truncated
    t_error = np.arange(min(500, len(error))) / sample_rate
    plt.plot(t_error, error[:len(t_error)], 'g-', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Error Amplitude')
    plt.title('Error Signal (Received - Original)')
    plt.grid(True, alpha=0.3)
    
    # Statistics comparison
    plt.subplot(2, 2, 4)
    stats_text = f"""Comparison Statistics:

    Original Signal:
    RMS: {np.sqrt(np.mean(orig_truncated**2)):.6f}
    Peak: {np.max(np.abs(orig_truncated)):.6f}
    
    Received Signal:
    RMS: {np.sqrt(np.mean(recv_truncated**2)):.6f}
    Peak: {np.max(np.abs(recv_truncated)):.6f}

    Error Metrics:
    MSE: {np.mean(error**2):.6f}
    SNR: {10*np.log10(np.mean(orig_truncated**2)/np.mean(error**2)):.1f} dB
    Correlation: {np.corrcoef(orig_truncated, recv_truncated)[0,1]:.4f}
    """
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
             fontfamily='monospace', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Comparison Metrics')
    
    plt.tight_layout()
    plt.show()


def create_ber_plot(snr_values, simulated_ber, show_theoretical=True, save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Simulated BER
    plt.semilogy(snr_values, simulated_ber, 'bo-', label='Simulated BER', 
                markersize=6, linewidth=2)
    
    # Theoretical BER for comparison
    if show_theoretical:
        theoretical_ber = calculate_theoretical_ber(snr_values)
        plt.semilogy(snr_values, theoretical_ber, 'r--', 
                    label='Theoretical BPSK', linewidth=2)
    
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BPSK Performance in AWGN Channel')
    plt.legend()
    plt.ylim([1e-6, 1])
    plt.xlim([min(snr_values), max(snr_values)])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"BER plot saved to {save_path}")
    
    plt.show()

def check_file_exists(filepath, description="File"):
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"{description} found: {filepath} ({file_size} bytes)")
        return True
    else:
        print(f"{description} not found: {filepath}")
        return False

def create_directories(dir_list):
    """Create multiple directories if they don't exist"""
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def print_system_info(carrier_freq, sample_rate, symbol_rate, quantization_bits):
    """Print formatted system information"""
    print("="*60)
    print("COMMUNICATION SYSTEM PARAMETERS")
    print("="*60)
    print(f"Carrier Frequency:     {carrier_freq:,} Hz")
    print(f"Sample Rate:           {sample_rate:,} Hz") 
    print(f"Symbol Rate:           {symbol_rate:,} Hz")
    print(f"Samples per Symbol:    {sample_rate // symbol_rate}")
    print(f"Quantization:          {quantization_bits} bits")
    print(f"Theoretical Bandwidth: ~{symbol_rate * 2:,} Hz")
    print(f"Data Rate:             {symbol_rate:,} bps")
    print("="*60)


def print_transmission_summary(metadata, transmission_time, snr_db):
    """Print formatted transmission summary"""
    print("-"*60)
    print("Summary:")
    
    if metadata.get('has_image'):
        print(f"Image: {metadata['image_width']}x{metadata['image_height']} pixels")
        print(f"   Bits: {metadata['image_pixels'] * 4}")
    
    if metadata.get('has_audio'):
        print(f"Audio: {metadata['audio_samples']} samples at {metadata['audio_sample_rate']} Hz")
        print(f"   Compression: {metadata['audio_downsample']}x downsample")
        print(f"   Duration: {metadata['audio_samples'] / metadata['audio_sample_rate']:.2f} seconds")
        print(f"   Bits: {metadata['audio_samples'] * 8}")
    
    print(f"Total Bits: {metadata['total_bits']:,}")
    print(f"Transmission Time: {transmission_time:.1f} seconds")
    print(f"SNR: {snr_db} dB")
    print(f"Effective Data Rate: {metadata['total_bits']/transmission_time:.0f} bps")
    print("-"*60)

def validate_test_files():
    print("Validating test files...")
    
    required_files = {
        "data/matlab_logo.png": "MATLAB logo image (project requirement)",
        "data/audio_recording.wav": "Audio recording with phrase 'Hi, How are you? My name is <name>'"
    }
    
    missing_files = []
    
    for filepath, description in required_files.items():
        if not check_file_exists(filepath, description):
            missing_files.append((filepath, description))
    
    if missing_files:
        print("input files missing")
        print("-"*50)
        for filepath, description in missing_files:
            print(f"{filepath}")
            print(f"   {description}")
            print()
        
        print("Setup Instructions:")
        print("1. Create 'data/' directory in project root")
        print("2. Add MATLAB logo image as 'matlab_logo.png'")
        print("3. Record audio with required phrase as 'audio_recording.wav'")
        
        return False
    else:
        print("All required test files found!")
        return True
