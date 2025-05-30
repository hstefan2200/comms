"""
Real Transmission Analysis
==========================

This script performs actual file transmissions (image, audio, combined) and analyzes their performance:
- BER analysis on real transmitted data
- SNR vs performance curves
- Transmission time vs quality trade-offs
- Compression effectiveness analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import erfc
from PIL import Image
from scipy.io import wavfile

# Import our communication system components
from src.transmitter2 import ImprovedTransmitter
from src.receiver2 import ImprovedReceiver
from src.utils import (
    create_directories, check_file_exists, format_file_size,
    estimate_transmission_time, calculate_theoretical_ber, create_ber_plot
)


class TransmissionAnalyzer:
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000, quantization_bits=16):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.quantization_bits = quantization_bits
        
        self.transmitter = ImprovedTransmitter(carrier_freq, sample_rate, symbol_rate, quantization_bits)
        self.receiver = ImprovedReceiver(carrier_freq, sample_rate, symbol_rate, quantization_bits)
        
        create_directories(["data/transmissions", "data/received", "results/analysis"])
        
        print("Transmission Analyzer initialized")
        print(f"   Carrier: {carrier_freq} Hz, Sample Rate: {sample_rate} Hz")
        print(f"   Symbol Rate: {symbol_rate} Hz, Quantization: {quantization_bits} bits")
    
    def run_single_transmission(self, test_name, image_path=None, audio_path=None, snr_db=15, image_quality=4):   
        print(f"\n{'-'*50}")
        print(f"Transmission test: {test_name.upper()}")

        print("Creating transmission packet...")
        try:
            packet_bits, metadata = self.transmitter.create_packet(image_path, audio_path, image_quality)
            
            if len(packet_bits) == 0:
                print("No data to transmit")
                return None

            transmission_time = estimate_transmission_time(len(packet_bits), self.symbol_rate)
            print(f"Estimated transmission time: {transmission_time:.2f} seconds")
            
            #transmit
            output_file = f"data/transmissions/{test_name}_transmission.wav"
            audio_signal, clean_signal, noise, actual_snr = self.transmitter.transmit_data(
                packet_bits, snr_db=snr_db, output_audio_file=output_file
            )
            
            print(f"Transmission complete: {output_file}")
            
        except Exception as e:
            print(f"Transmission failed: {e}")
            return None
        
        #Receive
        print(f"Receiving transmission...")
        try:
            results = self.receiver.receive_file(
                output_file, 
                original_bits=packet_bits,  # For ber calculation
                output_dir=f"data/received/{test_name}/"
            )
            
            if not results:
                print("Reception failed")
                return None
                
            print("reception complete")
            
        except Exception as e:
            print(f"Reception failed: {e}")
            return None
        
        #Compile results
        transmission_results = {
            'test_name': test_name,
            'metadata': metadata,
            'transmission_file': output_file,
            'packet_bits': packet_bits,
            'recovered_bits': results['recovered_bits'],
            'ber': results['ber'],
            'bit_errors': results['errors'],
            'snr_target': snr_db,
            'snr_actual': actual_snr,
            'transmission_time': transmission_time,
            'reconstructed_image': results['reconstructed_image'],
            'reconstructed_audio': results['reconstructed_audio'],
            'success': results['ber'] is not None and results['ber'] < 0.1  # 10% error threshold
        }
        
        return transmission_results
    
    def run_snr_analysis(self, test_name, image_path=None, audio_path=None, snr_range=None):        
        if snr_range is None:
            snr_range = [3, 4, 5, 6, 7, 8] #keeps plots interesting
        
        print(f"\n{'-'*50}")
        print(f"SNR analysis: {test_name.upper()}")
        print(f"Testing SNR range: {snr_range} dB")
        
        snr_results = {
            'snr_values': [],
            'ber_values': [],
            'success_rate': [],
            'transmission_times': [],
            'bit_counts': []
        }
        
        # Test each SNR level
        for snr_db in snr_range:            
            results = self.run_single_transmission(
                f"{test_name}_snr{snr_db}", 
                image_path, 
                audio_path, 
                snr_db=snr_db
            )
            
            if results:
                snr_results['snr_values'].append(snr_db)
                snr_results['ber_values'].append(results['ber'] if results['ber'] is not None else 1.0)
                snr_results['success_rate'].append(1.0 if results['success'] else 0.0)
                snr_results['transmission_times'].append(results['transmission_time'])
                snr_results['bit_counts'].append(len(results['packet_bits']))
            else:
                snr_results['snr_values'].append(snr_db)
                snr_results['ber_values'].append(1.0) 
                snr_results['success_rate'].append(0.0)
                snr_results['transmission_times'].append(0.0)
                snr_results['bit_counts'].append(0)
        
        #Plot ber vs snr
        self.plot_snr_analysis(snr_results, test_name)
        
        return snr_results
    
    def plot_snr_analysis(self, snr_results, test_name):
        
        print(f"{'SNR (dB)':<8} {'BER':<12} {'Quality':<12} {'Success':<8}")
        print("-"*45)              
        for snr, ber, success in zip(snr_results['snr_values'], snr_results['ber_values'], snr_results['success_rate']):
            # Categorize performance
            if ber < 0.0001:
                quality = "Excellent"
            elif ber < 0.01:
                quality = "Good"
            elif ber < 0.1:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"{snr:<8} {ber:<12.2e} {quality:<12} {success*100:<7.0f}%")
        
        total_bits = max(snr_results['bit_counts']) if snr_results['bit_counts'] else 0
        avg_time = np.mean(snr_results['transmission_times'])
        data_rate = total_bits / avg_time if avg_time > 0 else 0
        
        print(f"System paramters:")
        print(f"  Data size: {total_bits:,} bits")
        print(f"  Transmission time: {avg_time:.1f} seconds")
        print(f"  Data rate: {data_rate:.0f} bps (vs {self.symbol_rate} theoretical)")
        print(f"  Efficiency: {(data_rate/self.symbol_rate)*100:.1f}%")
        
        create_ber_plot(snr_results['snr_values'], snr_results['ber_values'], 
                       show_theoretical=True, 
                       save_path=f'results/analysis/{test_name}_ber.png')
    
    def _find_snr_threshold(self, snr_results, ber_threshold):
        suitable_snrs = [snr for snr, ber in zip(snr_results['snr_values'], snr_results['ber_values']) 
                        if ber <= ber_threshold]
        return min(suitable_snrs) if suitable_snrs else "N/A"
    
    def run_full_analysis(self):                
        image_path = "data/matlab_logo.png" if check_file_exists("data/matlab_logo.png") else None
        audio_path = "data/audio_recording.wav" if check_file_exists("data/audio_recording.wav") else None
        
        if not image_path and not audio_path:
            print("No test files found. Please add:")
            print("   - data/matlab_logo.png (project requirement)")
            print("   - data/audio_recording.wav (speech sample)")
            return
        
        all_results = {}
        
        #Test 1: Image only
        if image_path:
            print("ANALYZING IMAGE TRANSMISSION")
            all_results['image'] = self.run_snr_analysis('image_only', image_path=image_path)
        
        #Test 2: Audio only  
        if audio_path:
            print("ANALYZING AUDIO TRANSMISSION")
            all_results['audio'] = self.run_snr_analysis('audio_only', audio_path=audio_path)
        
        #Test 3: Combined transmission
        if image_path and audio_path:
            print("ANALYZING COMBINED TRANSMISSION")
            all_results['combined'] = self.run_snr_analysis('combined', image_path=image_path, audio_path=audio_path)
        
        if len(all_results) > 1:
            self.plot_comparison(all_results)
        
        self.generate_analysis_report(all_results)
        
        return all_results
    
    def plot_comparison(self, all_results):               
        print(f"{'Type':<12} {'Bits':<8} {'Time':<8} {'Rate':<8} {'Avg BER':<12} {'Best SNR':<10}")
        print("-"*65)
        
        for test_name, results in all_results.items():
            max_bits = max(results['bit_counts']) if results['bit_counts'] else 0
            avg_time = np.mean(results['transmission_times'])
            data_rate = max_bits / avg_time if avg_time > 0 else 0
            avg_ber = np.mean(results['ber_values'])
            best_snr = self._find_snr_threshold(results, 0.01)
            
            print(f"{test_name.title():<12} {max_bits//1000:<7}k {avg_time:<7.1f}s {data_rate:<7.0f} {avg_ber:<12.1e} {best_snr}")
    
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (test_name, results) in enumerate(all_results.items()):
            #Clean data
            snr_vals = np.array(results['snr_values'])
            ber_vals = np.array(results['ber_values'])

            valid_mask = (ber_vals > 0) & (~np.isnan(ber_vals)) & (~np.isinf(ber_vals))
            snr_clean = snr_vals[valid_mask]
            ber_clean = ber_vals[valid_mask]
            
            if len(snr_clean) > 0:
                plt.semilogy(snr_clean, ber_clean, f'{markers[i % len(markers)]}-', 
                            label=f'{test_name.title()}', markersize=8, linewidth=3,
                            color=colors[i % len(colors)])

                for snr, ber in zip(snr_clean, ber_clean):
                    plt.annotate(f'{ber:.1e}', (snr, ber), 
                                textcoords="offset points", xytext=(0,10), 
                                ha='center', fontsize=8, color=colors[i % len(colors)])
        
        # Add theoretical curve
        if all_results:
            all_snr_values = []
            for results in all_results.values():
                all_snr_values.extend(results['snr_values'])
            
            if all_snr_values:
                snr_range = np.linspace(min(all_snr_values)-2, max(all_snr_values)+2, 100)
                theoretical = calculate_theoretical_ber(snr_range)
                plt.semilogy(snr_range, theoretical, 'k--', label='Theoretical BPSK', 
                            alpha=0.7, linewidth=2)
        
        plt.grid(True, which="both", alpha=0.3)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Bit Error Rate', fontsize=12)
        plt.title('BER Comparison: All Transmission Types', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.ylim([1e-8, 1])
        
        # Set x-limits based on chosen snr values
        if all_results:
            all_snr = [snr for results in all_results.values() for snr in results['snr_values']]
            if all_snr:
                plt.xlim([min(all_snr)-1, max(all_snr)+1])
        
        plt.tight_layout()
        plt.savefig('results/analysis/ber_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def generate_analysis_report(self, all_results): 
        report = f"""
        TRANSMISSION ANALYSIS REPORT
        ============================

        System Configuration:
        - Carrier Frequency: {self.carrier_freq} Hz
        - Sample Rate: {self.sample_rate} Hz  
        - Symbol Rate: {self.symbol_rate} Hz
        - Quantization: {self.quantization_bits} bits

        """
        
        for test_name, results in all_results.items():
            avg_ber = np.mean(results['ber_values'])
            min_ber = min(results['ber_values'])
            max_bits = max(results['bit_counts']) if results['bit_counts'] else 0
            avg_time = np.mean(results['transmission_times'])
            success_rate = np.mean(results['success_rate']) * 100
            
            report += f"""
            {test_name.upper()} TRANSMISSION RESULTS:
            {'='*40}
            - Data Size: {max_bits:,} bits ({format_file_size(max_bits//8)})
            - Average BER: {avg_ber:.2e}
            - Best BER: {min_ber:.2e}
            - Success Rate: {success_rate:.1f}%
            - Transmission Time: {avg_time:.1f} seconds
            - Effective Data Rate: {max_bits/avg_time:.0f} bps
            """
        # Save report
        with open("results/analysis/transmission_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
        
        print(report)


def main():
    analyzer = TransmissionAnalyzer(
        carrier_freq=1000,
        sample_rate=8000, 
        symbol_rate=2000,
        quantization_bits=16
    )
    results = analyzer.run_full_analysis()
    
    print(f"Analysis Complete!")

if __name__ == "__main__":
    main()