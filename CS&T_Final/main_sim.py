import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import erfc

# Import all our modules
from src.channel_model import AWGNChannel
from src.modulation import BPSKModulator, BPSKDemodulator
from src.audio_processing import AudioProcessor
from src.image_processing import ImageProcessor
from src.transmitter import Transmitter
from src.receiver import Receiver
from src.ber_analysis import run_ber_analysis, plot_ber_curve

class CommunicationSystem:
    """Complete communication system with consistent parameters"""
    
    def __init__(self, carrier_freq=1000, sample_rate=8000, symbol_rate=2000, quantization_bits=16):
        # MASTER PARAMETERS - ALL COMPONENTS USE THESE
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.quantization_bits = quantization_bits
        self.samples_per_symbol = sample_rate // symbol_rate
        
        print("="*60)
        print("COMMUNICATION SYSTEM INITIALIZATION")
        print("="*60)
        print(f"Carrier Frequency: {self.carrier_freq} Hz")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Symbol Rate: {self.symbol_rate} Hz")
        print(f"Samples per Symbol: {self.samples_per_symbol}")
        print(f"Quantization Bits: {self.quantization_bits}")
        print("="*60)
        
        # Initialize all components with IDENTICAL parameters
        self.modulator = BPSKModulator(carrier_freq, sample_rate, symbol_rate)
        self.demodulator = BPSKDemodulator(carrier_freq, sample_rate, symbol_rate)
        self.channel = AWGNChannel()
        self.audio_processor = AudioProcessor(quantization_bits)
        self.image_processor = ImageProcessor()
        
        # Initialize transmitter and receiver with same parameters
        self.transmitter = Transmitter(carrier_freq, sample_rate, quantization_bits)
        self.receiver = Receiver(carrier_freq, sample_rate, quantization_bits)
        
        # Create directories
        os.makedirs("data/transmitted", exist_ok=True)
        os.makedirs("data/received", exist_ok=True)
        
    def verify_basic_modulation(self):
        """Test basic BPSK modulation without noise"""
        print("\n" + "="*50)
        print("BASIC MODULATION VERIFICATION")
        print("="*50)
        
        # Simple test pattern
        test_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 20)  # 160 bits
        print(f"Test pattern: {len(test_bits)} bits")
        print(f"First 32 bits: {test_bits[:32]}")
        
        # Modulate
        modulated_signal = self.modulator.modulate(test_bits)
        print(f"Modulated signal: {len(modulated_signal)} samples")
        print(f"Expected samples: {len(test_bits) * self.samples_per_symbol}")
        
        # Demodulate (no noise)
        recovered_bits = self.demodulator.demodulate(modulated_signal)
        print(f"Recovered bits: {len(recovered_bits)}")
        
        # Check errors
        if len(recovered_bits) != len(test_bits):
            min_len = min(len(test_bits), len(recovered_bits))
            test_bits = test_bits[:min_len]
            recovered_bits = recovered_bits[:min_len]
            print(f"WARNING: Length mismatch, comparing first {min_len} bits")
        
        errors = np.sum(test_bits != recovered_bits)
        print(f"Bit errors (no noise): {errors}/{len(test_bits)}")
        print(f"First 32 recovered: {recovered_bits[:32]}")
        
        if errors == 0:
            print("✅ Basic modulation PASSED")
            return True
        else:
            print("❌ Basic modulation FAILED")
            print(f"Error positions: {np.where(test_bits != recovered_bits)[0][:10]}")
            return False
    
    def test_with_noise(self, snr_db=10):
        """Test modulation with AWGN noise"""
        print(f"\n" + "="*50)
        print(f"NOISE TESTING (SNR = {snr_db} dB)")
        print("="*50)
        
        test_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 50)  # 400 bits
        print(f"Testing with {len(test_bits)} bits")
        
        # Modulate
        modulated_signal = self.modulator.modulate(test_bits)
        
        # Add noise
        noisy_signal, noise = self.channel.add_noise(modulated_signal, snr_db)
        actual_snr = self.channel.calculate_snr(modulated_signal, noise)
        print(f"Target SNR: {snr_db} dB, Actual SNR: {actual_snr:.2f} dB")
        
        # Demodulate
        recovered_bits = self.demodulator.demodulate(noisy_signal)
        
        # Calculate BER
        if len(recovered_bits) != len(test_bits):
            min_len = min(len(test_bits), len(recovered_bits))
            test_bits = test_bits[:min_len]
            recovered_bits = recovered_bits[:min_len]
        
        errors = np.sum(test_bits != recovered_bits)
        ber = errors / len(test_bits)
        
        print(f"Bit errors: {errors}/{len(test_bits)}")
        print(f"BER: {ber:.6f}")
        print(f"Success rate: {(1-ber)*100:.2f}%")
        
        return ber < 0.1  # Less than 10% error rate
    
    def test_image_transmission(self, image_path="data/matlab_logo.png", snr_db=15):
        """Test complete image transmission"""
        print(f"\n" + "="*50)
        print(f"IMAGE TRANSMISSION TEST (SNR = {snr_db} dB)")
        print("="*50)
        
        if not os.path.exists(image_path):
            print(f"❌ Image file {image_path} not found!")
            return False
        
        # TRANSMIT
        print("🔄 Preparing image data...")
        image_bits, metadata = self.transmitter.prepare_data_packet(image_path=image_path)
        print(f"Image converted to {len(image_bits)} bits")
        print(f"Metadata: {metadata}")
        
        print("📡 Transmitting...")
        output_file = "data/transmitted/image_transmission.wav"
        audio_signal, clean_signal, noise, actual_snr = self.transmitter.transmit_data(
            image_bits, snr_db=snr_db, output_audio_file=output_file
        )
        
        # RECEIVE
        print("📻 Receiving and processing...")
        results = self.receiver.receive_and_process(
            audio_file_path=output_file,
            original_bits=image_bits,
            output_dir="data/received/"
        )
        
        if results and results['reconstructed_image'] is not None:
            print("✅ Image transmission SUCCESSFUL")
            return True
        else:
            print("❌ Image transmission FAILED")
            return False
    
    def test_audio_transmission(self, audio_path="data/audio_recording.wav", snr_db=15):
        """Test complete audio transmission"""
        print(f"\n" + "="*50)
        print(f"AUDIO TRANSMISSION TEST (SNR = {snr_db} dB)")
        print("="*50)
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file {audio_path} not found!")
            return False
        
        # TRANSMIT
        print("🔄 Preparing audio data...")
        audio_bits, metadata = self.transmitter.prepare_data_packet(audio_path=audio_path)
        print(f"Audio converted to {len(audio_bits)} bits")
        print(f"Metadata: {metadata}")
        
        print("📡 Transmitting...")
        output_file = "data/transmitted/audio_transmission.wav"
        audio_signal, clean_signal, noise, actual_snr = self.transmitter.transmit_data(
            audio_bits, snr_db=snr_db, output_audio_file=output_file
        )
        
        # RECEIVE
        print("📻 Receiving and processing...")
        results = self.receiver.receive_and_process(
            audio_file_path=output_file,
            original_bits=audio_bits,
            output_dir="data/received/"
        )
        
        if results and results['reconstructed_audio'] is not None:
            print("✅ Audio transmission SUCCESSFUL")
            return True
        else:
            print("❌ Audio transmission FAILED")
            return False
    
    def test_combined_transmission(self, image_path="data/matlab_logo.png", 
                                 audio_path="data/audio_recording.wav", snr_db=15):
        """Test combined image + audio transmission"""
        print(f"\n" + "="*50)
        print(f"COMBINED TRANSMISSION TEST (SNR = {snr_db} dB)")
        print("="*50)
        
        if not os.path.exists(image_path):
            print(f"❌ Image file {image_path} not found!")
            return False
            
        if not os.path.exists(audio_path):
            print(f"❌ Audio file {audio_path} not found!")
            return False
        
        # TRANSMIT
        print("🔄 Preparing combined data...")
        combined_bits, metadata = self.transmitter.prepare_data_packet(
            image_path=image_path, audio_path=audio_path
        )
        print(f"Combined data: {len(combined_bits)} bits")
        print(f"Metadata: {metadata}")
        
        print("📡 Transmitting...")
        output_file = "data/transmitted/combined_transmission.wav"
        audio_signal, clean_signal, noise, actual_snr = self.transmitter.transmit_data(
            combined_bits, snr_db=snr_db, output_audio_file=output_file
        )
        
        # RECEIVE
        print("📻 Receiving and processing...")
        results = self.receiver.receive_and_process(
            audio_file_path=output_file,
            original_bits=combined_bits,
            output_dir="data/received/"
        )
        
        success = (results and 
                  results['reconstructed_image'] is not None and 
                  results['reconstructed_audio'] is not None)
        
        if success:
            print("✅ Combined transmission SUCCESSFUL")
            return True
        else:
            print("❌ Combined transmission FAILED")
            return False
    
    def run_ber_analysis(self, snr_range=None, num_bits=1000, num_trials=10):
        """Run comprehensive BER analysis"""
        print(f"\n" + "="*50)
        print("BER ANALYSIS")
        print("="*50)
        
        if snr_range is None:
            snr_range = np.arange(0, 16, 2)
        
        print(f"SNR range: {snr_range} dB")
        print(f"Bits per trial: {num_bits}")
        print(f"Trials per SNR: {num_trials}")
        
        snr_vals, ber_results = run_ber_analysis(
            self.modulator, self.demodulator, snr_range, num_bits, num_trials
        )
        
        # Plot results
        plot_ber_curve(snr_vals, ber_results, show_theoretical=True)
        
        return snr_vals, ber_results
    
    def run_complete_test_suite(self):
        """Run all tests in sequence"""
        print("\n" + "🚀" + "="*58 + "🚀")
        print("🚀" + " "*20 + "COMPLETE TEST SUITE" + " "*20 + "🚀")
        print("🚀" + "="*58 + "🚀")
        
        results = {}
        
        # Test 1: Basic modulation
        results['basic_modulation'] = self.verify_basic_modulation()
        
        if not results['basic_modulation']:
            print("\n❌ STOPPING: Basic modulation failed!")
            return results
        
        # Test 2: Noise resistance
        results['noise_test'] = self.test_with_noise(snr_db=10)
        
        # Test 3: Image transmission
        results['image_transmission'] = self.test_image_transmission(snr_db=15)
        
        # Test 4: Audio transmission
        results['audio_transmission'] = self.test_audio_transmission(snr_db=15)
        
        # Test 5: Combined transmission
        results['combined_transmission'] = self.test_combined_transmission(snr_db=15)
        
        # Summary
        print(f"\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:25}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System is working correctly! 🎉")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return results

def main():
    """Main simulation entry point"""
    print("Starting Communication System Simulation...")
    
    # Initialize system with consistent parameters
    comm_system = CommunicationSystem(
        carrier_freq=1000,
        sample_rate=8000,
        symbol_rate=2000,  # This gives us 4 samples per symbol
        quantization_bits=16
    )
    
    # Run complete test suite
    results = comm_system.run_complete_test_suite()
    
    # Optionally run BER analysis if basic tests pass
    if results.get('basic_modulation', False):
        print(f"\n" + "="*50)
        print("Running BER Analysis...")
        comm_system.run_ber_analysis(
            snr_range=np.arange(0, 16, 2),
            num_bits=500,  # Smaller for faster testing
            num_trials=5
        )

if __name__ == "__main__":
    main()