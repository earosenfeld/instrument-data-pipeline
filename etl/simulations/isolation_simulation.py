import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict
from etl.simulations.daq_sim import DAQSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationSimulator:
    """Simulates Isolation Resistance test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        
    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate Isolation Resistance test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for resistance and voltage
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to resistance/voltage sensors")
                
            # Read resistance and voltage data
            resistance_data = self.daq.read_data('analog', duration, num_channels=1)
            voltage_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': resistance_data['timestamps'],
                'resistance': resistance_data['data'][0] * 1e6,  # Scale to MΩ
                'voltage': voltage_data['data'][0] * 1000,  # Scale to V
            })
            
            # Calculate pass/fail based on resistance threshold
            df['pass_fail'] = (df['resistance'] > 100).astype(int)  # Pass if resistance > 100MΩ
            
            logger.info(f"Generated {len(df)} samples of Isolation Resistance test data")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Isolation Resistance test data: {str(e)}")
            raise
        finally:
            self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze Isolation Resistance test data.
        
        Args:
            data (pd.DataFrame): Test data to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Calculate pass rate
            pass_rate = data['pass_fail'].mean()
            
            # Calculate resistance statistics
            resistance_stats = {
                'mean': data['resistance'].mean(),
                'std': data['resistance'].std(),
                'min': data['resistance'].min(),
                'max': data['resistance'].max()
            }
            
            # Calculate voltage statistics
            voltage_stats = {
                'mean': data['voltage'].mean(),
                'std': data['voltage'].std(),
                'min': data['voltage'].min(),
                'max': data['voltage'].max()
            }
            
            # Plot results
            self._plot_results(data)
            
            return {
                'pass_rate': pass_rate,
                'resistance_stats': resistance_stats,
                'voltage_stats': voltage_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Isolation Resistance test data: {str(e)}")
            raise
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot Isolation Resistance test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot resistance
            ax1.plot(data['timestamp'], data['resistance'], label='Resistance (MΩ)')
            ax1.axhline(y=100, color='r', linestyle='--', label='Resistance Limit (100MΩ)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Resistance (MΩ)')
            ax1.set_title('Isolation Resistance Over Time')
            ax1.legend()
            
            # Plot voltage
            ax2.plot(data['timestamp'], data['voltage'], label='Voltage (V)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Voltage (V)')
            ax2.set_title('Test Voltage Over Time')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting Isolation Resistance test results: {str(e)}")
            raise

def main():
    """Main function to run the Isolation Resistance simulation."""
    try:
        # Create simulator
        simulator = IsolationSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=60)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nIsolation Resistance Test Results:")
        print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
        print("\nResistance Statistics:")
        print(f"Mean: {results['resistance_stats']['mean']:.2f} MΩ")
        print(f"Std: {results['resistance_stats']['std']:.2f} MΩ")
        print(f"Min: {results['resistance_stats']['min']:.2f} MΩ")
        print(f"Max: {results['resistance_stats']['max']:.2f} MΩ")
        print("\nVoltage Statistics:")
        print(f"Mean: {results['voltage_stats']['mean']:.2f} V")
        print(f"Std: {results['voltage_stats']['std']:.2f} V")
        print(f"Min: {results['voltage_stats']['min']:.2f} V")
        print(f"Max: {results['voltage_stats']['max']:.2f} V")
        
    except Exception as e:
        logger.error(f"Error in Isolation Resistance simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 