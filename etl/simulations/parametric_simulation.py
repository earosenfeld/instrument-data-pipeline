import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict
from etl.simulations.daq_sim import DAQSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParametricSimulator:
    """Simulates Parametric test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        
    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate Parametric test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for voltage and current
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to voltage/current sensors")
                
            # Read voltage and current data
            voltage_data = self.daq.read_data('analog', duration, num_channels=1)
            current_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': voltage_data['timestamps'],
                'voltage': voltage_data['data'][0] * 1000,  # Scale to mV
                'current': current_data['data'][0] * 1000,  # Scale to mA
            })
            
            # Calculate pass/fail based on voltage and current thresholds
            df['pass_fail'] = ((df['voltage'] >= 4500) & (df['voltage'] <= 5500) & 
                             (df['current'] >= 90) & (df['current'] <= 110)).astype(int)
            
            logger.info(f"Generated {len(df)} samples of Parametric test data")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Parametric test data: {str(e)}")
            raise
        finally:
            self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze Parametric test data.
        
        Args:
            data (pd.DataFrame): Test data to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Calculate pass rate
            pass_rate = data['pass_fail'].mean()
            
            # Calculate voltage statistics
            voltage_stats = {
                'mean': data['voltage'].mean(),
                'std': data['voltage'].std(),
                'min': data['voltage'].min(),
                'max': data['voltage'].max()
            }
            
            # Calculate current statistics
            current_stats = {
                'mean': data['current'].mean(),
                'std': data['current'].std(),
                'min': data['current'].min(),
                'max': data['current'].max()
            }
            
            # Plot results
            self._plot_results(data)
            
            return {
                'pass_rate': pass_rate,
                'voltage_stats': voltage_stats,
                'current_stats': current_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Parametric test data: {str(e)}")
            raise
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot Parametric test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot voltage
            ax1.plot(data['timestamp'], data['voltage'], label='Voltage (mV)')
            ax1.axhline(y=4500, color='r', linestyle='--', label='Voltage Limits (4.5-5.5V)')
            ax1.axhline(y=5500, color='r', linestyle='--')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Voltage (mV)')
            ax1.set_title('Parametric Voltage Over Time')
            ax1.legend()
            
            # Plot current
            ax2.plot(data['timestamp'], data['current'], label='Current (mA)')
            ax2.axhline(y=90, color='r', linestyle='--', label='Current Limits (90-110mA)')
            ax2.axhline(y=110, color='r', linestyle='--')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Current (mA)')
            ax2.set_title('Parametric Current Over Time')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting Parametric test results: {str(e)}")
            raise

def main():
    """Main function to run the Parametric simulation."""
    try:
        # Create simulator
        simulator = ParametricSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=60)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nParametric Test Results:")
        print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
        print("\nVoltage Statistics:")
        print(f"Mean: {results['voltage_stats']['mean']:.2f} mV")
        print(f"Std: {results['voltage_stats']['std']:.2f} mV")
        print(f"Min: {results['voltage_stats']['min']:.2f} mV")
        print(f"Max: {results['voltage_stats']['max']:.2f} mV")
        print("\nCurrent Statistics:")
        print(f"Mean: {results['current_stats']['mean']:.2f} mA")
        print(f"Std: {results['current_stats']['std']:.2f} mA")
        print(f"Min: {results['current_stats']['min']:.2f} mA")
        print(f"Max: {results['current_stats']['max']:.2f} mA")
        
    except Exception as e:
        logger.error(f"Error in Parametric simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 