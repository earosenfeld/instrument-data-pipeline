import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
from typing import Dict
from etl.simulations.daq_sim import DAQSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParametricSimulator:
    """Simulates Parametric test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        self.report_dir = 'reports/parametric'
        os.makedirs(self.report_dir, exist_ok=True)
        
    def _save_plot(self, fig: plt.Figure, name: str) -> None:
        """Save plot to the reports directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.png"
        filepath = os.path.join(self.report_dir, filename)
        fig.savefig(filepath)
        logger.info(f"Saved plot to {filepath}")
        
    def _save_statistics(self, results: Dict, data: pd.DataFrame) -> None:
        """Save statistics to the reports directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to JSON file
        filename = f"parametric_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"parametric_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")
    
    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate Parametric test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for voltage and current measurements
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to measurement system")
                
            # Read voltage and current data
            voltage_data = self.daq.read_data('analog', duration, num_channels=1)
            current_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Create DataFrame with numpy array data
            data = []
            for i in range(len(voltage_data['timestamps'])):
                data.append({
                    'timestamp': voltage_data['timestamps'][i],
                    'voltage': abs(voltage_data['data'][0][i]) * 1000,  # Scale to mV
                    'current': abs(current_data['data'][0][i]) * 1000,  # Scale to mA
                })
            
            df = pd.DataFrame(data)
            
            # Calculate power and pass/fail
            df['power'] = df['voltage'] * df['current'] / 1000  # Power in mW
            df['pass_fail'] = ((df['voltage'] >= 3200) & (df['voltage'] <= 3400) &  # 3.2V - 3.4V
                             (df['current'] >= 450) & (df['current'] <= 550) &      # 450mA - 550mA
                             (df['power'] <= 2000)).astype(int)                     # Max 2W
            
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
            
            # Calculate statistics
            results = {
                'pass_rate': pass_rate,
                'voltage_stats': {
                    'mean': data['voltage'].mean(),
                    'std': data['voltage'].std(),
                    'min': data['voltage'].min(),
                    'max': data['voltage'].max()
                },
                'current_stats': {
                    'mean': data['current'].mean(),
                    'std': data['current'].std(),
                    'min': data['current'].min(),
                    'max': data['current'].max()
                },
                'power_stats': {
                    'mean': data['power'].mean(),
                    'std': data['power'].std(),
                    'min': data['power'].min(),
                    'max': data['power'].max()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Plot and save results
            self._plot_results(data)
            self._plot_spc_charts(data)
            self._save_statistics(results, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing Parametric test data: {str(e)}")
            raise
            
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot Parametric test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot voltage
            ax1.plot(data['timestamp'], data['voltage'], label='Voltage')
            ax1.axhline(y=3200, color='r', linestyle='--', label='Limits')
            ax1.axhline(y=3400, color='r', linestyle='--')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Voltage (mV)')
            ax1.set_title('Voltage Over Time')
            ax1.legend()
            
            # Plot current
            ax2.plot(data['timestamp'], data['current'], label='Current')
            ax2.axhline(y=450, color='r', linestyle='--', label='Limits')
            ax2.axhline(y=550, color='r', linestyle='--')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Current (mA)')
            ax2.set_title('Current Over Time')
            ax2.legend()
            
            # Plot power
            ax3.plot(data['timestamp'], data['power'], label='Power')
            ax3.axhline(y=2000, color='r', linestyle='--', label='Limit')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Power (mW)')
            ax3.set_title('Power Over Time')
            ax3.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'parametric_timeseries')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting Parametric test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot voltage SPC
            voltage_mean = data['voltage'].mean()
            voltage_std = data['voltage'].std()
            
            ax1.plot(data['timestamp'], data['voltage'], 'b-', label='Voltage')
            ax1.axhline(y=voltage_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=voltage_mean + 3 * voltage_std, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=voltage_mean - 3 * voltage_std, color='r', linestyle='--', label='LCL')
            ax1.set_title('Voltage X-bar Chart')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Voltage (mV)')
            ax1.legend()
            
            # Plot current SPC
            current_mean = data['current'].mean()
            current_std = data['current'].std()
            
            ax2.plot(data['timestamp'], data['current'], 'b-', label='Current')
            ax2.axhline(y=current_mean, color='g', linestyle='-', label='Mean')
            ax2.axhline(y=current_mean + 3 * current_std, color='r', linestyle='--', label='UCL')
            ax2.axhline(y=current_mean - 3 * current_std, color='r', linestyle='--', label='LCL')
            ax2.set_title('Current X-bar Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Current (mA)')
            ax2.legend()
            
            # Plot power SPC
            power_mean = data['power'].mean()
            power_std = data['power'].std()
            
            ax3.plot(data['timestamp'], data['power'], 'b-', label='Power')
            ax3.axhline(y=power_mean, color='g', linestyle='-', label='Mean')
            ax3.axhline(y=power_mean + 3 * power_std, color='r', linestyle='--', label='UCL')
            ax3.axhline(y=power_mean - 3 * power_std, color='r', linestyle='--', label='LCL')
            ax3.set_title('Power X-bar Chart')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Power (mW)')
            ax3.legend()
            
            # Plot R chart for voltage (as example)
            voltage_r = data['voltage'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax4.plot(data['timestamp'], voltage_r, 'b-', label='Range')
            ax4.axhline(y=voltage_r.mean(), color='g', linestyle='-', label='Mean')
            ax4.axhline(y=voltage_r.mean() + 3 * voltage_r.std(), color='r', linestyle='--', label='UCL')
            ax4.axhline(y=voltage_r.mean() - 3 * voltage_r.std(), color='r', linestyle='--', label='LCL')
            ax4.set_title('Voltage R Chart')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Range (mV)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'parametric_spc')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

def main():
    """Main function to run the Parametric simulation."""
    try:
        # Create simulator
        simulator = ParametricSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=10)
        
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
        
        print("\nPower Statistics:")
        print(f"Mean: {results['power_stats']['mean']:.2f} mW")
        print(f"Std: {results['power_stats']['std']:.2f} mW")
        print(f"Min: {results['power_stats']['min']:.2f} mW")
        print(f"Max: {results['power_stats']['max']:.2f} mW")
        
    except Exception as e:
        logger.error(f"Error in Parametric simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 