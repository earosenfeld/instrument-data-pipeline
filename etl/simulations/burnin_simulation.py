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

class BurnInSimulator:
    """Simulates burn-in test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        self.report_dir = 'reports/burnin'
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
        
        # Calculate process capability indices for temperature
        temp_mean = data['temperature'].mean()
        temp_std = data['temperature'].std()
        temp_cp = (90 - temp_mean) / (3 * temp_std)  # Upper limit only for temperature
        temp_cpk = temp_cp  # Since we only care about upper limit
        
        # Calculate capability indices for voltage and current
        voltage_mean = data['voltage'].mean()
        voltage_std = data['voltage'].std()
        voltage_cp = min(3.6 - voltage_mean, voltage_mean - 3.0) / (3 * voltage_std)
        voltage_cpk = min((3.6 - voltage_mean) / (3 * voltage_std), (voltage_mean - 3.0) / (3 * voltage_std))
        
        current_mean = data['current'].mean()
        current_std = data['current'].std()
        current_cp = min(0.7 - current_mean, current_mean - 0.3) / (3 * current_std)
        current_cpk = min((0.7 - current_mean) / (3 * current_std), (current_mean - 0.3) / (3 * current_std))
        
        # Add capability indices to results
        results['temperature_capability'] = {
            'cp': temp_cp,
            'cpk': temp_cpk
        }
        results['voltage_capability'] = {
            'cp': voltage_cp,
            'cpk': voltage_cpk
        }
        results['current_capability'] = {
            'cp': current_cp,
            'cpk': current_cpk
        }
        
        # Save to JSON file
        filename = f"burnin_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"burnin_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot burn-in test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot temperature
            ax1.plot(data['timestamp'], data['temperature'], label='Temperature (C)')
            ax1.axhline(y=90, color='r', linestyle='--', label='Temperature Limit')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Temperature (C)')
            ax1.set_title('Burn-In Temperature Over Time')
            ax1.legend()
            
            # Plot voltage and current
            ax2.plot(data['timestamp'], data['voltage'], label='Voltage (V)')
            ax2.axhline(y=3.0, color='r', linestyle='--', label='Voltage Limits')
            ax2.axhline(y=3.6, color='r', linestyle='--')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Voltage (V)')
            ax2.set_title('Voltage Over Time')
            ax2.legend()
            
            ax3.plot(data['timestamp'], data['current'], label='Current (A)')
            ax3.axhline(y=0.3, color='r', linestyle='--', label='Current Limits')
            ax3.axhline(y=0.7, color='r', linestyle='--')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Current (A)')
            ax3.set_title('Current Over Time')
            ax3.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'burnin_timeseries')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting burn-in test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots for temperature
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Calculate control limits for temperature
            temp_mean = data['temperature'].mean()
            temp_std = data['temperature'].std()
            temp_ucl = temp_mean + 3 * temp_std
            temp_lcl = temp_mean - 3 * temp_std
            
            # Plot temperature X-bar chart
            ax1.plot(data['timestamp'], data['temperature'], 'b-', label='Temperature')
            ax1.axhline(y=temp_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=temp_ucl, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=temp_lcl, color='r', linestyle='--', label='LCL')
            ax1.set_title('Temperature X-bar Chart')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Temperature (C)')
            ax1.legend()
            
            # Plot temperature R chart
            temp_r = data['temperature'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax2.plot(data['timestamp'], temp_r, 'b-', label='Range')
            ax2.axhline(y=temp_r.mean(), color='g', linestyle='-', label='Mean')
            ax2.axhline(y=temp_r.mean() + 3 * temp_r.std(), color='r', linestyle='--', label='UCL')
            ax2.axhline(y=temp_r.mean() - 3 * temp_r.std(), color='r', linestyle='--', label='LCL')
            ax2.set_title('Temperature R Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Range (C)')
            ax2.legend()
            
            plt.tight_layout()
            self._save_plot(fig1, 'burnin_temp_spc')
            plt.close(fig1)
            
            # Create figure with subplots for voltage and current
            fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Calculate control limits for voltage
            voltage_mean = data['voltage'].mean()
            voltage_std = data['voltage'].std()
            voltage_ucl = voltage_mean + 3 * voltage_std
            voltage_lcl = voltage_mean - 3 * voltage_std
            
            # Plot voltage X-bar chart
            ax1.plot(data['timestamp'], data['voltage'], 'b-', label='Voltage')
            ax1.axhline(y=voltage_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=voltage_ucl, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=voltage_lcl, color='r', linestyle='--', label='LCL')
            ax1.set_title('Voltage X-bar Chart')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Voltage (V)')
            ax1.legend()
            
            # Plot voltage R chart
            voltage_r = data['voltage'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax2.plot(data['timestamp'], voltage_r, 'b-', label='Range')
            ax2.axhline(y=voltage_r.mean(), color='g', linestyle='-', label='Mean')
            ax2.axhline(y=voltage_r.mean() + 3 * voltage_r.std(), color='r', linestyle='--', label='UCL')
            ax2.axhline(y=voltage_r.mean() - 3 * voltage_r.std(), color='r', linestyle='--', label='LCL')
            ax2.set_title('Voltage R Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Range (V)')
            ax2.legend()
            
            # Calculate control limits for current
            current_mean = data['current'].mean()
            current_std = data['current'].std()
            current_ucl = current_mean + 3 * current_std
            current_lcl = current_mean - 3 * current_std
            
            # Plot current X-bar chart
            ax3.plot(data['timestamp'], data['current'], 'b-', label='Current')
            ax3.axhline(y=current_mean, color='g', linestyle='-', label='Mean')
            ax3.axhline(y=current_ucl, color='r', linestyle='--', label='UCL')
            ax3.axhline(y=current_lcl, color='r', linestyle='--', label='LCL')
            ax3.set_title('Current X-bar Chart')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Current (A)')
            ax3.legend()
            
            # Plot current R chart
            current_r = data['current'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax4.plot(data['timestamp'], current_r, 'b-', label='Range')
            ax4.axhline(y=current_r.mean(), color='g', linestyle='-', label='Mean')
            ax4.axhline(y=current_r.mean() + 3 * current_r.std(), color='r', linestyle='--', label='UCL')
            ax4.axhline(y=current_r.mean() - 3 * current_r.std(), color='r', linestyle='--', label='LCL')
            ax4.set_title('Current R Chart')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Range (A)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig2, 'burnin_vi_spc')
            plt.close(fig2)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

    def generate_test_data(self, duration: float = 3600) -> pd.DataFrame:
        """
        Generate burn-in test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for temperature
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to temperature sensor")
                
            # Read temperature data
            temp_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Connect to digital I/O for status
            if not self.daq.connect('digital'):
                raise ConnectionError("Failed to connect to status monitor")
                
            # Read status data
            status_data = self.daq.read_data('digital', duration, num_channels=1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': temp_data['timestamps'],
                'temperature': temp_data['data'][0],
                'status': status_data['data'][0]
            })
            
            # Add derived measurements
            df['voltage'] = np.random.normal(loc=3.3, scale=0.1, size=len(df))
            df['current'] = np.random.normal(loc=0.5, scale=0.05, size=len(df))
            
            # Calculate failures based on temperature thresholds
            df['failures'] = (df['temperature'] > 90).astype(int)
            
            logger.info(f"Generated {len(df)} samples of burn-in test data")
            return df
            
        except Exception as e:
            logger.error(f"Error generating burn-in test data: {str(e)}")
            raise
        finally:
            self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze burn-in test data.
        
        Args:
            data (pd.DataFrame): Test data to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Calculate failure rate
            failure_rate = data['failures'].mean()
            
            # Calculate temperature statistics
            temp_stats = {
                'mean': data['temperature'].mean(),
                'std': data['temperature'].std(),
                'min': data['temperature'].min(),
                'max': data['temperature'].max()
            }
            
            # Calculate voltage and current statistics
            voltage_stats = {
                'mean': data['voltage'].mean(),
                'std': data['voltage'].std(),
                'min': data['voltage'].min(),
                'max': data['voltage'].max()
            }
            
            current_stats = {
                'mean': data['current'].mean(),
                'std': data['current'].std(),
                'min': data['current'].min(),
                'max': data['current'].max()
            }
            
            results = {
                'failure_rate': failure_rate,
                'temperature_stats': temp_stats,
                'voltage_stats': voltage_stats,
                'current_stats': current_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Plot and save results
            self._plot_results(data)
            self._plot_spc_charts(data)
            self._save_statistics(results, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing burn-in test data: {str(e)}")
            raise

def main():
    """Main function to run the burn-in simulation."""
    try:
        # Create simulator
        simulator = BurnInSimulator()
        
        # Generate test data (1 hour duration)
        data = simulator.generate_test_data(duration=10)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nBurn-In Test Results:")
        print(f"Failure Rate: {results['failure_rate']*100:.2f}%")
        print("\nTemperature Statistics:")
        print(f"Mean: {results['temperature_stats']['mean']:.2f}°C")
        print(f"Std: {results['temperature_stats']['std']:.2f}°C")
        print(f"Min: {results['temperature_stats']['min']:.2f}°C")
        print(f"Max: {results['temperature_stats']['max']:.2f}°C")
        print("\nVoltage Statistics:")
        print(f"Mean: {results['voltage_stats']['mean']:.2f}V")
        print(f"Std: {results['voltage_stats']['std']:.2f}V")
        print(f"Min: {results['voltage_stats']['min']:.2f}V")
        print(f"Max: {results['voltage_stats']['max']:.2f}V")
        print("\nCurrent Statistics:")
        print(f"Mean: {results['current_stats']['mean']:.2f}A")
        print(f"Std: {results['current_stats']['std']:.2f}A")
        print(f"Min: {results['current_stats']['min']:.2f}A")
        print(f"Max: {results['current_stats']['max']:.2f}A")
        
    except Exception as e:
        logger.error(f"Error in burn-in simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 