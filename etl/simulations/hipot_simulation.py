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

class HiPotSimulator:
    """Simulates HiPot test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        self.report_dir = 'reports/hipot'
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
        
        # Calculate process capability indices for voltage
        voltage_mean = data['voltage'].mean()
        voltage_std = data['voltage'].std()
        voltage_cp = min(5.5 - voltage_mean, voltage_mean - 4.5) / (3 * voltage_std)
        voltage_cpk = min((5.5 - voltage_mean) / (3 * voltage_std), (voltage_mean - 4.5) / (3 * voltage_std))
        
        # Calculate capability indices for current
        current_mean = data['current'].mean()
        current_std = data['current'].std()
        current_cp = (1.0 - current_mean) / (3 * current_std)  # Upper limit only
        current_cpk = current_cp  # Since we only care about upper limit
        
        # Add capability indices to results
        results['voltage_capability'] = {
            'cp': voltage_cp,
            'cpk': voltage_cpk
        }
        results['current_capability'] = {
            'cp': current_cp,
            'cpk': current_cpk
        }
        
        # Save to JSON file
        filename = f"hipot_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"hipot_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot HiPot test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot voltage
            ax1.plot(data['timestamp'], data['voltage'], label='Voltage (kV)')
            ax1.axhline(y=4.5, color='r', linestyle='--', label='Voltage Limits (4.5-5.5kV)')
            ax1.axhline(y=5.5, color='r', linestyle='--')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Voltage (kV)')
            ax1.set_title('HiPot Voltage Over Time')
            ax1.legend()
            
            # Plot current
            ax2.plot(data['timestamp'], data['current'], label='Current (mA)')
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Current Limit (1mA)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Current (mA)')
            ax2.set_title('HiPot Current Over Time')
            ax2.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'hipot_timeseries')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting HiPot test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
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
            ax1.set_ylabel('Voltage (kV)')
            ax1.legend()
            
            # Plot voltage R chart
            voltage_r = data['voltage'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax2.plot(data['timestamp'], voltage_r, 'b-', label='Range')
            ax2.axhline(y=voltage_r.mean(), color='g', linestyle='-', label='Mean')
            ax2.axhline(y=voltage_r.mean() + 3 * voltage_r.std(), color='r', linestyle='--', label='UCL')
            ax2.axhline(y=voltage_r.mean() - 3 * voltage_r.std(), color='r', linestyle='--', label='LCL')
            ax2.set_title('Voltage R Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Range (kV)')
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
            ax3.set_ylabel('Current (mA)')
            ax3.legend()
            
            # Plot current R chart
            current_r = data['current'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax4.plot(data['timestamp'], current_r, 'b-', label='Range')
            ax4.axhline(y=current_r.mean(), color='g', linestyle='-', label='Mean')
            ax4.axhline(y=current_r.mean() + 3 * current_r.std(), color='r', linestyle='--', label='UCL')
            ax4.axhline(y=current_r.mean() - 3 * current_r.std(), color='r', linestyle='--', label='LCL')
            ax4.set_title('Current R Chart')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Range (mA)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'hipot_spc')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate HiPot test data using simulated data acquisition.
        
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
            
            # Create DataFrame with numpy array data
            data = []
            for i in range(len(voltage_data['timestamps'])):
                data.append({
                    'timestamp': voltage_data['timestamps'][i],
                    'voltage': abs(voltage_data['data'][0][i]) * 5.0,  # Scale to kV
                    'current': abs(current_data['data'][0][i]) * 1.0,  # Scale to mA
                })
            
            df = pd.DataFrame(data)
            
            # Calculate pass/fail based on current threshold
            df['pass_fail'] = (df['current'] < 1.0).astype(int)  # Pass if current < 1mA
            
            logger.info(f"Generated {len(df)} samples of HiPot test data")
            return df
            
        except Exception as e:
            logger.error(f"Error generating HiPot test data: {str(e)}")
            raise
        finally:
            self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze HiPot test data.
        
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
            
            results = {
                'pass_rate': pass_rate,
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
            logger.error(f"Error analyzing HiPot test data: {str(e)}")
            raise

def main():
    """Main function to run the HiPot simulation."""
    try:
        # Create simulator
        simulator = HiPotSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=60)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nHiPot Test Results:")
        print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
        print("\nVoltage Statistics:")
        print(f"Mean: {results['voltage_stats']['mean']:.2f} kV")
        print(f"Std: {results['voltage_stats']['std']:.2f} kV")
        print(f"Min: {results['voltage_stats']['min']:.2f} kV")
        print(f"Max: {results['voltage_stats']['max']:.2f} kV")
        print("\nCurrent Statistics:")
        print(f"Mean: {results['current_stats']['mean']:.2f} mA")
        print(f"Std: {results['current_stats']['std']:.2f} mA")
        print(f"Min: {results['current_stats']['min']:.2f} mA")
        print(f"Max: {results['current_stats']['max']:.2f} mA")
        
    except Exception as e:
        logger.error(f"Error in HiPot simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 