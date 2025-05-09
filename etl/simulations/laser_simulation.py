import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
from typing import Dict
from etl.simulations.ethernet_sim import EthernetSimulator
from etl.simulations.daq_sim import DAQSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaserSimulator:
    """Simulates Laser Profile test data acquisition and analysis."""
    
    def __init__(self, connection_type: str = 'ethernet'):
        """
        Initialize the laser simulator.
        
        Args:
            connection_type (str): Type of connection to use ('ethernet' or 'daq')
        """
        self.connection_type = connection_type.lower()
        self.report_dir = 'reports/laser'
        os.makedirs(self.report_dir, exist_ok=True)
        
        if self.connection_type == 'ethernet':
            # Initialize network-connected instruments
            self.power_meter = EthernetSimulator(host='localhost', port=5025)
            self.wavelength_analyzer = EthernetSimulator(host='localhost', port=5026)
        else:
            # Initialize DAQ for analog measurements
            self.daq = DAQSimulator()
            
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
        
        # Calculate process capability indices
        power_mean = data['power'].mean()
        power_std = data['power'].std()
        power_cp = min(100 - power_mean, power_mean - 10) / (3 * power_std)
        power_cpk = min((100 - power_mean) / (3 * power_std), (power_mean - 10) / (3 * power_std))
        
        wavelength_mean = data['wavelength'].mean()
        wavelength_std = data['wavelength'].std()
        wavelength_cp = min(850 - wavelength_mean, wavelength_mean - 800) / (3 * wavelength_std)
        wavelength_cpk = min((850 - wavelength_mean) / (3 * wavelength_std), (wavelength_mean - 800) / (3 * wavelength_std))
        
        # Add capability indices to results
        results['power_capability'] = {
            'cp': power_cp,
            'cpk': power_cpk
        }
        results['wavelength_capability'] = {
            'cp': wavelength_cp,
            'cpk': wavelength_cpk
        }
        
        # Save to JSON file
        filename = f"laser_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"laser_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")

    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate Laser Profile test data using simulated instruments.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            if self.connection_type == 'ethernet':
                # Connect to network instruments
                if not self.power_meter.connect():
                    raise ConnectionError("Failed to connect to laser power meter")
                    
                if not self.wavelength_analyzer.connect():
                    raise ConnectionError("Failed to connect to wavelength analyzer")
                    
                # Configure instruments
                self.power_meter.send_command("CONF:POW:DC")
                self.wavelength_analyzer.send_command("CONF:WAV")
                
                # Read power and wavelength data
                power_data = self.power_meter.read_data(duration, num_channels=1)
                wavelength_data = self.wavelength_analyzer.read_data(duration, num_channels=1)
                
            else:
                # Connect to DAQ
                if not self.daq.connect('analog'):
                    raise ConnectionError("Failed to connect to DAQ")
                    
                # Read power and wavelength data from analog channels
                power_data = self.daq.read_data('analog', duration, num_channels=1)
                wavelength_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': power_data['timestamps'],
                'power': power_data['data'][0] * 50 + 55,  # Scale to 5-105 mW range
                'wavelength': power_data['data'][0] * 25 + 825,  # Scale to 800-850 nm range
            })
            
            # Calculate pass/fail based on power and wavelength thresholds
            df['pass_fail'] = ((df['power'] >= 10) & (df['power'] <= 100) & 
                             (df['wavelength'] >= 800) & (df['wavelength'] <= 850)).astype(int)
            
            logger.info(f"Generated {len(df)} samples of Laser Profile test data using {self.connection_type} connection")
            return df
            
        except Exception as e:
            logger.error(f"Error generating Laser Profile test data: {str(e)}")
            raise
        finally:
            if self.connection_type == 'ethernet':
                self.power_meter.disconnect()
                self.wavelength_analyzer.disconnect()
            else:
                self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze Laser Profile test data.
        
        Args:
            data (pd.DataFrame): Test data to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Calculate pass rate
            pass_rate = data['pass_fail'].mean()
            
            # Calculate power statistics
            power_stats = {
                'mean': data['power'].mean(),
                'std': data['power'].std(),
                'min': data['power'].min(),
                'max': data['power'].max()
            }
            
            # Calculate wavelength statistics
            wavelength_stats = {
                'mean': data['wavelength'].mean(),
                'std': data['wavelength'].std(),
                'min': data['wavelength'].min(),
                'max': data['wavelength'].max()
            }
            
            results = {
                'pass_rate': pass_rate,
                'power_stats': power_stats,
                'wavelength_stats': wavelength_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Plot and save results
            self._plot_results(data)
            self._plot_spc_charts(data)
            self._save_statistics(results, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing Laser Profile test data: {str(e)}")
            raise
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot Laser Profile test results."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot power
            ax1.plot(data['timestamp'], data['power'], label='Power (mW)')
            ax1.axhline(y=10, color='r', linestyle='--', label='Power Limits (10-100mW)')
            ax1.axhline(y=100, color='r', linestyle='--')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Power (mW)')
            ax1.set_title('Laser Power Over Time')
            ax1.legend()
            
            # Plot wavelength
            ax2.plot(data['timestamp'], data['wavelength'], label='Wavelength (nm)')
            ax2.axhline(y=800, color='r', linestyle='--', label='Wavelength Limits (800-850nm)')
            ax2.axhline(y=850, color='r', linestyle='--')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Wavelength (nm)')
            ax2.set_title('Laser Wavelength Over Time')
            ax2.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'laser_timeseries')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting Laser Profile test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Calculate control limits for power
            power_mean = data['power'].mean()
            power_std = data['power'].std()
            power_ucl = power_mean + 3 * power_std
            power_lcl = power_mean - 3 * power_std
            
            # Calculate control limits for wavelength
            wavelength_mean = data['wavelength'].mean()
            wavelength_std = data['wavelength'].std()
            wavelength_ucl = wavelength_mean + 3 * wavelength_std
            wavelength_lcl = wavelength_mean - 3 * wavelength_std
            
            # Plot power X-bar chart
            ax1.plot(data['timestamp'], data['power'], 'b-', label='Power')
            ax1.axhline(y=power_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=power_ucl, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=power_lcl, color='r', linestyle='--', label='LCL')
            ax1.set_title('Power X-bar Chart')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Power (mW)')
            ax1.legend()
            
            # Plot power R chart
            power_r = data['power'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax2.plot(data['timestamp'], power_r, 'b-', label='Range')
            ax2.axhline(y=power_r.mean(), color='g', linestyle='-', label='Mean')
            ax2.axhline(y=power_r.mean() + 3 * power_r.std(), color='r', linestyle='--', label='UCL')
            ax2.axhline(y=power_r.mean() - 3 * power_r.std(), color='r', linestyle='--', label='LCL')
            ax2.set_title('Power R Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Range (mW)')
            ax2.legend()
            
            # Plot wavelength X-bar chart
            ax3.plot(data['timestamp'], data['wavelength'], 'b-', label='Wavelength')
            ax3.axhline(y=wavelength_mean, color='g', linestyle='-', label='Mean')
            ax3.axhline(y=wavelength_ucl, color='r', linestyle='--', label='UCL')
            ax3.axhline(y=wavelength_lcl, color='r', linestyle='--', label='LCL')
            ax3.set_title('Wavelength X-bar Chart')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Wavelength (nm)')
            ax3.legend()
            
            # Plot wavelength R chart
            wavelength_r = data['wavelength'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax4.plot(data['timestamp'], wavelength_r, 'b-', label='Range')
            ax4.axhline(y=wavelength_r.mean(), color='g', linestyle='-', label='Mean')
            ax4.axhline(y=wavelength_r.mean() + 3 * wavelength_r.std(), color='r', linestyle='--', label='UCL')
            ax4.axhline(y=wavelength_r.mean() - 3 * wavelength_r.std(), color='r', linestyle='--', label='LCL')
            ax4.set_title('Wavelength R Chart')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Range (nm)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'laser_spc')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

def main():
    """Main function to run the Laser Profile simulation."""
    try:
        # Create simulator with ethernet connection
        simulator = LaserSimulator(connection_type='ethernet')
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=60)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nLaser Profile Test Results:")
        print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
        print("\nPower Statistics:")
        print(f"Mean: {results['power_stats']['mean']:.2f} mW")
        print(f"Std: {results['power_stats']['std']:.2f} mW")
        print(f"Min: {results['power_stats']['min']:.2f} mW")
        print(f"Max: {results['power_stats']['max']:.2f} mW")
        print("\nWavelength Statistics:")
        print(f"Mean: {results['wavelength_stats']['mean']:.2f} nm")
        print(f"Std: {results['wavelength_stats']['std']:.2f} nm")
        print(f"Min: {results['wavelength_stats']['min']:.2f} nm")
        print(f"Max: {results['wavelength_stats']['max']:.2f} nm")
        
    except Exception as e:
        logger.error(f"Error in Laser Profile simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 