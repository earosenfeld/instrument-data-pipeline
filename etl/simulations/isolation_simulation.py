import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict
from etl.simulations.daq_sim import DAQSimulator
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationSimulator:
    """Simulates Isolation Resistance test data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        self.report_dir = 'reports/isolation'
        os.makedirs(self.report_dir, exist_ok=True)
        
    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate Isolation Resistance test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for resistance measurement
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to resistance measurement system")
                
            # Read resistance data
            resistance_data = self.daq.read_data('analog', duration, num_channels=1)
            
            # Create DataFrame with numpy array data
            data = []
            for i in range(len(resistance_data['timestamps'])):
                data.append({
                    'timestamp': resistance_data['timestamps'][i],
                    'resistance': abs(resistance_data['data'][0][i]) * 1e6,  # Scale to MΩ
                })
            
            df = pd.DataFrame(data)
            
            # Calculate pass/fail based on resistance threshold
            df['pass_fail'] = (df['resistance'] >= 100.0).astype(int)  # Pass if resistance >= 100MΩ
            
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
            
            results = {
                'pass_rate': pass_rate,
                'resistance_stats': resistance_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Plot and save results
            self._plot_results(data)
            self._plot_spc_charts(data)
            self._save_statistics(results, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing Isolation Resistance test data: {str(e)}")
            raise
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot Isolation Resistance test results."""
        try:
            # Create figure with subplots
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Plot resistance over time
            ax.plot(data['timestamp'], data['resistance'], label='Resistance')
            ax.axhline(y=100.0, color='r', linestyle='--', label='Minimum (100MΩ)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Resistance (MΩ)')
            ax.set_title('Isolation Resistance Over Time')
            ax.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'isolation_timeseries')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting Isolation Resistance test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Calculate control limits
            resistance_mean = data['resistance'].mean()
            resistance_std = data['resistance'].std()
            
            # Plot X-bar chart
            ax1.plot(data['timestamp'], data['resistance'], 'b-', label='Resistance')
            ax1.axhline(y=resistance_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=resistance_mean + 3 * resistance_std, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=resistance_mean - 3 * resistance_std, color='r', linestyle='--', label='LCL')
            ax1.set_title('Isolation Resistance X-bar Chart')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Resistance (MΩ)')
            ax1.legend()
            
            # Plot R chart
            resistance_r = data['resistance'].rolling(window=2).apply(lambda x: x.max() - x.min())
            ax2.plot(data['timestamp'], resistance_r, 'b-', label='Range')
            ax2.axhline(y=resistance_r.mean(), color='g', linestyle='-', label='Mean')
            ax2.axhline(y=resistance_r.mean() + 3 * resistance_r.std(), color='r', linestyle='--', label='UCL')
            ax2.axhline(y=resistance_r.mean() - 3 * resistance_r.std(), color='r', linestyle='--', label='LCL')
            ax2.set_title('Isolation Resistance R Chart')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Range (MΩ)')
            ax2.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'isolation_spc')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

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
        filename = f"isolation_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"isolation_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")

def main():
    """Main function to run the Isolation Resistance simulation."""
    try:
        # Create simulator
        simulator = IsolationSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=10)
        
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
        
    except Exception as e:
        logger.error(f"Error in Isolation Resistance simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 