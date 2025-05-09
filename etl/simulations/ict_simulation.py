import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
import os
from typing import Dict, List, Tuple
from etl.simulations.daq_sim import DAQSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICTSimulator:
    """Simulates In-Circuit Test (ICT) data acquisition and analysis."""
    
    def __init__(self):
        self.daq = DAQSimulator()
        self.report_dir = 'reports/ict'
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Define test points and their specifications
        self.test_points = {
            'continuity': {
                'points': ['TP1', 'TP2', 'TP3', 'TP4', 'TP5'],
                'resistance_limit': 1.0  # Ohms
            },
            'resistors': {
                'points': ['R1', 'R2', 'R3', 'R4'],
                'tolerance': 0.05  # 5%
            },
            'capacitors': {
                'points': ['C1', 'C2', 'C3'],
                'tolerance': 0.10  # 10%
            },
            'digital': {
                'points': ['D1', 'D2', 'D3', 'D4'],
                'voltage_high': 3.3,
                'voltage_low': 0.0
            },
            'power': {
                'points': ['VCC', 'GND'],
                'voltage_nominal': 3.3,
                'tolerance': 0.05  # 5%
            }
        }
        
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
        filename = f"ict_stats_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved statistics to {filepath}")
        
        # Save raw data to CSV
        csv_filename = f"ict_data_{timestamp}.csv"
        csv_filepath = os.path.join(self.report_dir, csv_filename)
        data.to_csv(csv_filepath, index=False)
        logger.info(f"Saved raw data to {csv_filepath}")
    
    def _plot_results(self, data: pd.DataFrame) -> None:
        """Plot ICT test results."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot continuity test results
            continuity_data = data[data['test_type'] == 'continuity']
            ax1.bar(continuity_data['test_point'], continuity_data['resistance'])
            ax1.axhline(y=self.test_points['continuity']['resistance_limit'], 
                       color='r', linestyle='--', label='Limit')
            ax1.set_title('Continuity Test Results')
            ax1.set_xlabel('Test Point')
            ax1.set_ylabel('Resistance (Ω)')
            ax1.legend()
            
            # Plot resistor measurements
            resistor_data = data[data['test_type'] == 'resistor']
            ax2.bar(resistor_data['test_point'], resistor_data['value'])
            ax2.set_title('Resistor Measurements')
            ax2.set_xlabel('Test Point')
            ax2.set_ylabel('Resistance (Ω)')
            
            # Plot capacitor measurements
            cap_data = data[data['test_type'] == 'capacitor']
            ax3.bar(cap_data['test_point'], cap_data['value'])
            ax3.set_title('Capacitor Measurements')
            ax3.set_xlabel('Test Point')
            ax3.set_ylabel('Capacitance (μF)')
            
            # Plot power supply measurements
            power_data = data[data['test_type'] == 'power']
            ax4.bar(power_data['test_point'], power_data['value'])
            ax4.axhline(y=self.test_points['power']['voltage_nominal'], 
                       color='g', linestyle='-', label='Nominal')
            ax4.axhline(y=self.test_points['power']['voltage_nominal'] * 
                       (1 + self.test_points['power']['tolerance']), 
                       color='r', linestyle='--', label='Limits')
            ax4.axhline(y=self.test_points['power']['voltage_nominal'] * 
                       (1 - self.test_points['power']['tolerance']), 
                       color='r', linestyle='--')
            ax4.set_title('Power Supply Measurements')
            ax4.set_xlabel('Test Point')
            ax4.set_ylabel('Voltage (V)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'ict_measurements')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting ICT test results: {str(e)}")
            raise
            
    def _plot_spc_charts(self, data: pd.DataFrame) -> None:
        """Plot Statistical Process Control (SPC) charts."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot power supply SPC
            power_data = data[data['test_type'] == 'power']
            power_mean = power_data['value'].mean()
            power_std = power_data['value'].std()
            
            ax1.plot(power_data['sequence_num'], power_data['value'], 'b-', label='Voltage')
            ax1.axhline(y=power_mean, color='g', linestyle='-', label='Mean')
            ax1.axhline(y=power_mean + 3 * power_std, color='r', linestyle='--', label='UCL')
            ax1.axhline(y=power_mean - 3 * power_std, color='r', linestyle='--', label='LCL')
            ax1.set_title('Power Supply X-bar Chart')
            ax1.set_xlabel('Measurement Sequence')
            ax1.set_ylabel('Voltage (V)')
            ax1.legend()
            
            # Plot resistor SPC
            resistor_data = data[data['test_type'] == 'resistor']
            resistor_mean = resistor_data['value'].mean()
            resistor_std = resistor_data['value'].std()
            
            ax2.plot(resistor_data['sequence_num'], resistor_data['value'], 'b-', label='Resistance')
            ax2.axhline(y=resistor_mean, color='g', linestyle='-', label='Mean')
            ax2.axhline(y=resistor_mean + 3 * resistor_std, color='r', linestyle='--', label='UCL')
            ax2.axhline(y=resistor_mean - 3 * resistor_std, color='r', linestyle='--', label='LCL')
            ax2.set_title('Resistor X-bar Chart')
            ax2.set_xlabel('Measurement Sequence')
            ax2.set_ylabel('Resistance (Ω)')
            ax2.legend()
            
            # Plot capacitor SPC
            cap_data = data[data['test_type'] == 'capacitor']
            cap_mean = cap_data['value'].mean()
            cap_std = cap_data['value'].std()
            
            ax3.plot(cap_data['sequence_num'], cap_data['value'], 'b-', label='Capacitance')
            ax3.axhline(y=cap_mean, color='g', linestyle='-', label='Mean')
            ax3.axhline(y=cap_mean + 3 * cap_std, color='r', linestyle='--', label='UCL')
            ax3.axhline(y=cap_mean - 3 * cap_std, color='r', linestyle='--', label='LCL')
            ax3.set_title('Capacitor X-bar Chart')
            ax3.set_xlabel('Measurement Sequence')
            ax3.set_ylabel('Capacitance (μF)')
            ax3.legend()
            
            # Plot continuity SPC
            continuity_data = data[data['test_type'] == 'continuity']
            continuity_mean = continuity_data['resistance'].mean()
            continuity_std = continuity_data['resistance'].std()
            
            ax4.plot(continuity_data['sequence_num'], continuity_data['resistance'], 'b-', label='Resistance')
            ax4.axhline(y=continuity_mean, color='g', linestyle='-', label='Mean')
            ax4.axhline(y=continuity_mean + 3 * continuity_std, color='r', linestyle='--', label='UCL')
            ax4.axhline(y=continuity_mean - 3 * continuity_std, color='r', linestyle='--', label='LCL')
            ax4.set_title('Continuity X-bar Chart')
            ax4.set_xlabel('Measurement Sequence')
            ax4.set_ylabel('Resistance (Ω)')
            ax4.legend()
            
            plt.tight_layout()
            self._save_plot(fig, 'ict_spc')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error plotting SPC charts: {str(e)}")
            raise

    def generate_test_data(self, duration: float = 60) -> pd.DataFrame:
        """
        Generate ICT test data using simulated data acquisition.
        
        Args:
            duration (float): Duration of the test in seconds
            
        Returns:
            pd.DataFrame: Generated test data
        """
        try:
            # Connect to analog DAQ for measurements
            if not self.daq.connect('analog'):
                raise ConnectionError("Failed to connect to measurement system")
            
            # Initialize data storage
            data = []
            
            # Generate continuity test data
            for idx, point in enumerate(self.test_points['continuity']['points']):
                resistance = np.random.normal(0.5, 0.1)  # Mean 0.5Ω, std 0.1Ω
                data.append({
                    'sequence_num': idx + 1,
                    'test_type': 'continuity',
                    'test_point': point,
                    'resistance': resistance,
                    'pass_fail': resistance < self.test_points['continuity']['resistance_limit']
                })
            
            # Generate resistor measurements
            start_idx = len(data)
            for idx, point in enumerate(self.test_points['resistors']['points']):
                value = np.random.normal(1000, 50)  # Mean 1kΩ, std 50Ω
                data.append({
                    'sequence_num': start_idx + idx + 1,
                    'test_type': 'resistor',
                    'test_point': point,
                    'value': value,
                    'pass_fail': abs(value - 1000) / 1000 <= self.test_points['resistors']['tolerance']
                })
            
            # Generate capacitor measurements
            start_idx = len(data)
            for idx, point in enumerate(self.test_points['capacitors']['points']):
                value = np.random.normal(10, 0.5)  # Mean 10μF, std 0.5μF
                data.append({
                    'sequence_num': start_idx + idx + 1,
                    'test_type': 'capacitor',
                    'test_point': point,
                    'value': value,
                    'pass_fail': abs(value - 10) / 10 <= self.test_points['capacitors']['tolerance']
                })
            
            # Generate power supply measurements
            start_idx = len(data)
            for idx, point in enumerate(self.test_points['power']['points']):
                value = np.random.normal(
                    self.test_points['power']['voltage_nominal'],
                    self.test_points['power']['voltage_nominal'] * 0.01
                )
                data.append({
                    'sequence_num': start_idx + idx + 1,
                    'test_type': 'power',
                    'test_point': point,
                    'value': value,
                    'pass_fail': abs(value - self.test_points['power']['voltage_nominal']) / \
                               self.test_points['power']['voltage_nominal'] <= \
                               self.test_points['power']['tolerance']
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            logger.info(f"Generated {len(df)} samples of ICT test data")
            return df
            
        except Exception as e:
            logger.error(f"Error generating ICT test data: {str(e)}")
            raise
        finally:
            self.daq.disconnect()
    
    def analyze_test_data(self, data: pd.DataFrame) -> Dict:
        """
        Analyze ICT test data.
        
        Args:
            data (pd.DataFrame): Test data to analyze
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Calculate overall pass rate
            pass_rate = data['pass_fail'].mean()
            
            # Calculate statistics by test type
            results = {
                'overall_pass_rate': pass_rate,
                'test_type_stats': {},
                'timestamp': datetime.now().isoformat()
            }
            
            for test_type in data['test_type'].unique():
                type_data = data[data['test_type'] == test_type]
                
                # Handle different value columns for different test types
                if test_type == 'continuity':
                    value_col = 'resistance'
                else:
                    value_col = 'value'
                
                results['test_type_stats'][test_type] = {
                    'pass_rate': type_data['pass_fail'].mean(),
                    'mean': type_data[value_col].mean(),
                    'std': type_data[value_col].std(),
                    'min': type_data[value_col].min(),
                    'max': type_data[value_col].max()
                }
            
            # Plot and save results
            self._plot_results(data)
            self._plot_spc_charts(data)
            self._save_statistics(results, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing ICT test data: {str(e)}")
            raise

def main():
    """Main function to run the ICT simulation."""
    try:
        # Create simulator
        simulator = ICTSimulator()
        
        # Generate test data (1 minute duration)
        data = simulator.generate_test_data(duration=60)
        
        # Analyze test data
        results = simulator.analyze_test_data(data)
        
        # Print results
        print("\nICT Test Results:")
        print(f"Overall Pass Rate: {results['overall_pass_rate']*100:.2f}%")
        
        for test_type, stats in results['test_type_stats'].items():
            print(f"\n{test_type.title()} Statistics:")
            print(f"Pass Rate: {stats['pass_rate']*100:.2f}%")
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Std: {stats['std']:.2f}")
            print(f"Min: {stats['min']:.2f}")
            print(f"Max: {stats['max']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in ICT simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 