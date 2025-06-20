#!/usr/bin/env python3
"""
Master script to run all simulation tests and provide a summary of results.
"""

import subprocess
import sys
import os
from datetime import datetime
import json
import pandas as pd

def run_simulation(simulation_name, module_path):
    """Run a single simulation and return the results."""
    print(f"\n{'='*60}")
    print(f"Running {simulation_name} simulation...")
    print(f"{'='*60}")
    
    try:
        # Run the simulation
        result = subprocess.run([
            sys.executable, '-m', module_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            print(result.stdout)
            return True, result.stdout
        else:
            print("❌ FAILED")
            print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT - Test took too long")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False, str(e)

def collect_results():
    """Collect and summarize results from all test reports."""
    print(f"\n{'='*60}")
    print("COLLECTING RESULTS SUMMARY")
    print(f"{'='*60}")
    
    results_summary = {}
    
    # Define test directories and their expected files
    test_dirs = {
        'burnin': 'reports/burnin',
        'hipot': 'reports/hipot', 
        'isolation': 'reports/isolation',
        'laser': 'reports/laser',
        'parametric': 'reports/parametric',
        'ict': 'reports/ict'
    }
    
    for test_name, test_dir in test_dirs.items():
        if os.path.exists(test_dir):
            # Find the most recent JSON stats file
            json_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
            if json_files:
                latest_json = max(json_files, key=lambda x: os.path.getctime(os.path.join(test_dir, x)))
                json_path = os.path.join(test_dir, latest_json)
                
                try:
                    with open(json_path, 'r') as f:
                        stats = json.load(f)
                    results_summary[test_name] = stats
                    print(f"✅ {test_name.upper()}: Found results")
                except Exception as e:
                    print(f"❌ {test_name.upper()}: Error reading results - {str(e)}")
            else:
                print(f"⚠️  {test_name.upper()}: No JSON results found")
        else:
            print(f"❌ {test_name.upper()}: Directory not found")
    
    return results_summary

def print_summary(results_summary):
    """Print a formatted summary of all test results."""
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for test_name, results in results_summary.items():
        print(f"\n{test_name.upper()} TEST:")
        print("-" * 40)
        
        if 'failure_rate' in results:
            print(f"Failure Rate: {results['failure_rate']*100:.2f}%")
        elif 'pass_rate' in results:
            print(f"Pass Rate: {results['pass_rate']*100:.2f}%")
        elif 'overall_pass_rate' in results:
            print(f"Overall Pass Rate: {results['overall_pass_rate']*100:.2f}%")
        
        # Print key statistics
        for key, value in results.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"{key.replace('_', ' ').title()}:")
                print(f"  Mean: {value['mean']:.2f}")
                print(f"  Std: {value['std']:.2f}")
                print(f"  Min: {value['min']:.2f}")
                print(f"  Max: {value['max']:.2f}")

def main():
    """Main function to run all tests and collect results."""
    print("🚀 Starting all simulation tests...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define all simulations to run
    simulations = [
        ("Burn-In Test", "etl.simulations.burnin_simulation"),
        ("HiPot Test", "etl.simulations.hipot_simulation"),
        ("Isolation Test", "etl.simulations.isolation_simulation"),
        ("Laser Test", "etl.simulations.laser_simulation"),
        ("Parametric Test", "etl.simulations.parametric_simulation"),
        ("ICT Test", "etl.simulations.ict_simulation")
    ]
    
    # Run all simulations
    results = {}
    for test_name, module_path in simulations:
        success, output = run_simulation(test_name, module_path)
        results[test_name] = {'success': success, 'output': output}
    
    # Collect and display results summary
    results_summary = collect_results()
    print_summary(results_summary)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    print(f"Tests completed: {successful_tests}/{total_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n📁 Results saved in: reports/")
    print(f"📊 View plots and data in the respective test directories")

if __name__ == "__main__":
    main() 