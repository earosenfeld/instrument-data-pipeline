#!/usr/bin/env python3
"""
Script to view and explore test results data.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def list_available_results():
    """List all available test results."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("❌ No reports directory found. Run tests first.")
        return {}
    
    results = {}
    for test_dir in reports_dir.iterdir():
        if test_dir.is_dir():
            test_name = test_dir.name
            files = list(test_dir.glob("*"))
            
            # Count different file types
            png_files = [f for f in files if f.suffix == '.png']
            csv_files = [f for f in files if f.suffix == '.csv']
            json_files = [f for f in files if f.suffix == '.json']
            
            results[test_name] = {
                'plots': len(png_files),
                'data_files': len(csv_files),
                'stats_files': len(json_files),
                'latest_files': {
                    'png': max(png_files, key=lambda x: x.stat().st_mtime) if png_files else None,
                    'csv': max(csv_files, key=lambda x: x.stat().st_mtime) if csv_files else None,
                    'json': max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
                }
            }
    
    return results

def show_test_summary(test_name):
    """Show summary for a specific test."""
    json_file = Path(f"reports/{test_name}") / "*.json"
    json_files = list(Path("reports").glob(f"{test_name}/*.json"))
    
    if not json_files:
        print(f"❌ No results found for {test_name}")
        return
    
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_json, 'r') as f:
            stats = json.load(f)
        
        print(f"\n📊 {test_name.upper()} TEST SUMMARY")
        print("=" * 50)
        
        # Show pass/fail rates
        if 'failure_rate' in stats:
            print(f"Failure Rate: {stats['failure_rate']*100:.2f}%")
        elif 'pass_rate' in stats:
            print(f"Pass Rate: {stats['pass_rate']*100:.2f}%")
        elif 'overall_pass_rate' in stats:
            print(f"Overall Pass Rate: {stats['overall_pass_rate']*100:.2f}%")
        
        # Show statistics
        for key, value in stats.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"\n{key.replace('_', ' ').title()}:")
                print(f"  Mean: {value['mean']:.2f}")
                print(f"  Std: {value['std']:.2f}")
                print(f"  Min: {value['min']:.2f}")
                print(f"  Max: {value['max']:.2f}")
        
        print(f"\n📁 Files:")
        print(f"  Stats: {latest_json.name}")
        
        # Show other files
        test_dir = Path(f"reports/{test_name}")
        csv_files = list(test_dir.glob("*.csv"))
        png_files = list(test_dir.glob("*.png"))
        
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"  Data: {latest_csv.name}")
        
        if png_files:
            latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
            print(f"  Plot: {latest_png.name}")
            
    except Exception as e:
        print(f"❌ Error reading {test_name} results: {str(e)}")

def show_raw_data(test_name, num_rows=10):
    """Show raw data for a specific test."""
    csv_files = list(Path("reports").glob(f"{test_name}/*.csv"))
    
    if not csv_files:
        print(f"❌ No data files found for {test_name}")
        return
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    try:
        df = pd.read_csv(latest_csv)
        print(f"\n📋 {test_name.upper()} RAW DATA (first {num_rows} rows)")
        print("=" * 60)
        print(df.head(num_rows).to_string(index=False))
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error reading {test_name} data: {str(e)}")

def main():
    """Main function to view results."""
    print("🔍 Test Results Viewer")
    print("=" * 30)
    
    # List available results
    results = list_available_results()
    
    if not results:
        print("No test results found. Run tests first using:")
        print("python run_all_tests.py")
        return
    
    print(f"\n📁 Available test results ({len(results)} tests):")
    for test_name, info in results.items():
        print(f"  {test_name}: {info['plots']} plots, {info['data_files']} data files, {info['stats_files']} stats files")
    
    # Interactive menu
    while True:
        print(f"\n{'='*50}")
        print("OPTIONS:")
        print("1. Show summary for all tests")
        print("2. Show summary for specific test")
        print("3. Show raw data for specific test")
        print("4. List all available files")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("ALL TEST SUMMARIES")
            print("="*60)
            for test_name in results.keys():
                show_test_summary(test_name)
                print()
        
        elif choice == '2':
            test_name = input("Enter test name (burnin/hipot/isolation/laser/parametric/ict): ").strip()
            if test_name in results:
                show_test_summary(test_name)
            else:
                print(f"❌ Test '{test_name}' not found")
        
        elif choice == '3':
            test_name = input("Enter test name (burnin/hipot/isolation/laser/parametric/ict): ").strip()
            if test_name in results:
                num_rows = input("Number of rows to show (default 10): ").strip()
                num_rows = int(num_rows) if num_rows.isdigit() else 10
                show_raw_data(test_name, num_rows)
            else:
                print(f"❌ Test '{test_name}' not found")
        
        elif choice == '4':
            print("\n📁 ALL AVAILABLE FILES:")
            for test_name, info in results.items():
                print(f"\n{test_name.upper()}:")
                test_dir = Path(f"reports/{test_name}")
                for file_path in test_dir.iterdir():
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        print(f"  {file_path.name} ({size} bytes)")
        
        elif choice == '5':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 