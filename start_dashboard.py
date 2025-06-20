#!/usr/bin/env python3
"""
Simple script to start the web dashboard.
"""

import subprocess
import sys
import webbrowser
import time
import socket
import re

def find_available_port(start_port=8050, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def main():
    print("🚀 Starting Instrument Test Results Dashboard...")
    print("=" * 50)
    
    # Find available port
    port = find_available_port()
    if port is None:
        print("❌ No available ports found in range 8050-8059")
        return
    
    print(f"📍 Using port: {port}")
    
    try:
        # Start the dashboard
        print("Starting web server...")
        process = subprocess.Popen([
            sys.executable, 'dash_app/app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"❌ Dashboard failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return
        
        # Open the browser
        print("Opening dashboard in your browser...")
        webbrowser.open(f'http://localhost:{port}')
        
        print(f"\n✅ Dashboard is running!")
        print(f"📊 Open your browser to: http://localhost:{port}")
        print("🛑 Press Ctrl+C to stop the dashboard")
        
        # Keep the script running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            process.wait()
            print("✅ Dashboard stopped")
            
    except Exception as e:
        print(f"❌ Error starting dashboard: {str(e)}")
        print("\nMake sure you have:")
        print("1. Activated your virtual environment")
        print("2. Installed requirements: pip install -r requirements.txt")
        print("3. Run some tests first to generate data")

if __name__ == '__main__':
    main() 