# Instrument Data Pipeline

This repository contains a suite of simulation tools for generating and analyzing test data for various electronic component tests, including burn-in, HiPot, isolation resistance, laser profile, parametric, and in-circuit tests.

## Overview

The project simulates data acquisition and analysis for different test types, generating realistic test data, plotting results, and saving statistics and raw data for further analysis. It includes both command-line tools and a web-based dashboard for viewing results.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd instrument-data-pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install additional dependencies for the web dashboard:
   ```bash
   pip install dash plotly
   ```

## Usage

### Running Simulations

Each simulation script can be run as a module. For example, to run the burn-in simulation:

```bash
python -m etl.simulations.burnin_simulation
```

Other available simulations:
- HiPot: `python -m etl.simulations.hipot_simulation`
- Isolation: `python -m etl.simulations.isolation_simulation`
- Laser: `python -m etl.simulations.laser_simulation`
- Parametric: `python -m etl.simulations.parametric_simulation`
- ICT: `python -m etl.simulations.ict_simulation`

### Running All Tests at Once

Use the master script to run all simulations:

```bash
python run_all_tests.py
```

This will:
- Run all 6 test simulations
- Generate plots, data files, and statistics
- Provide a summary of results

### Web Dashboard

After running tests, you can view results in an interactive web dashboard:

```bash
python simple_dashboard.py
```

The dashboard will:
- Automatically find an available port (starting from 8050)
- Open your browser to the dashboard
- Display interactive charts and statistics for all tests

**Dashboard Features:**
- **Test Selector**: Choose which test to view
- **Summary Statistics**: Pass/fail rates and key metrics
- **Time Series Plots**: Interactive charts showing data over time
- **Raw Data Tables**: View the actual test data
- **Generated Images**: View the PNG plots created by simulations

### Viewing Results

- **Console Output:** Each simulation prints summary statistics and results to the terminal.
- **Generated Files:** Detailed results, plots, and raw data are saved in the `reports/` directory:
  - `reports/burnin/`
  - `reports/hipot/`
  - `reports/isolation/`
  - `reports/laser/`
  - `reports/parametric/`
  - `reports/ict/`

  Each folder contains:
  - **PNG files:** Plots and SPC charts.
  - **CSV files:** Raw data.
  - **JSON files:** Statistics and summary results.

### Interactive Data Viewer

For command-line exploration of results:

```bash
python view_results.py
```

This provides an interactive menu to:
- See summaries of all tests
- View specific test results
- Explore raw data
- List all generated files

## Quick Start Guide

1. **Set up environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install dash plotly
   ```

2. **Run all tests:**
   ```bash
   python run_all_tests.py
   ```

3. **View results in web dashboard:**
   ```bash
   python simple_dashboard.py
   ```

4. **Explore data:**
   ```bash
   python view_results.py
   ```

## Customization

- Adjust the test duration in each simulation script's `main()` function to generate more or fewer samples.
- Modify the simulation parameters in the respective simulator classes to suit your testing needs.
- The web dashboard automatically finds available ports to avoid conflicts.

## Troubleshooting

- **Port conflicts**: The dashboard automatically finds available ports starting from 8050
- **Import errors**: Make sure you've activated the virtual environment and installed the package
- **No data**: Run the tests first before viewing the dashboard
- **Slow performance**: Reduce test duration for quicker results

## License

This project is licensed under the MIT License - see the LICENSE file for details.