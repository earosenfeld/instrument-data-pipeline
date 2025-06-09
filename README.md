# Instrument Data Pipeline

This repository contains a suite of simulation tools for generating and analyzing test data for various electronic component tests, including burn-in, HiPot, isolation resistance, laser profile, parametric, and in-circuit tests.

## Overview

The project simulates data acquisition and analysis for different test types, generating realistic test data, plotting results, and saving statistics and raw data for further analysis.

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

## Customization

- Adjust the test duration in each simulation script's `main()` function to generate more or fewer samples.
- Modify the simulation parameters in the respective simulator classes to suit your testing needs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.