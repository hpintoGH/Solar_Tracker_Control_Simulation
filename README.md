# Solution to MATLAB and Simulink Challenge project 249 Solar Tracker Control Simulation

[Program link](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)

[projects/Solar Tracker Control Simulation](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Solar%20Tracker%20Control%20Simulation)


# Project details

A comprehensive MATLAB/Simulink simulation framework for intelligent solar tracking systems using machine learning models to optimize photovoltaic panel positioning based on real-world weather data and solar radiation patterns.

## Overview

This repository contains a complete intelligent solar tracker control system that combines machine learning prediction models with Simulink simulation to control dual-axis solar tracking mechanisms. The system uses trained models (Random Forest and Deep Learning) to predict optimal panel positioning based on weather conditions and solar radiation data, providing an alternative to traditional astronomical sun-tracking methods.

## Key Features

- **Machine Learning-Based Tracking**: Two trained models for intelligent panel positioning
  - Random Forest model with MRMR feature selection
  - Multi-Layer Perceptron (MLP) with 4 fully-connected layers
- **Real-World Data Integration**: NSRDB (National Solar Radiation Database) 2018-2023 dataset
- **Comprehensive Simulation Environment**: Full Simulink model with NEMA 17 stepper motor control
- **Performance Comparison Tools**: Python-based irradiance analysis using PVlib comparing ML vs. astronomical tracking
- **Location-Specific Training**: Models trained for Daytona Beach, Florida conditions
- **Flexible Solar Position Calculation**: Multiple algorithms including NREL's SPA and Python PVlib integration

## Dataset and Training

### Data Source
- **NSRDB (National Solar Radiation Database)** - 6 years of data (2018-2023)
- **Location**: Daytona Beach, Florida
- **Temporal Resolution**: 10-minute intervals
- **Geographic Scope**: Location-specific training (models optimized for Florida climate conditions)

### Input Variables
- **DNI (Direct Normal Irradiance)**
- **DHI (Diffuse Horizontal Irradiance)**  
- **GHI (Global Horizontal Irradiance)**
- **Relative Humidity**
- **Wind Speed**
- **Albedo** (for PVlib irradiance calculations)
- **Optional: Temperature** (for solar position calculations with PVlib)

### Target Variable

- **Optimal Panel Position** (azimuth and elevation angles)
- **Calculation Method:** PVlib's get_total_irradiance function using the Perez model
- **Input Parameters:** Solar zenith angle, azimuth angle, DNI, DHI, GHI, extraterrestrial irradiance, and albedo
- **Optimization Objective:** Maximum total irradiance on tilted panel surface

### Training Strategy
- **Training Period**: 2018-2021 (4 years)
- **Validation Period**: 2022 (1 year)
- **Test Period**: 2023 (1 year)
- **Feature Selection**: MRMR (Maximum Relevance Minimum Redundancy) for Random Forest
- **Deep Learning Optimization**: Iterative variable selection based on Random Forest results
- **Training Platform**: MATLAB Machine Learning Toolbox for both Random Forest and Deep Learning models

## Machine Learning Models

### Random Forest Model 
- Feature selection using MRMR algorithm
- Optimized variable subset for weather-based tracking prediction
- Robust performance across varying weather conditions

### Deep Learning Model (MLP)
- 4 fully-connected layers
- Input variables refined through iterative training
- Enhanced prediction accuracy for complex weather patterns

## Solar Position Algorithms

### Primary: NREL's Solar Position Algorithm (SPA)
- Implementation: Meysam Mahooti's MATLAB adaptation
- **Reference**: Meysam Mahooti (2025). NREL's Solar Position Algorithm (SPA), MATLAB Central File Exchange
- **Custom Integration**: Script for performing time-based calculations with user-defined intervals

### Alternative:
- Python PVlib Integration
- NOAA Solar Calculator

## Simulation Environment

### Simulink Model Components
- **Dual-axis tracking system** with azimuth and elevation control
- **NEMA 17 stepper motors** (1.8° per step resolution)
- **Real-time weather data integration**
- **Model selection interface** (Random Forest or Deep Learning)

### Test Scenarios
- **Sunny Day Simulation**: Clear sky conditions from 2023 dataset
- **Cloudy Day Simulation**: Overcast conditions from 2023 dataset
- **Comparative Analysis**: ML-based vs. astronomical tracking performance

## File Structure

```
├── src/
│   ├── ml_models/           # Trained Random Forest and Deep Learning models
│   ├── solar_position/      # NREL SPA and PVlib implementations
│   ├── simulink/           # Simulink models and motor control
│   ├── data_processing/    # NSRDB data preprocessing scripts
│   └── feature_selection/  # MRMR and variable selection algorithms
├── data/
│   ├── nsrdb_raw/          # Raw NSRDB data (2018-2023)
│   ├── processed/          # Preprocessed training/validation/test sets
│   └── results/            # Simulation results and performance metrics
├── python_tools/
│   ├── pvlib_solar_pos/    # Python PVlib solar position calculations
│   └── irradiance_analysis/ # PVlib-based performance comparison tools
├── simulation/
│   ├── main_simulation.mlx  # Main simulation interface (model selection)
│   └── test_scenarios/     # Sunny and cloudy day test cases
└── docs/                   # Documentation and methodology
```

### Quick Start
1. Clone this repository
2. Download NSRDB data for Daytona Beach, Florida (2018-2023) or your desired location
3. Run data preprocessing scripts in `src/data_processing/`
4. Execute `simulation/main_simulation.mlx` to start the simulation
5. Select between Random Forest or Deep Learning model
6. Choose test scenario (sunny/cloudy day)

**Note**: The trained models are specifically optimized for Daytona Beach, Florida conditions. For other locations, you'll need to retrain the models using the provided scripts with location-specific NSRDB data.

### Model Training (Optional)
1. Prepare NSRDB dataset using provided preprocessing scripts
2. Run MRMR feature selection for Random Forest model in MATLAB
3. Train Deep Learning model with iterative variable selection in MATLAB
4. Validate models using 2022 data and test with 2023 data
5. Use Python PVlib tools for irradiance analysis and performance comparison

**Important**: Models are location-specific. To use this system for locations other than Daytona Beach, Florida, you must retrain both models using local NSRDB data following the same methodology.

## Performance Metrics

### Tracking Accuracy
- **ML Model Precision**: Comparison against astronomical tracking
- **Irradiance Capture Efficiency**: Energy collection optimization
- **Weather Adaptation**: Performance under varying conditions

### Hardware Specifications
- **Motor Resolution**: 1.8° per step (NEMA 17)
- **Positioning Accuracy**: Sub-degree precision
- **Response Time**: Real-time weather-based adjustments

## Research Applications

- **Intelligent Solar Tracking**: Weather-predictive panel positioning
- **Machine Learning in Renewable Energy**: Comparative ML model analysis
- **Solar Resource Assessment**: NSRDB data utilization methodologies
- **Control Systems Education**: Advanced Simulink modeling techniques

## Key Innovations

1. **Weather-Based Tracking**: ML models predict optimal positioning based on atmospheric conditions
2. **Comprehensive Data Integration**: 6-year NSRDB dataset with 10-minute resolution
3. **Dual Algorithm Approach**: Random Forest and Deep Learning model comparison
4. **Cross-Platform Validation**: MATLAB/Python integration for result verification
5. **Real-World Testing**: Actual weather scenarios from 2023 dataset

## Dependencies

### MATLAB/Simulink
- Machine Learning Toolbox (for Random Forest and Deep Learning training)
- Control System Toolbox
- Signal Processing Toolbox

### Python
- PVlib (for solar position calculations and irradiance analysis)
- pandas
- numpy
- matplotlib

## Contributing

Contributions welcome! Areas of interest:
- **Location Adaptation**: Scripts for retraining models for different geographic locations
- **Additional ML Models**: Enhanced weather prediction algorithms
- **Extended Geographic Validation**: Multi-location performance analysis
- **Hardware Integration**: Real-world implementation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:
```
[Your Name] (2025). Intelligent Solar Tracker Control System using Machine Learning 
and NSRDB Data. GitHub Repository: [repository_url]
```

## Acknowledgments

- **NREL** for the Solar Position Algorithm and NSRDB dataset
- **Meysam Mahooti** for the MATLAB SPA implementation
- **PVlib Development Team** for the photovoltaic modeling library
- **NSRDB Contributors** for the comprehensive solar radiation database

# How to run section

## Installation and Usage

### Prerequisites
- MATLAB R2024b or later
- Simulink
- Simscape
- Simscape Multibody
- Simscape Electrical
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Deep Learning Toolbox
- Python 3.9+ with PVlib library and Matplotlib (for irradiance analysis)

### Quick Start
1. Clone this repository
2. Download NSRDB data for Daytona Beach, Florida (2018-2023) or your desired location
3. Run data preprocessing scripts in `src/data_processing/`
4. Execute `simulation/main_simulation.mlx` to start the simulation
5. Select between Random Forest or Deep Learning model
6. Choose test scenario (sunny/cloudy day)

**Note**: The trained models are specifically optimized for Daytona Beach, Florida conditions. For other locations, you'll need to retrain the models using the provided scripts with location-specific NSRDB data.

# Demo
Add a video or animated gif/picture to showcase the code in operation.
  
# Reference
Add reference papers, data, or supporting material that has been used, if any.
