# Solution to MATLAB and Simulink Challenge project 249 Solar Tracker Control Simulation

![image](https://github.com/user-attachments/assets/91604bb0-f217-4016-881f-0210dcbf2815)

[Program link](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)

[Solar Tracker Control Simulation](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Solar%20Tracker%20Control%20Simulation)

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
- Month
- Day of the year (julian day)
- Sine of the hour
- Cosine of the hour
- DNI (Direct Normal Irradiance)
- DHI (Diffuse Horizontal Irradiance)
- GHI (Global Horizontal Irradiance)
- Azimuth of the sun
- Zenith of the sun
- Wind Speed
- Relative Humidity

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
├── Machine_Learning_model/
│   ├── nsrdb_raw/          # Raw NSRDB data (2018-2023)
│   ├── processed/          # Preprocessed training/validation/test sets
│   └── results/            # Simulation results and performance metrics
├── Machine_Learning_model/
│   ├── pvlib_solar_pos/    # Python PVlib solar position calculations
│   └── irradiance_analysis/ # PVlib-based performance comparison tools
├── WormAndGearConstraintSupport/
│   ├── main_simulation.mlx  # Main simulation interface (model selection)
│   └── test_scenarios/     # Sunny and cloudy day test cases
```

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

This project serves multiple research domains, providing a comprehensive framework for intelligent solar tracking through weather-predictive panel positioning that goes beyond traditional astronomical methods. The implementation offers valuable insights for machine learning applications in renewable energy, particularly through comparative analysis of Random Forest and Deep Learning models for solar optimization tasks. Additionally, the project serves as an educational resource for control systems engineering, featuring Simulink modeling techniques that integrate machine learning predictions with hardware simulation for dual-axis solar tracking mechanisms.

## Key Innovations

The project introduces several groundbreaking approaches to solar tracking technology. The system employs irradiance-optimized tracking where machine learning models predict panel positions that maximize total irradiance using PVlib's established Perez model, moving beyond sun-following algorithms to weather-driven optimization that accounts for atmospheric conditions affecting solar radiation. The comprehensive data integration approach utilizes a six-year NSRDB dataset with 10-minute temporal resolution, providing robust training data that captures seasonal variations and weather patterns specific to the Daytona Beach location. The dual algorithm methodology compares Random Forest and Deep Learning approaches for optimal positioning predictions, offering insights into model performance under varying atmospheric conditions. The physics-based target generation employs the Perez model for diffuse irradiance calculations, ensuring that the machine learning models learn from  optimal positions. Finally, the real-world validation framework uses actual weather scenarios from the 2023 dataset with irradiance-based performance metrics, demonstrating practical applicability of the predictive models under diverse meteorological conditions.

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

### Quick start
1. Clone this repository
2. To start the simulation, open Solar_tracker_simulation.mlx and read the instructions. It is advisable to execute it section by section, particularly when using an external IDE for the Python code, to prevent potential errors.
3. Choose the test scenario (sunny/cloudy day) from the data available in the data_for_simulation folder
4. Select between Random Forest or MLP model
5. If you plan to run Python code from MATLAB, follow the instructions provided in the script. Otherwise, you will need to use an external IDE and set up the working environment in that IDE for PVlib.

To visualize simulations with additional data, use the CSV data tables located in the data_for_simulation folder. You can modify or select the desired time period, although the available data only covers the years 2023 and 2024. If you need more recent or real-time data, you must obtain it from a data provider.

**Note**: The trained models are specifically optimized for Daytona Beach, Florida conditions. For other locations, you'll need to retrain the models using the provided scripts with location-specific NSRDB data or other sources.

### Model Training (Optional)
1. Prepare NSRDB dataset using provided preprocessing scripts
2. Calculate optimal panel positions using the PVlib code Irradiation_pvlib.py
3. Run MRMR feature selection for Random Forest model in MATLAB
4. Train Deep Learning model with iterative variable selection in MATLAB
5. Validate models using 2022 data and test with 2023 data
6. Use Python PVlib tools for irradiance analysis and performance comparison

**Important**: Models are location-specific. To use this system for locations other than Daytona Beach, Florida, you must retrain both models using local NSRDB data following the same methodology.

# Simulation and Results


| ![RF_solar_position_sunny_day](https://github.com/user-attachments/assets/7a89f703-dfab-4df9-a311-b5f336b25d6c) | ![SPA_solar_position_sunny_day](https://github.com/user-attachments/assets/de0e15b2-a5c5-4b57-a327-5de52a694c28) |
|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|



## Citation

If you use this work in your research, please cite:
```
[Your Name] (2025). Intelligent Solar Tracker Control System using Machine Learning 
and NSRDB Data. GitHub Repository: [repository_url]
```

# Reference
Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M., and Driesse, A. “pvlib python: 2023 project update.” Journal of Open Source Software, 8(92), 5994, (2023). DOI: 10.21105/joss.05994.

Sengupta, M., Xie, Y., Lopez, A., Habte, A., Maclaurin, G., Shelby, J., 2018. The National Solar Radiation Data Base (NSRDB). Renew. Sustain. Energy Rev. 89, 51-60. https://doi.org/10.1016/j.rser.2018.03.003.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
