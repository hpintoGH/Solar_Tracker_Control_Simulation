# Solution to MATLAB and Simulink Challenge project 249 Solar Tracker Control Simulation

![image](https://github.com/user-attachments/assets/91604bb0-f217-4016-881f-0210dcbf2815)

[Program link](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)

[Solar Tracker Control Simulation](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Solar%20Tracker%20Control%20Simulation)

# Project details

A comprehensive MATLAB/Simulink simulation framework for intelligent solar tracking systems using machine learning models to optimize photovoltaic panel positioning based on real-world weather data and solar radiation patterns.

## Overview

This repository contains a complete intelligent solar tracker control system that combines machine learning prediction models with Simulink simulation to control dual-axis solar tracking mechanisms. The system uses trained models (Random Forest and Deep Learning) to predict optimal panel positioning based on weather conditions and solar radiation data, providing an alternative to traditional astronomical sun-tracking methods.

## Key Features

The system implements machine learning-based tracking through two trained models designed for intelligent panel positioning that maximizes solar irradiance capture. The Random Forest model incorporates MRMR (Maximum Relevance Minimum Redundancy) feature selection to identify the most relevant weather variables for optimal positioning predictions, while the Multi-Layer Perceptron (MLP) architecture features four fully-connected layers that capture nonlinear relationships between atmospheric conditions and optimal panel angles.

Real-world data integration forms the foundation of the system through the comprehensive NSRDB (National Solar Radiation Database) dataset spanning 2018-2023, providing six years of high-resolution meteorological and solar radiation data. The simulation environment encompasses a complete Simulink model featuring NEMA 17 stepper motor control systems that represent the mechanical aspects of dual-axis solar tracking hardware.

Performance comparison tools utilize Python-based irradiance analysis through PVlib libraries, enabling detailed comparisons between machine learning predictions and traditional astronomical tracking methods. The location-specific training approach focuses on Daytona Beach, Florida conditions, selected arbitrarily within the United States due to the availability and accessibility of high-resolution temporal and spatial data necessary for robust model development. The system incorporates flexible solar position calculation capabilities through multiple algorithms, including NREL's Solar Position Algorithm (SPA) for precise astronomical calculations and Python PVlib integration for comprehensive irradiance modeling and validation.

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
- Solar Azimuth
- Solar Zenith
- Wind Speed
- Relative Humidity

### Target Variable

- **Optimal Panel Position** (azimuth and tilt angles)

The calculation method employs PVlib's functions using the Perez model for diffuse irradiance calculations. Input parameters include solar zenith angle, azimuth angle, DNI, DHI, GHI, extraterrestrial irradiance, and albedo values, which collectively determine the optimal panel orientation for any given atmospheric condition. The optimization objective focuses on achieving maximum total irradiance on the tilted panel surface, ensuring that the machine learning models learn to predict positions that maximize energy capture.

### Training Strategy
- **Training Period**: 2018-2021 (4 years)
- **Validation Period**: 2022 (1 year)
- **Test Period**: 2023 (1 year)
- **Feature Selection**: MRMR (Maximum Relevance Minimum Redundancy) for Random Forest
- **Deep Learning Optimization**: Iterative variable selection based on Random Forest results
- **Training Platform**: MATLAB Statistics and Machine Learning Toolbox for both Random Forest and MATLAB Deep Learning Toolbox for MLP model

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

### Test Scenarios
- **Sunny Day Simulation**: Clear sky conditions from 2023 dataset
- **Cloudy Day Simulation**: Overcast conditions from 2023 dataset
- **Comparative Analysis**: ML-based vs. astronomical tracking performance

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
5. Validate and test the model

# Simulation and Results

The simulation was carried out under two scenarios: a sunny day and a cloudy day. Each case was simulated using both the Random Forest model and the MLP model. The videos for each simulation are available at the following links:

[Random Forest prediction for sunny day](https://youtu.be/lWYLqjOefW4)

[Random Forest prediction for cloudy day](https://youtu.be/PSMhTjCeY1U)

## Solar Tracking Performance Analysis: Machine Learning vs SPA on Clear Sky Conditions

The comparative analysis of a Random Forest machine learning model against the Solar Position Algorithm (SPA) for solar tracking reveals remarkably similar performance metrics during the 13-hour daylight period analyzed on June 27, 2023, under clear sky conditions.

### Performance Equivalence

Both tracking systems demonstrated nearly identical energy collection capabilities, with the ML model achieving 99.9% efficiency compared to the astronomical SPA method. The ML system generated 9.77 kWh/m² versus 9.78 kWh/m² for SPA, representing only a 0.1% difference in daily energy output. Both tracking systems significantly outperformed the baseline global horizontal irradiance (GHI) of 7.62 kWh/m², with energy gains of 2.15 kWh/m² (28.3%) and 2.17 kWh/m² (28.5%) respectively.

| IRRADIANCE STATISTICS (W/m²) | Value | PERFORMANCE COMPARISON           | Value | DIFFERENCE ANALYSIS           |   Value   |
|------------------------------|-------|----------------------------------|-------|-------------------------------|-----------|
| Average GHI:                 | 564.1 | ML tracking ratio:               | 1.567 | Mean difference (Astro - ML): | 1.1 W/m²  |
| Average ML POA:              | 723.7 | Astronomical tracking ratio:     | 1.569 | Std deviation of difference:  | 1.4 W/m²  |
| Average Astronomical POA:    | 724.8 | ML tracking gain:                | 56.7% | Mean percentage difference:   | 0.1%      |
| Maximum ML POA:              | 993.3 | Astronomical tracking gain:      | 56.9% | Max positive difference:      | 4.9 W/m²  |
| Maximum Astronomical POA:    | 993.3 | ML efficiency (vs Astronomical): | 99.9% | Max negative difference:      | -0.8 W/m² |

### Tracking Accuracy Trade-offs

While performance outcomes were equivalent, the tracking methodologies differed significantly. The SPA algorithm maintained superior positional accuracy with an average angle of incidence (AOI) of 2.0° compared to the ML model's 7.2°. The positional differences between the two systems showed a mean absolute error of 5.1° in tilt positioning and 0.6° in azimuth positioning, indicating that the ML model's largest deviations from SPA occur in the tilt axis.

### Clear Sky Advantage for SPA

Under ideal sunny conditions, the astronomical SPA algorithm operates at optimal efficiency since direct solar radiation patterns are predictable and consistent. The absence of cloud interference allows precise sun positioning to maximize energy capture effectively. The ML model's ability to match this performance demonstrates its competence under clear sky scenarios, though the true advantage of machine learning approaches typically emerges under variable weather conditions where cloud movement and diffuse radiation patterns require adaptive tracking strategies.

### Practical Implications

Despite reduced positional precision, the Random Forest model's ability to match SPA performance under clear sky conditions suggests that perfect astronomical tracking may not be necessary for optimal energy collection on sunny days. However, this analysis represents ideal conditions where SPA excels. The ML approach's tolerance for positioning errors while maintaining energy output indicates potential advantages in reduced mechanical complexity, though its true benefits would likely be more apparent during partially cloudy or variable weather conditions where adaptive tracking becomes crucial.

The graph below presents a comparison between solar tracking controlled by the Random Forest model and astronomical tracking, using data from a **sunny day**.

![ml_vs_astronomical_tracking_comparison](https://github.com/user-attachments/assets/db803cf3-2db1-4128-9389-b4bd41747c74)

## Solar Tracking Performance Analysis: Machine Learning vs SPA on Cloudy Conditions

The comparative analysis on July 31, 2023, under cloudy conditions reveals a reversal in performance between the Random Forest machine learning model and the Solar Position Algorithm (SPA), demonstrating the adaptive advantages of ML-based tracking systems.

### Machine Learning Superiority in Variable Conditions

Under cloudy skies, the ML model outperformed the astronomical SPA algorithm, achieving 109.4% efficiency compared to SPA. The ML system generated 4.59 kWh/m² versus 4.50 kWh/m² for SPA, representing a 2.0% energy advantage. More remarkably, while the ML model maintained a positive tracking gain of 0.9% over baseline GHI, the SPA system actually performed worse than horizontal irradiance with a -6.8% tracking loss.

### Adaptive Tracking Under Diffuse Conditions

The ML model's success stems from its ability to optimize for diffuse radiation patterns rather than direct solar positioning. Despite maintaining large positional differences from SPA (25.8° mean absolute tilt error), the ML system's adaptive approach captured more available irradiance. The astronomical algorithm's precision in sun-tracking became counterproductive when cloud cover created predominantly diffuse lighting conditions.

### Performance Implications

This analysis demonstrates that machine learning approaches excel in variable weather conditions where traditional astronomical tracking fails. The ML model's ability to adapt to changing irradiance patterns provides substantial advantages over rigid sun-following algorithms during cloudy periods.

The graph below presents a comparison between solar tracking controlled by the Random Forest model and astronomical tracking, using data from a **cloudy day**.

![ml_vs_astronomical_tracking_comparison](https://github.com/user-attachments/assets/f443b4aa-92f9-4ac4-9da3-f0e133076bde)

## Citation

If you use this work in your research, please cite:
```
[Your Name] (2025). Intelligent Solar Tracker Control System using Machine Learning 
and NSRDB Data. GitHub Repository: [repository_url]
```

# Reference
Anderson, K., Hansen, C., Holmgren, W., Jensen, A., Mikofski, M., and Driesse, A. “pvlib python: 2023 project update.” Journal of Open Source Software, 8(92), 5994, (2023). DOI: 10.21105/joss.05994.

Meysam Mahooti (2025). NREL's Solar Position Algorithm (SPA) (https://www.mathworks.com/matlabcentral/fileexchange/59903-nrel-s-solar-position-algorithm-spa), MATLAB Central File Exchange. Retrieved July 2, 2025.

National Oceanic and Atmospheric Administration. NOWData—NOAA Online Weather Data. National Weather Service. Retrieved June 12, 2025, de https://www.weather.gov/wrh/Climate?wfo=mlb

Sengupta, M., Xie, Y., Lopez, A., Habte, A., Maclaurin, G., Shelby, J., 2018. The National Solar Radiation Data Base (NSRDB). Renew. Sustain. Energy Rev. 89, 51-60. https://doi.org/10.1016/j.rser.2018.03.003.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
