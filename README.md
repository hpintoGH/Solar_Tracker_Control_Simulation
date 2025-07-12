# Solar Tracker Control System using Machine Learning 
## *Solution to MATLAB and Simulink Challenge project 249 Solar Tracker Control Simulation*

[MATLAB and Simulink Challenge Program](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)

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
- NSRDB (National Solar Radiation Database) - 6 years of data (2018-2023)
- Location: Daytona Beach, Florida
- Temporal Resolution: 10-minute intervals
- Geographic Scope: Location-specific training (models optimized for Florida climate conditions)

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

- Optimal Panel Position (azimuth and tilt angles)

The calculation method employs PVlib's functions using the Perez model for diffuse irradiance calculations. Input parameters include solar zenith angle, azimuth angle, DNI, DHI, GHI, extraterrestrial irradiance, and albedo values, which collectively determine the optimal panel orientation for any given atmospheric condition. The optimization objective focuses on achieving maximum total irradiance on the tilted panel surface, ensuring that the machine learning models learn to predict positions that maximize energy capture.

### Training Strategy
- Training Period: 2018-2021 (4 years)
- Validation Period: 2022 (1 year)
- Test Period: 2023 (1 year)
- Feature Selection: MRMR (Maximum Relevance Minimum Redundancy) for Random Forest
- Deep Learning Optimization: Iterative variable selection based on Random Forest results
- Training Platform: MATLAB Statistics and Machine Learning Toolbox for both Random Forest and MATLAB Deep Learning Toolbox for MLP model

## Machine Learning Models

### Random Forest Model 
- Target: Optimal panel position calculated via PVlib
- Training: MATLAB Statistics and Machine Learning Toolbox
- Objective: Predict panel angles that maximize total irradiance under given weather conditions

The Random Forest model employs separate optimized variable sets for azimuth and tilt angle predictions. For azimuth angle prediction, the final selected variables include solar azimuth angle, Global Horizontal Irradiance (GHI), month, and sinusoidal hour transformation, capturing both solar geometry and temporal patterns. The tilt angle prediction utilizes solar zenith angle, azimuth angle, Direct Normal Irradiance (DNI), Diffuse Horizontal Irradiance (DHI), and Global Horizontal Irradiance (GHI), incorporating comprehensive solar radiation components alongside positional information.

### Deep Learning Model (MLP)
- Target: Same optimal panel position from PVlib
- Architecture: 4 fully-connected layers
- Training: MATLAB Deep Learning Toolbox
- Enhanced prediction: Complex weather pattern recognition for irradiance maximization

The Multi-Layer Perceptron model features distinct variable configurations for each output angle. Azimuth angle prediction incorporates month, sinusoidal and cosine hour transformations, Global Horizontal Irradiance (GHI), solar azimuth angle, and zenith angle, providing comprehensive temporal and solar position information. For tilt angle prediction, the model utilizes day of year, sinusoidal hour transformation, Direct Normal Irradiance (DNI), Diffuse Horizontal Irradiance (DHI), Global Horizontal Irradiance (GHI), and solar zenith angle, emphasizing radiation components and annual temporal variations for optimal positioning decisions.

## Solar Position Algorithms

### Primary: NREL's Solar Position Algorithm (SPA)
- Implementation: Meysam Mahooti's MATLAB adaptation
- Reference: Meysam Mahooti (2025). NREL's Solar Position Algorithm (SPA), MATLAB Central File Exchange
- Custom Integration: Script for performing time-based calculations with user-defined intervals

### Alternative:
- Python PVlib Integration
- NOAA Solar Calculator

## Simulation Environment

### Simulink Model Components
- Dual-axis tracking system with azimuth and elevation control
- NEMA 17 stepper motors (1.8° per step resolution)
- Stepper motor control based on MathWorks ["Stepper Motor with Control"](https://www.mathworks.com/help/sps/ug/stepper-motor-with-control.html) example

### Test Scenarios
- Sunny Day Simulation: Clear sky conditions from 2023 dataset
- Cloudy Day Simulation: Overcast conditions from 2023 dataset

In both scenarios data is sampled every 30 minutes, where each time step corresponds to 1 second in the simulation.

## Research Applications

This project serves multiple research domains, providing a comprehensive framework for intelligent solar tracking through weather-predictive panel positioning that goes beyond traditional astronomical methods. The implementation offers valuable insights for machine learning applications in renewable energy, particularly through comparative analysis of Random Forest and Deep Learning models for solar optimization tasks. Additionally, the project serves as an educational resource for control systems engineering, featuring Simulink modeling techniques that integrate machine learning predictions with hardware simulation for dual-axis solar tracking mechanisms.

## Key Innovations

The system employs irradiance-optimized tracking where machine learning models predict panel positions that maximize total irradiance using PVlib, moving beyond sun-following algorithms to weather-driven optimization that accounts for atmospheric conditions affecting solar radiation. The comprehensive data integration approach utilizes a six-year NSRDB dataset with 10-minute temporal resolution, providing robust training data that captures seasonal variations and weather patterns specific to the Daytona Beach location. The dual algorithm methodology compares Random Forest and Deep Learning approaches for optimal positioning predictions, offering insights into model performance under varying atmospheric conditions. The physics-based target generation employs the Perez model for diffuse irradiance calculations, ensuring that the machine learning models learn from  optimal positions. Finally, the real-world validation framework uses actual weather scenarios from the 2023 dataset with irradiance-based performance metrics, demonstrating practical applicability of the predictive models under diverse meteorological conditions.

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
3. From MATLAB, navigate to the folder Machine_Learning_model to run the model training script ML_solar_tracker.m. Due to the size of the resulting file and GitHub storage limitations, it is not included in the repository.
4. Once you have trained the models, in Solar_tracker_simulation.mlx choose the test scenario (sunny/cloudy day) from the data available in the data_for_simulation folder
5. Select between Random Forest or MLP model
6. If you plan to run Python code from MATLAB, follow the instructions provided in the script. Otherwise, you will need to use an external IDE and set up the working environment in that IDE for PVlib.

To visualize simulations with additional data, use the CSV data tables located in the data_for_simulation folder. You can modify or select the desired time period, although the available data only covers the years 2023 and 2024. If you need more recent or real-time data, you must obtain it from a data provider.

**Note**: The trained models are specifically optimized for Daytona Beach, Florida conditions. For other locations, you'll need to retrain the models using the provided scripts with location-specific NSRDB data or other sources.

### Model training for a different location (optional)
1. Calculate the astronomical position of the sun for the geographic location and the training, validation, and test periods. You can use the adapted MATLAB version of the NREL SPA code found in the NREL_spa folder, or other tools such as the NOAA Solar Calculator or PVlib.
2. Obtain environmental variable data such as DNI, DHI, GHI, wind speed, and relative humidity — or additional variables if needed. You can use the NSRDB (National Solar Radiation Database) as a data source.
3. Calculate optimal panel positions using the PVlib code Irradiation_pvlib.py
4. Run MRMR feature selection for Random Forest model in MATLAB
5. Train Deep Learning model with iterative variable selection in MATLAB
6. Validate and test the model

# Simulation and Results

## Comparison Between the Random Forest Model and the MLP Deep Learning Model

The comparative analysis between Random Forest and MLP Deep Learning models for solar tracking prediction reveals Random Forest's superior performance across both azimuth and tilt predictions. Random Forest achieved exceptional accuracy with R² values of 0.99998 for azimuth and 0.99889 for tilt, compared to Deep Learning's 0.99715 and 0.95567 respectively.

Random Forest demonstrated significantly lower error rates, with MSE values of 0.076 (azimuth) and 0.730 (tilt) versus Deep Learning's 13.629 and 29.179. The Mean Absolute Errors were also substantially lower: 0.196° vs 3.074° for azimuth, and 0.409° vs 2.572° for tilt.

The relatively low azimuth errors in both models likely stem from the fact that optimal azimuth calculated by PVlib closely resembles the predictable astronomical solar azimuth. The greater variation occurs in tilt angle, which must adapt to momentary conditions such as cloud cover and atmospheric changes, making it the more challenging parameter to predict accurately.

The performance difference suggests that Random Forest's ensemble approach and feature handling capabilities are particularly well-suited for solar tracking applications, providing more precise and reliable predictions than the deep learning alternative. The graph below compares the predicted values from each model with the actual target values for both azimuth and tilt angles.

![Predictions vs actual values](https://github.com/user-attachments/assets/d2d3f80b-1716-4864-aa13-8382e67b4726)

## Simulink Simulation

The simulation was carried out under two scenarios: a sunny day and a cloudy day. Each case was simulated using both the Random Forest model and the MLP model. The videos for each simulation are available at the following links:

[Random Forest prediction for a sunny day](https://youtu.be/lWYLqjOefW4)

[Random Forest prediction for a cloudy day](https://youtu.be/PSMhTjCeY1U)

[MLP Deep Learning prediction a for sunny day](https://youtu.be/OSfu5jRcld8)

[MLP Deep Learning prediction a for cloudy day](https://youtu.be/UOo1wOEq56s)

### Solar Tracking Performance Analysis: Random Forest model vs SPA on Clear Sky Conditions

The comparative analysis of a Random Forest machine learning model against the Solar Position Algorithm (SPA) for solar tracking reveals remarkably similar performance metrics during the 13-hour daylight period analyzed on June 27, 2023, under clear sky conditions.

#### Performance Equivalence

Both tracking systems demonstrated nearly identical energy collection capabilities, with the ML model achieving 99.9% efficiency compared to the astronomical SPA method. The ML system generated 9.77 kWh/m² versus 9.78 kWh/m² for SPA, representing only a 0.1% difference in daily energy output. Both tracking systems significantly outperformed the baseline global horizontal irradiance (GHI) of 7.62 kWh/m², with energy gains of 2.15 kWh/m² (28.3%) and 2.17 kWh/m² (28.5%) respectively.

#### Tracking Accuracy Trade-offs

While performance outcomes were equivalent, the tracking methodologies differed significantly. The SPA algorithm maintained superior positional accuracy with an average angle of incidence (AOI) of 2.0° compared to the ML model's 7.2°. The positional differences between the two systems showed a mean absolute error of 5.1° in tilt positioning and 0.6° in azimuth positioning, indicating that the ML model's largest deviations from SPA occur in the tilt axis.

#### Clear Sky Advantage for SPA

Under ideal sunny conditions, the astronomical SPA algorithm operates at optimal efficiency since direct solar radiation patterns are predictable and consistent. The absence of cloud interference allows precise sun positioning to maximize energy capture effectively. The ML model's ability to match this performance demonstrates its competence under clear sky scenarios, though the true advantage of machine learning approaches typically emerges under variable weather conditions where cloud movement and diffuse radiation patterns require adaptive tracking strategies.

#### Practical Implications

Despite reduced positional precision, the Random Forest model's ability to match SPA performance under clear sky conditions suggests that perfect astronomical tracking may not be necessary for optimal energy collection on sunny days. However, this analysis represents ideal conditions where SPA excels. The ML approach's tolerance for positioning errors while maintaining energy output indicates potential advantages in reduced mechanical complexity, though its true benefits would likely be more apparent during partially cloudy or variable weather conditions where adaptive tracking becomes crucial.

The graph below presents a comparison between solar tracking controlled by the Random Forest model and astronomical tracking, using data from a **sunny day**.

<img width="4770" height="3543" alt="ml_vs_astronomical_tracking_comparison" src="https://github.com/user-attachments/assets/b3b7f300-51cd-4f9b-a1b4-15bc70370dd8" />

### Solar Tracking Performance Analysis: Random Forest model vs SPA on Cloudy Conditions

The comparative analysis on July 31, 2023, under cloudy conditions reveals a reversal in performance between the Random Forest machine learning model and the Solar Position Algorithm (SPA), demonstrating the adaptive advantages of ML-based tracking systems.

#### Machine Learning Superiority in Variable Conditions

Under cloudy skies, the ML model outperformed the astronomical SPA algorithm, achieving 109.4% efficiency compared to SPA. The ML system generated 4.59 kWh/m² versus 4.50 kWh/m² for SPA, representing a 2.0% energy advantage. More remarkably, while the ML model maintained a positive tracking gain of 0.9% over baseline GHI, the SPA system actually performed worse than horizontal irradiance with a -6.8% tracking loss.

#### Adaptive Tracking Under Diffuse Conditions

The ML model's success stems from its ability to optimize for diffuse radiation patterns rather than direct solar positioning. Despite maintaining large positional differences from SPA (25.8° mean absolute tilt error), the ML system's adaptive approach captured more available irradiance. The astronomical algorithm's precision in sun-tracking became counterproductive when cloud cover created predominantly diffuse lighting conditions.

#### Performance Implications

This analysis demonstrates that machine learning approaches excel in variable weather conditions where traditional astronomical tracking fails. The ML model's ability to adapt to changing irradiance patterns provides substantial advantages over rigid sun-following algorithms during cloudy periods.

The graph below presents a comparison between solar tracking controlled by the Random Forest model and astronomical tracking, using data from a **cloudy day**.

<img width="4770" height="3543" alt="ml_vs_astronomical_tracking_comparison" src="https://github.com/user-attachments/assets/d39fd723-f73c-4838-95b7-864e5ef0f31d" />

### Solar Tracking Performance Analysis: MLP Deep Learning model vs SPA on Clear Sky Conditions

<img width="4771" height="3543" alt="ml_vs_astronomical_tracking_comparison" src="https://github.com/user-attachments/assets/7cc11903-821b-471f-ba08-23dca8dcaf3b" />

### Solar Tracking Performance Analysis: MLP Deep Learning model vs SPA on Cloudy Conditions

<img width="4770" height="3543" alt="ml_vs_astronomical_tracking_comparison" src="https://github.com/user-attachments/assets/9abb9b38-68b7-4b81-9ce4-aa56d710274e" />

## Comparison of energy storage on a cloudy day

<img width="2928" height="1502" alt="image" src="https://github.com/user-attachments/assets/b9cb25d8-1d30-411a-8cf4-27bbd3ff3154" />

Based on the simulation results for cloudy conditions, using machine learning for solar tracking yields a nearly identical battery SOC compared to the traditional SPA method, with only a 0.94% difference. This suggests little advantage in terms of stored energy under the simulated scenario. While this improvement may seem modest, it could become significant over long-term operation, in locations with frequent cloud cover, or when bifacial panels are used.

It is important to note that these findings are based on simulations, not real-world measurements. Factors such as sensor noise, shading variability, and system dynamics may affect actual performance. Therefore, while machine learning shows promising results and generalization potential, its practical benefits should be validated through field experiments. Overall, it can be considered a viable and potentially superior alternative to SPA under specific environmental conditions.

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
