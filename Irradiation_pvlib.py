import pandas as pd
import numpy as np
from pvlib import irradiance
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Read data
print("Loading data...")
df = pd.read_csv("solar_position_data.csv", parse_dates=['datetime']) # File with solar positions
df = df.set_index("datetime")

# Filter only valid daytime data (zenith < 90 and GHI > 0)
df = df[(df['zenith_deg'] < 90) & (df['GHI'] > 0)]
print(f"Valid daytime data: {len(df)} records")

# Extraterrestrial DNI calculation for the Perez model
print("Calculating extraterrestrial DNI...")
dates = pd.to_datetime(df.index)
day_of_year = dates.strftime('%j').astype(int)
years = dates.year.unique()
print(f"Years in the data: {years.min()} - {years.max()}")
df['dni_extra'] = irradiance.get_extra_radiation(day_of_year, epoch_year=years)
print("Extraterrestrial DNI calculated")

# Define search range for tilt and azimuth
tilt_values = np.arange(0, 91, 1)           # from 0° to 90° every 1°
azimuth_values = np.arange(0, 361, 1)      # from 0° to 360° every 1°

print(f"Evaluating {len(tilt_values)} tilt angles and {len(azimuth_values)} azimuth angles")
print(f"Total combinations per record: {len(tilt_values) * len(azimuth_values)}")

# Vectorize the calculation for better efficiency
def find_optimal_angles_vectorized(row):
    """Find optimal angles for a specific row using vectorization"""
    
    # Create a meshgrid for all combinations
    tilt_grid, azimuth_grid = np.meshgrid(tilt_values, azimuth_values, indexing='ij')
    tilt_flat = tilt_grid.flatten()
    azimuth_flat = azimuth_grid.flatten()
    
    # Compute irradiance for all combinations at once
    try:
        irr_results = irradiance.get_total_irradiance(
            surface_tilt=tilt_flat,
            surface_azimuth=azimuth_flat,
            solar_zenith=row["zenith_deg"],
            solar_azimuth=row["azimuth_deg"],
            dni=row["DNI"],
            ghi=row["GHI"],
            dhi=row["DHI"],
            dni_extra=row["dni_extra"],  # Extraterrestrial DNI required for Perez model
            albedo=row["Surface_Albedo"],
            model='perez'
        )
        
        poa_values = irr_results["poa_global"]
        
        # Find the index of the maximum value
        max_idx = np.argmax(poa_values)
        
        return {
            "best_tilt": tilt_flat[max_idx],
            "best_azimuth": azimuth_flat[max_idx],
            "poa": poa_values[max_idx]            
        }
    except Exception as e:
        print(f"Error in calculation: {e}")
        return {
            "best_tilt": np.nan,
            "best_azimuth": np.nan,
            "poa": np.nan            
        }

# Process data with a progress bar
print("Processing data...")
start_time = time.time()

results = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Optimizing angles"):
    result = find_optimal_angles_vectorized(row)
    result["datetime"] = index
    results.append(result)

# Create a DataFrame with the results
results_df = pd.DataFrame(results).set_index("datetime")

# Save the results
results_df.to_csv("optimized_tilt_azimuth_results.csv")
print(f"\nResults saved in 'optimized_tilt_azimuth_results.csv'")

print(f"DNI_extra range: {df['dni_extra'].min():.0f} - {df['dni_extra'].max():.0f} W/m²")
print(f"DNI_extra average: {df['dni_extra'].mean():.0f} W/m²")