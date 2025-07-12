import pandas as pd
import pvlib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compare_ml_vs_astronomical_tracking(csv_file_path, latitude, longitude, timezone):
    """
    Compare irradiance between ML model positions and astronomical positions
    
    Parameters:
    -----------
    csv_file_path : str
        Path to CSV file containing both ML and astronomical tracker positions with irradiance data
    latitude : float
        Site latitude in degrees
    longitude : float
        Site longitude in degrees  
    timezone : str
        Timezone string
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated irradiance values for both tracking methods
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Expected column names (adjust these based on your CSV structure)
    expected_columns = {
        'datetime': ['Datetime', 'timestamp', 'time', 'datetime'],
        # ML model positions
        'ml_tilt': ['ml_tilt', 'Predicted_tilt'],
        'ml_azimuth': ['ml_azimuth', 'Predicted_azimuth'],
        # Astronomical positions
        'astro_tilt': ['Astro_tilt', 'sun_tilt'],
        'astro_azimuth': ['Astro_azimuth', 'sun_azimuth'],
        # Irradiance data
        'dni': ['dni', 'DNI'],
        'dhi': ['dhi', 'DHI'], 
        'ghi': ['ghi', 'GHI']
    }
    
    # Find actual column names
    column_mapping = {}
    for key, possible_names in expected_columns.items():
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                column_mapping[key] = col
                break
    
    print(f"Column mapping: {column_mapping}")
    
    # Check if all required columns are found
    required_cols = ['datetime', 'astro_tilt', 'astro_azimuth', 'dni', 'dhi', 'ghi']
    missing_cols = [col for col in required_cols if col not in column_mapping]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print("Please ensure your CSV has columns for: datetime, astro_tilt, astro_azimuth, DNI, DHI, GHI")
        return None
    
    # Rename columns for consistency
    df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Convert datetime column
    try:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['datetime'], utc=False)
        df_renamed.set_index('datetime', inplace=True)
    except Exception as e:
        print(f"Error converting datetime: {e}")
        return None
    
    # Create location object
    location = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=timezone, altitude=8)
    
    # Calculate solar position
    solar_position = location.get_solarposition(df_renamed.index, pressure=101900, temperature=27)
        
    # Calculate AOI for ML model positions
    aoi_ml = pvlib.irradiance.aoi(
        surface_tilt=df_renamed['ml_tilt'],
        surface_azimuth=df_renamed['ml_azimuth'], 
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    
    # Calculate AOI for astronomical positions
    aoi_astro = pvlib.irradiance.aoi(
        surface_tilt=df_renamed['astro_tilt'],
        surface_azimuth=df_renamed['astro_azimuth'], 
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    
    # Calculate POA irradiance for ML model positions
    poa_ml = pvlib.irradiance.get_total_irradiance(
        surface_tilt=df_renamed['ml_tilt'],
        surface_azimuth=df_renamed['ml_azimuth'],
        dni=df_renamed['dni'],
        ghi=df_renamed['ghi'], 
        dhi=df_renamed['dhi'],
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],        
        albedo=0.18,
        model='isotropic'
    )
    
    # Calculate POA irradiance for astronomical positions
    poa_astro = pvlib.irradiance.get_total_irradiance(
        surface_tilt=df_renamed['astro_tilt'],
        surface_azimuth=df_renamed['astro_azimuth'],
        dni=df_renamed['dni'],
        ghi=df_renamed['ghi'], 
        dhi=df_renamed['dhi'],
        solar_zenith=solar_position['zenith'],
        solar_azimuth=solar_position['azimuth'],        
        albedo=0.18,
        model='isotropic'
    )
    
    # Create results dataframe
    results = pd.DataFrame(index=df_renamed.index)

    # Add original data
    results['dni'] = df_renamed['dni']
    results['dhi'] = df_renamed['dhi']
    results['ghi'] = df_renamed['ghi']
    
    # Add solar position
    results['solar_zenith'] = solar_position['zenith']
    results['solar_azimuth'] = solar_position['azimuth']
    results['solar_elevation'] = solar_position['elevation']
    
    # Add ML model data
    results['ml_tilt'] = df_renamed['ml_tilt']
    results['ml_azimuth'] = df_renamed['ml_azimuth']
    results['ml_aoi'] = aoi_ml
    results['ml_poa_global'] = poa_ml['poa_global']
    results['ml_poa_direct'] = poa_ml['poa_direct']
    results['ml_poa_diffuse'] = poa_ml['poa_diffuse']
    results['ml_tracking_ratio'] = results['ml_poa_global'] / results['ghi']
    
    # Add astronomical data
    results['astro_tilt'] = df_renamed['astro_tilt']
    results['astro_azimuth'] = df_renamed['astro_azimuth']
    results['astro_aoi'] = aoi_astro
    results['astro_poa_global'] = poa_astro['poa_global']
    results['astro_poa_direct'] = poa_astro['poa_direct']
    results['astro_poa_diffuse'] = poa_astro['poa_diffuse']
    results['astro_tracking_ratio'] = results['astro_poa_global'] / results['ghi']
    
    # Calculate comparison metrics
    results['poa_difference'] = results['astro_poa_global'] - results['ml_poa_global']
    results['poa_difference_percent'] = (results['poa_difference'] / results['ml_poa_global']) * 100
    results['efficiency_ratio'] = results['ml_poa_global'] / results['astro_poa_global']
    
    # Replace infinite values with NaN
    results = results.replace([np.inf, -np.inf], np.nan)
    
    return results

def plot_comparison_results(results, save_plots=True):
    """
    Create comprehensive plots comparing ML vs astronomical tracking
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('ML Model vs Astronomical Solar Tracking Comparison', fontsize=16, color='white')
    fig.set_facecolor('black')

    for ax in axes.flat:
        ax.set_facecolor('#171717')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')        
    
    # No time zone
    time_wo_tz = [dt.replace(tzinfo=None) for dt in results.index]
    
    # Plot 1: Irradiance comparison
    axes[0,0].plot(time_wo_tz, results['ghi'], label='GHI', alpha=0.7, color='gold')
    axes[0,0].plot(time_wo_tz, results['ml_poa_global'], label='ML Model POA', alpha=0.8, color='deepskyblue')
    axes[0,0].plot(time_wo_tz, results['astro_poa_global'], label='Astronomical POA', alpha=0.8, color='red')
    axes[0,0].set_ylabel('Irradiance (W/m²)', color='white')
    axes[0,0].set_title('Irradiance Comparison', color='white')
    axes[0,0].legend(facecolor='dimgrey', labelcolor='white')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0,0].autoscale(enable=True, axis='x', tight=True)
        
    # Plot 2: Position comparison - Tilt
    axes[0,1].plot(time_wo_tz, results['ml_tilt'], label='ML Tilt', alpha=0.8, color='deepskyblue')
    axes[0,1].plot(time_wo_tz, results['astro_tilt'], label='Astronomical Tilt', alpha=0.8, color='red')
    axes[0,1].set_ylabel('Tilt Angle (degrees)', color='white')
    axes[0,1].set_title('Tilt Position Comparison', color='white')
    axes[0,1].legend(facecolor='dimgrey', labelcolor='white')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0,1].autoscale(enable=True, axis='x', tight=True)
        
    # Plot 3: Position comparison - Azimuth
    axes[1,0].plot(time_wo_tz, results['ml_azimuth'], label='ML Azimuth', alpha=0.8, color='deepskyblue')
    axes[1,0].plot(time_wo_tz, results['astro_azimuth'], label='Astronomical Azimuth', alpha=0.8, color='red')
    axes[1,0].set_ylabel('Azimuth Angle (degrees)', color='white')
    axes[1,0].set_title('Azimuth Position Comparison', color='white')
    axes[1,0].legend(facecolor='dimgrey', labelcolor='white')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1,0].autoscale(enable=True, axis='x', tight=True)
        
    # Plot 4: AOI comparison
    axes[1,1].plot(time_wo_tz, results['ml_aoi'], label='ML AOI', alpha=0.8, color='deepskyblue')
    axes[1,1].plot(time_wo_tz, results['astro_aoi'], label='Astronomical AOI', alpha=0.8, color='red')
    axes[1,1].set_ylabel('Angle of Incidence (degrees)', color='white')
    axes[1,1].set_title('AOI Comparison', color='white')
    axes[1,1].legend(facecolor='dimgrey', labelcolor='white')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1,1].autoscale(enable=True, axis='x', tight=True)
       
    # Plot 5: Performance difference
    axes[2,0].plot(time_wo_tz, results['poa_difference'], color='lime', alpha=0.8)
    axes[2,0].axhline(y=0, color='royalblue', linestyle='--', alpha=0.5)
    axes[2,0].set_ylabel('Irradiance Difference (W/m²)', color='white')
    axes[2,0].set_title('Performance Difference (Astronomical - ML)', color='white')
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2,0].autoscale(enable=True, axis='x', tight=True)
        
    # Plot 6: Efficiency ratio and tracking ratios
    ax1 = axes[2,1]
    ax2 = ax1.twinx()
    axes[2,1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    line1 = ax1.plot(time_wo_tz, results['ml_tracking_ratio'], color='deepskyblue', label='ML Tracking Ratio', alpha=0.7)
    line2 = ax1.plot(time_wo_tz, results['astro_tracking_ratio'], 'r-', label='Astro Tracking Ratio', alpha=0.7)
    line3 = ax2.plot(time_wo_tz, results['efficiency_ratio'], color='lime', label='ML/Astro Efficiency', alpha=0.7)
    
    ax1.set_ylabel('Tracking Ratio', color='white')
    ax2.set_ylabel('ML/Astronomical Ratio', color='lime')
    ax1.set_title('Tracking Performance Ratios', color='white')
    ax2.tick_params(colors='white')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', facecolor='dimgrey', labelcolor='white')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('ml_vs_astronomical_tracking_comparison.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'ml_vs_astronomical_tracking_comparison.png'")
    
    plt.show()

def plot_scatter_analysis(results, save_plots=True):
    """
    Create scatter plots for detailed analysis
    """
    # Filter daylight hours
    daylight = results[results['solar_elevation'] > 0].copy()
    
    if len(daylight) == 0:
        print("No daylight hours found for scatter analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ML vs Astronomical Tracking - Detailed Analysis', fontsize=16)
    
    # Scatter plot 1: POA comparison
    axes[0,0].scatter(daylight['astro_poa_global'], daylight['ml_poa_global'], 
                     alpha=0.6, color='blue', s=20)
    # Perfect correlation line
    max_val = max(daylight['astro_poa_global'].max(), daylight['ml_poa_global'].max())
    axes[0,0].plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Perfect Correlation')
    axes[0,0].set_xlabel('Astronomical POA (W/m²)')
    axes[0,0].set_ylabel('ML Model POA (W/m²)')
    axes[0,0].set_title('POA Irradiance Correlation')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Scatter plot 2: Position correlation - Tilt
    axes[0,1].scatter(daylight['astro_tilt'], daylight['ml_tilt'], 
                     alpha=0.6, color='green', s=20)
    max_tilt = max(daylight['astro_tilt'].max(), daylight['ml_tilt'].max())
    axes[0,1].plot([0, max_tilt], [0, max_tilt], 'r--', alpha=0.8, label='Perfect Correlation')
    axes[0,1].set_xlabel('Astronomical Tilt (degrees)')
    axes[0,1].set_ylabel('ML Model Tilt (degrees)')
    axes[0,1].set_title('Tilt Position Correlation')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Scatter plot 3: Position correlation - Azimuth
    axes[1,0].scatter(daylight['astro_azimuth'], daylight['ml_azimuth'], 
                     alpha=0.6, color='orange', s=20)
    axes[1,0].plot([0, 360], [0, 360], 'r--', alpha=0.8, label='Perfect Correlation')
    axes[1,0].set_xlabel('Astronomical Azimuth (degrees)')
    axes[1,0].set_ylabel('ML Model Azimuth (degrees)')
    axes[1,0].set_title('Azimuth Position Correlation')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Histogram of differences
    axes[1,1].hist(daylight['poa_difference'].dropna(), bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Zero Difference')
    axes[1,1].set_xlabel('POA Difference (W/m²)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Performance Differences')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('ml_vs_astronomical_scatter_analysis.png', dpi=300, bbox_inches='tight')
        print("Scatter analysis saved as 'ml_vs_astronomical_scatter_analysis.png'")
    
    plt.show()

def print_comparison_statistics(results):
    """
    Print comprehensive comparison statistics
    """
    print("\n" + "="*80)
    print("ML MODEL vs ASTRONOMICAL TRACKING COMPARISON")
    print("="*80)
    
    # Filter daylight hours (solar elevation > 0)
    daylight = results[results['solar_elevation'] > 0].copy()
    
    if len(daylight) == 0:
        print("No daylight hours found in the data")
        return
    
    hours_lapse = (daylight.index[-1] - daylight.index[0]).total_seconds()/3600
    
    print(f"\nDaylight hours analyzed: {hours_lapse}")
    print(f"Total period: {results.index[0]} to {results.index[-1]}")
    
    # Irradiance statistics
    print(f"\n{'IRRADIANCE STATISTICS (W/m²)':<50}")
    print("-" * 50)
    print(f"{'Average GHI:':<30} {daylight['ghi'].mean():.1f}")
    print(f"{'Average ML POA:':<30} {daylight['ml_poa_global'].mean():.1f}")
    print(f"{'Average Astronomical POA:':<30} {daylight['astro_poa_global'].mean():.1f}")
    print(f"{'Maximum ML POA:':<30} {daylight['ml_poa_global'].max():.1f}")
    print(f"{'Maximum Astronomical POA:':<30} {daylight['astro_poa_global'].max():.1f}")
    
    # Performance comparison
    print(f"\n{'PERFORMANCE COMPARISON':<50}")
    print("-" * 50)
    ml_ratio = daylight['ml_tracking_ratio'].mean()
    astro_ratio = daylight['astro_tracking_ratio'].mean()
    
    print(f"{'ML tracking ratio:':<30} {ml_ratio:.3f}")
    print(f"{'Astronomical tracking ratio:':<30} {astro_ratio:.3f}")
    print(f"{'ML tracking gain:':<30} {(ml_ratio-1)*100:.1f}%")
    print(f"{'Astronomical tracking gain:':<30} {(astro_ratio-1)*100:.1f}%")
    
    # Efficiency analysis
    avg_efficiency = daylight['efficiency_ratio'].mean()
    print(f"{'ML efficiency (vs Astronomical):':<30} {avg_efficiency:.3f} ({avg_efficiency*100:.1f}%)")
    
    # Differences analysis
    mean_diff = daylight['poa_difference'].mean()
    std_diff = daylight['poa_difference'].std()
    mean_diff_percent = daylight['poa_difference_percent'].mean()
    
    print(f"\n{'DIFFERENCE ANALYSIS':<50}")
    print("-" * 50)
    print(f"{'Mean difference (Astro - ML):':<30} {mean_diff:.1f} W/m²")
    print(f"{'Std deviation of difference:':<30} {std_diff:.1f} W/m²")
    print(f"{'Mean percentage difference:':<30} {mean_diff_percent:.1f}%")
    print(f"{'Max positive difference:':<30} {daylight['poa_difference'].max():.1f} W/m²")
    print(f"{'Max negative difference:':<30} {daylight['poa_difference'].min():.1f} W/m²")
    
    # AOI analysis
    print(f"\n{'ANGLE OF INCIDENCE ANALYSIS':<50}")
    print("-" * 50)
    print(f"{'Average ML AOI:':<30} {daylight['ml_aoi'].mean():.1f}°")
    print(f"{'Average Astronomical AOI:':<30} {daylight['astro_aoi'].mean():.1f}°")
    print(f"{'Minimum ML AOI:':<30} {daylight['ml_aoi'].min():.1f}°")
    print(f"{'Minimum Astronomical AOI:':<30} {daylight['astro_aoi'].min():.1f}°")
    
    # Energy analysis
    intervals_duration = hours_lapse / (len(daylight)-1)
    ml_energy = daylight['ml_poa_global'].sum() / 1000 * intervals_duration  # kWh/m² (30min intervals)
    astro_energy = daylight['astro_poa_global'].sum() / 1000 * intervals_duration  # kWh/m²
    ghi_energy = daylight['ghi'].sum() / 1000 * intervals_duration  # kWh/m²   

    print(f"\n{'DAILY ENERGY ANALYSIS (kWh/m²)':<50}")
    print("-" * 50)
    print(f"{'GHI Energy:':<30} {ghi_energy:.2f}")
    print(f"{'ML Model Energy:':<30} {ml_energy:.2f}")
    print(f"{'Astronomical Energy:':<30} {astro_energy:.2f}")
    print(f"{'ML Energy gain vs GHI:':<30} {ml_energy - ghi_energy:.2f} ({((ml_energy/ghi_energy-1)*100):.1f}%)")
    print(f"{'Astronomical gain vs GHI:':<30} {astro_energy - ghi_energy:.2f} ({((astro_energy/ghi_energy-1)*100):.1f}%)")
    print(f"{'Energy difference (Astro-ML):':<30} {astro_energy - ml_energy:.2f} ({((astro_energy/ml_energy-1)*100):.1f}%)")
    
    # Position accuracy
    print(f"\n{'POSITION ACCURACY':<50}")
    print("-" * 50)
    tilt_mae = np.abs(daylight['ml_tilt'] - daylight['astro_tilt']).mean()
    azimuth_mae = np.abs(daylight['ml_azimuth'] - daylight['astro_azimuth']).mean()
    
    print(f"{'Mean Absolute Tilt Error:':<30} {tilt_mae:.1f}°")
    print(f"{'Mean Absolute Azimuth Error:':<30} {azimuth_mae:.1f}°")

# Usage
if __name__ == "__main__":
    # Configuration - adjust these values for your location
    CSV_FILE = "prediction_results.csv"  # Path to your CSV file
    LATITUDE = 29.184641
    LONGITUDE = -81.067368
    TIMEZONE = 'America/New_York'  # Timezone
    
    print("ML Model vs Astronomical Solar Tracking Comparison")
    print("="*60)
    
    # Perform comparison analysis
    results = compare_ml_vs_astronomical_tracking(
        csv_file_path=CSV_FILE,
        latitude=LATITUDE, 
        longitude=LONGITUDE,
        timezone=TIMEZONE
    )
    
    if results is not None:
        # Print comprehensive statistics
        print_comparison_statistics(results)
        
        # Create comparison plots
        plot_comparison_results(results, save_plots=True)
        
        # Create scatter analysis plots
        plot_scatter_analysis(results, save_plots=True)
        
        # Save results to CSV
        output_file = "ml_vs_astronomical_comparison_results.csv"
        results.to_csv(output_file)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Display first few rows
        print(f"\nFirst 5 rows of comparison results:")
        print(results[['ml_poa_global', 'astro_poa_global', 'poa_difference', 'efficiency_ratio']].head())
        
    else:
        print("Analysis failed. Please check your CSV file format and column names.")

# CSV file format:
"""
Expected CSV format:

datetime,ml_tilt,ml_azimuth,astro_tilt,astro_azimuth,dni,dhi,ghi
2024-01-01 00:00:00,0,180,0,180,0,0,0
2024-01-01 00:30:00,0,180,0,180,0,0,0
2024-01-01 07:00:00,18,125,15,120,450,80,250
2024-01-01 07:30:00,22,130,20,125,520,85,300
2024-01-01 08:00:00,25,135,24,132,580,90,350
...

Where:
- datetime: timestamp in local timezone
- ml_tilt/ml_azimuth: ML model predicted tracker positions (degrees)
- astro_tilt/astro_azimuth: astronomical optimal tracker positions (degrees)
- dni/dhi/ghi: irradiance measurements (W/m²)

Column names are flexible - the script will try to detect variations like:
- ML positions: 'predicted_tilt', 'model_azimuth', etc.
- Astronomical: 'optimal_tilt', 'sun_azimuth', etc.
"""