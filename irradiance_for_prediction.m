function [poa_global_ML, poa_global_AP] = irradiance_for_prediction(Datetime, Predicted_azimuth, Predicted_tilt, Astro_azimuth, Astro_tilt, DNI, DHI, GHI)

% Configuration - adjust these values for your location
LATITUDE = 29.184641;
LONGITUDE = -81.067368;
ALTITUDE = 8;

disp("ML Model vs Astronomical Solar Tracking Comparison")
disp("==================================================")

% Perform comparison analysis

Time = struct( ...
    'year', year(Datetime), ...
    'month', month(Datetime), ...
    'day', day(Datetime), ...
    'hour', hour(Datetime), ...
    'minute', minute(Datetime), ...
    'second', second(Datetime),...
    'UTCOffset', hours(tzoffset(Datetime)));

% Define location structure for solar position calculation
Location = struct('latitude', LATITUDE, 'longitude', LONGITUDE, 'altitude', ALTITUDE);

% Placeholder for pressure and temperature values
Pressure = 101900;  % Atmospheric pressure in Pa
Temperature = 27;     % Average temperature in degrees Celsius

% Calculate solar position
[SunAz, SunEl, ApparentSunEl] = pvl_spa(Time, Location, Pressure, Temperature);
ApparentSunZen = 90-ApparentSunEl;
SunZe = 90 - SunEl;

% Calculate AOI for ML model positions
aoi_ml = pvl_getaoi(Predicted_tilt, Predicted_azimuth, SunZe, SunAz);
% Calculate AOI for astronomical positions
aoi_astro = pvl_getaoi(Astro_tilt, Astro_azimuth, SunZe, SunAz);

% Calculate POA irradiance for ML model positions
HExtra = pvl_extraradiation(day(Datetime, "dayofyear"));
AM = pvl_relativeairmass(ApparentSunZen, 'simple');

[SkyDiffuse_ML,SkyDiffuse_Iso_ML,SkyDiffuse_Cir_ML,SkyDiffuse_Hor_ML] = pvl_perez(Predicted_tilt, ...
    Predicted_azimuth, DHI, DNI, HExtra, SunZe, SunAz, AM);

GR_ML = pvl_grounddiffuse(Predicted_tilt, GHI, 0.18);
I_beam_ML = DNI .* cos(aoi_ml .* (pi/180));
poa_global_ML = I_beam_ML + SkyDiffuse_ML + GR_ML;

% Calculate POA irradiance for astronomical positions
[SkyDiffuse_AP,SkyDiffuse_Iso_AP,SkyDiffuse_Cir_AP,SkyDiffuse_Hor_AP] = pvl_perez(Astro_tilt, ...
    Astro_azimuth, DHI, DNI, HExtra, SunZe, SunAz, AM);

GR_AP = pvl_grounddiffuse(Astro_tilt, GHI, 0.18);
I_beam_AP = DNI .* cos(aoi_astro .* (pi/180));
poa_global_AP = I_beam_AP + SkyDiffuse_AP + GR_AP;

poa_difference = poa_global_AP - poa_global_ML;
ml_tracking_ratio = poa_global_ML ./ GHI;
ap_tracking_ratio = poa_global_AP ./ GHI;
efficiency_ratio = poa_global_ML ./ poa_global_AP;

dt = datetime(Datetime, "Format","dd/MM/uuuu HH:mm:ss");

% Create comprehensive plots comparing ML vs astronomical tracking

figure(3);
sgtitle('ML Model vs Astronomical Solar Tracking Comparison', 'FontSize', 16)

subplot(3,2,1)
plot(dt, GHI, 'y', dt, poa_global_ML, 'c', dt, poa_global_AP, 'r');
xlabel('Time');
ylabel('Irradiance (W/m^2)');
legend('GHI', 'ML Model POA', 'Astronomical POA');
title('Irradiance Comparison');
grid on;

subplot(3,2,2)
plot(dt, Predicted_tilt, 'c', dt, Astro_tilt, 'r');
xlabel('Time');
ylabel('Tilt Angle (degrees)');
legend('ML tilt', 'Astronomical tilt');
title('Tilt Position Comparison');
grid on;

subplot(3,2,3)
plot(dt, Predicted_azimuth, 'c', dt, Astro_azimuth, 'r');
xlabel('Time');
ylabel('Azimuth Angle (degrees)');
legend('ML Azimuth', 'Astronomical Azimuth');
title('Azimuth Position Comparison');
grid on;

subplot(3,2,4)
plot(dt, aoi_ml, 'c', dt, aoi_astro, 'r');
xlabel('Time');
ylabel('Angle of Incidence (degrees)');
legend('ML AOI', 'Astronomical AOI');
title('AOI Comparison');
grid on;

subplot(3,2,5)
plot(dt, poa_difference, 'g');
xlabel('Time');
ylabel('Irradiance Difference (W/m²)');
title('Performance Difference (Astronomical - ML)');
grid on;

subplot(3,2,6)
yyaxis left;
plot(dt, ml_tracking_ratio, 'c', dt, ap_tracking_ratio, 'r');
ylabel('Tracking Ratio');
yyaxis right;
plot(dt, efficiency_ratio, 'g');
ylabel('ML/Astronomical Ratio');
title('Tracking Performance Ratios');
grid on;
xlabel('Time');
legend('ML Tracking Ratio', 'Astro Tracking Ratio', 'ML/Astro Efficiency');

% Create scatter plots for detailed analysis

% Filter daylight hours
daylight = ApparentSunEl > 0;

figure(4);
sgtitle('ML vs Astronomical Tracking - Detailed Analysis', 'FontSize', 16)

% Scatter plot 1: POA comparison
subplot(2,2,1)
scatter(poa_global_AP(daylight), poa_global_ML(daylight), "cyan", "filled")
max_val = max(max(poa_global_AP(daylight)), max(poa_global_ML(daylight)));
hold on
plot([0 max_val],[0 max_val], '--r')
xlabel('Astronomical POA (W/m²)');
ylabel('ML Model POA (W/m²)');
title('POA Irradiance Correlation');
grid on;

% Scatter plot 2: Position correlation - Tilt
subplot(2,2,2)
scatter(Astro_tilt(daylight), Predicted_tilt(daylight), "green", "filled");
max_tilt = max(max(Astro_tilt(daylight)), max(Predicted_tilt(daylight)));
hold on
plot([0 max_tilt],[0 max_tilt], '--r')
xlabel('Astronomical Tilt (degrees)');
ylabel('ML Model Tilt (degrees)');
title('Tilt Position Correlation');
grid on;

% Scatter plot 3: Position correlation - Azimuth
subplot(2,2,3)
scatter(Astro_azimuth(daylight), Predicted_azimuth(daylight), "filled", "MarkerFaceColor", "#F5AF29");
hold on
plot([0 360],[0 360], '--r')
xlabel('Astronomical Azimuth (degrees)');
ylabel('ML Model Azimuth (degrees)');
xlim("tight")
title('Azimuth Position Correlation');
grid on;

% Histogram of differences
subplot(2,2,4)
histogram(poa_difference(daylight),"NumBins", 30, "FaceColor", "magenta")
xlabel('POA Difference (W/m²)');
ylabel('Frequency');
title('Distribution of Performance Differences');
grid on;

% Print comprehensive comparison statistics

disp("============================================")
disp("ML MODEL vs ASTRONOMICAL TRACKING COMPARISON")
disp("============================================")

t = dt(daylight);
hours_lapse = hours(t(end) - t(1));

fprintf("\nDaylight hours analyzed: %s", string(hours_lapse));
fprintf("\nTotal period: %s to %s \n", string(dt(1)), string(dt(end)));

% Irradiance statistics
fprintf("\nIRRADIANCE STATISTICS (W/m²)\n")
disp("--------------------------------------")
fprintf("Average GHI: %.1f", mean(GHI(daylight)));
fprintf("\nAverage ML POA: %.1f", mean(poa_global_ML(daylight), "omitnan"));
fprintf("\nAverage Astronomical POA: %.1f", mean(poa_global_AP(daylight),"omitnan"));
fprintf("\nMaximum ML POA: %.1f", max(poa_global_ML(daylight)));
fprintf("\nMaximum Astronomical POA: %.1f \n", max(poa_global_AP(daylight)));

% Performance comparison
fprintf("\nPERFORMANCE COMPARISON\n")
disp("------------------------")
ml_ratio = mean(ml_tracking_ratio(daylight), "omitnan");
astro_ratio = mean(ap_tracking_ratio(daylight), "omitnan");

fprintf("ML tracking ratio: %.3f", ml_ratio);
fprintf("\nAstronomical tracking ratio: %.3f", astro_ratio);
fprintf("\nML tracking gain: %.1f%%", (ml_ratio-1)*100);
fprintf("\nAstronomical tracking gain: %.1f%%", (astro_ratio-1)*100);

% Efficiency analysis
avg_efficiency = mean(efficiency_ratio(daylight), "omitnan");
fprintf("\nML efficiency (vs Astronomical): %.3f (%.1f%%)\n", avg_efficiency, avg_efficiency*100)

% Differences analysis
poa_difference_percent = poa_difference ./ poa_global_ML;
mean_diff = mean(poa_difference(daylight));
std_diff = std(poa_difference(daylight));
mean_diff_percent = mean(poa_difference_percent(daylight), "omitnan");

fprintf("\nDIFFERENCE ANALYSIS\n")
disp("------------------------------")
fprintf("Mean difference (Astro - ML): %.1f W/m²", mean_diff);
fprintf("\nStd deviation of difference: %.1f W/m²", std_diff);
fprintf("\nMean percentage difference: %.1f%%", mean_diff_percent);
fprintf("\nMax positive difference: %.1f W/m²", max(poa_difference(daylight)));
fprintf("\nMax negative difference: %.1f W/m²\n", min(poa_difference(daylight)));

% AOI analysis
fprintf("\nANGLE OF INCIDENCE ANALYSIS\n")
disp("---------------------------------------")
fprintf("Average ML AOI: %.1f°", mean(aoi_ml(daylight)));
fprintf("\nAverage Astronomical AOI: %.1f°", mean(aoi_astro(daylight)));
fprintf("\nMinimum ML AOI: %.1f°", min(aoi_ml(daylight)));
fprintf("\nMinimum Astronomical AOI: %.1f°\n", min(aoi_astro(daylight)));

% Energy analysis
intervals_duration = hours_lapse / sum(int32(daylight));
ml_energy = (sum(poa_global_ML(daylight)) ./ 1000)* intervals_duration;  % kWh/m²
astro_energy = (sum(poa_global_AP(daylight)) ./ 1000)* intervals_duration;  % kWh/m²
ghi_energy = (sum(GHI(daylight)) ./ 1000) * intervals_duration;  % kWh/m²

fprintf("\nDAILY ENERGY ANALYSIS (kWh/m²)\n")
disp("---------------------------------------")
fprintf("GHI Energy: %.2f", ghi_energy)
fprintf("\nML Model Energy: %.2f", ml_energy)
fprintf("\nAstronomical Energy: %.2f", astro_energy)
fprintf("\nML Energy gain vs GHI: %.2f (%.1f%%)", ml_energy - ghi_energy, (ml_energy/ghi_energy-1)*100)
fprintf("\nAstronomical gain vs GHI: %.2f (%.1f%%)", astro_energy - ghi_energy, (astro_energy/ghi_energy-1)*100)
fprintf("\nEnergy difference (Astro-ML): %.2f (%.1f%%)\n", astro_energy - ml_energy, (astro_energy/ml_energy-1)*100)

% Position accuracy
fprintf("\nPOSITION ACCURACY\n")
disp("---------------------------------------")
tilt_mae = mean(abs(Predicted_tilt(daylight) - Astro_tilt(daylight)));
azimuth_mae = mean(abs(Predicted_azimuth(daylight) - Astro_azimuth(daylight)));

fprintf("Mean Absolute Tilt Error: %.1f°", tilt_mae)
fprintf("\nMean Absolute Azimuth Error: %.1f°", azimuth_mae)


end
