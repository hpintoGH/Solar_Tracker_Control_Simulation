%% Solar Position Calculation using NREL SPA Algorithm
% This script calculates the solar position (zenith and azimuth angles) 
% for a specific location over a user-defined time range with customizable 
% time intervals. It is based on the NREL Solar Position Algorithm (SPA) 
% implementation for MATLAB by Meysam Mahooti.

clear
clc

% Create date range
start_time = datetime(2023,1,1,0,0,0,'TimeZone','America/New_York');        % Start date
end_time   = datetime(2023,12,31,23,59,59,'TimeZone','America/New_York');        % Final date
step       = minutes(10);                    % Sampling rate
times = start_time:step:end_time;

% Inicializar arrays para guardar resultados
n = length(times);
zeniths = zeros(n,1);
azimuths = zeros(n,1);
datetimes = strings(n,1);
sunrises = zeros(n,1);
sunsets = zeros(n,1);

% Temperatures for the location (daily)
temperatures = importdata("Temperatures.csv"); % https://www.weather.gov/wrh/Climate?wfo=mlb or https://www.ncei.noaa.gov/access/us-climate-normals/

for i = 1:n
    t = times(i);   

    % Llenar estructura SPA
    spa_year          = year(t);
    spa_month         = month(t);
    spa_day           = day(t);
    spa_hour          = hour(t);
    spa_minute        = minute(t);
    spa_second        = second(t);
    spa_timezone      = hours(tzoffset(t));
    spa_delta_ut1     = 0;
    spa_delta_t       = 69.198;        % Available in https://maia.usno.navy.mil/products/deltaT
    spa_longitude     = -81.067368;   % Daytona Beach, FL, USA. Just for simulation purpose.
    spa_latitude      = 29.184641;
    spa_elevation     = 8;
    spa_pressure      = 1019;
    spa_temperature   = temperatures.data(day(t),month(t));
    spa_slope         = 0;
    spa_azm_rotation  = 0;
    spa_atmos_refract = 0.5667;
    spa_function      = 2;        % See spa_const.m

    % Calling the SPA function
    [result, spa] = spa_NREL(spa_year, spa_month, spa_day, spa_hour, spa_minute, spa_second, spa_timezone, ...
        spa_delta_ut1, spa_delta_t, spa_longitude, spa_latitude, spa_elevation, spa_pressure, ...
        spa_temperature, spa_slope, spa_azm_rotation, spa_atmos_refract, spa_function);

    if (result == 0)
        % Saving results
        zeniths(i) = spa.zenith;
        azimuths(i) = spa.azimuth;
        datetimes(i) = string(t);
        sunrises(i) = spa.sunrise;
        sunsets(i) = spa.sunset;
    else
        fprintf('SPA Error Code: %d\n', result);
    end
end

if (result == 0)
    % Crear tabla y exportar a CSV
    T = table(datetimes, zeniths, azimuths, sunrises, sunsets, ...
              'VariableNames', {'datetime', 'zenith_deg', 'azimuth_deg', 'sunrise', 'sunset'});
    
    writetable(T, 'spa_solar_position_dataset_2023.csv');
end

