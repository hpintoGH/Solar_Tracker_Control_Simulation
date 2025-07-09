%% LOAD AND PROCESS DATA

% Load data
files = 2018:2023;
allData = cell(numel(files),1);

for k = 1:numel(files)
    allData{k} = readtable("spa_solar_position_dataset_" + string(files(k)) + ".csv"); % Files with spa and enviromental data
end

data = vertcat(allData{:});

% Correct date range
start_time = datetime(2018,1,1,0,0,0,'TimeZone','America/New_York');        % Start date
end_time   = datetime(2023,12,31,23,59,59,'TimeZone','America/New_York');        % Final date
step       = minutes(10);                    % Sampling rate
times = start_time:step:end_time;
data.datetime = transpose(times);

% Filter only useful data (Sun is above the horizon)
idx_filter = data.zenith_deg <= 90 & data.GHI > 0;
data_util = data(idx_filter, :);
clear data allData

%% Create input features
% MRMR and MLP
hour_decimal = hour(data_util.datetime) + minute(data_util.datetime)/60;
day_of_year = day(data_util.datetime, 'dayofyear');
month_of_year = month(data_util.datetime);

% Trigonometric variables
hour_sin = sin(2*pi*hour_decimal/24);
hour_cos = cos(2*pi*hour_decimal/24);

%% Target variables (Y) - optimal tracker positions with PVlib
allData = cell(numel(files),1);

for k = 1:numel(files)    
    file_name = "optimized_tilt_azimuth_results_" + string(files(k)) + ".csv"; % Files with best position for irradiance from PVlib
    allData{k} = readtable(fullfile("PVlib_irradiance", file_name));
end
best_position = vertcat(allData{:});
clear allData

% Azimuth adjustment for zero values (position that PVlib returns when there is not enough sunlight)
idx_zero = best_position.best_azimuth == 0;
best_position.best_azimuth(idx_zero) = round(data_util.azimuth_deg(idx_zero)); % Solar azimuth

best_position.datetime.TimeZone = times.TimeZone;

%% Input variables (X)

X = [month_of_year, day_of_year, hour_sin, hour_cos, data_util.DNI, ...
    data_util.DHI, data_util.GHI, data_util.azimuth_deg, data_util.zenith_deg, ...
    data_util.Wind_Speed, data_util.Relative_Humidity];

% Variable names for X
feature_names = { 'month', 'day_of_year', 'hour_sin', 'hour_cos', ...
    'DNI', 'DHI', 'GHI', 'azimuth_deg', 'zenith_deg', 'Wind_Speed', ...
    'Relative_Humidity' };

% Index of variables that will be normalized for Deep Learning (modify manually)
idx_norm = [1:2 5:size(X,2)];

%% TEMPORAL DIVISION OF DATA
train_idx = year(data_util.datetime) >= 2018 & year(data_util.datetime) <= 2021;  % 4 years of training
val_idx = year(data_util.datetime) == 2022;                   % 1 year of validation
test_idx = year(data_util.datetime) == 2023;                  % 1 year of testing

X_train = X(train_idx, :);
Y_train_az = best_position.best_azimuth(train_idx);
Y_train_ti = best_position.best_tilt(train_idx);

X_val = X(val_idx, :);
Y_val_az = best_position.best_azimuth(val_idx);
Y_val_ti = best_position.best_tilt(val_idx);

X_test = X(test_idx, :);
Y_test_az = best_position.best_azimuth(test_idx);
Y_test_ti = best_position.best_tilt(test_idx);
