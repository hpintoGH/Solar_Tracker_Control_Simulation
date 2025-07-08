function results = MLP_dl_model(X_train, Y_train_az, Y_train_ti, ...
                               X_val, Y_val_az, Y_val_ti, ...
                               X_test, Y_test_az, Y_test_ti, idx_norm, feature_names)
%% FUNCTION: MLP_dl_model
% Train deep neural networks for azimuth and tilt
%
% Inputs:
%   X_train, Y_train_az, Y_train_ti: Training data
%   X_val, Y_val_az, Y_val_ti: Validation data
%   X_test, Y_test_az, Y_test_ti: Test data
%
% Output:
%   results: Structure with results and metrics
%
% Variable names for this project
% feature_names = { 'month', 'day_of_year', 'hour_sin', 'hour_cos', ...
%     'DNI', 'DHI', 'GHI', 'azimuth_deg', 'zenith_deg', 'Wind_Speed', ...
%     'Relative_Humidity' };

fprintf('Starting Deep Learning training...\n');

%% DATA NORMALIZATION
% Normalize based on statistics from the training set
X_train_norm = X_train;
train_mu = zeros(1, size(X_train_norm,2));
train_sigma = ones(1, size(X_train_norm,2));
[X_train_norm(:,idx_norm), train_mu(idx_norm), train_sigma(idx_norm)] = normalize(X_train(:,idx_norm));

X_val_norm = normalize(X_val, 'center', train_mu, 'scale', train_sigma);
X_test_norm = normalize(X_test, 'center', train_mu, 'scale', train_sigma);

%% CONFIGURATION OF THE NEURAL NETWORK
% Network architecture
% Adjust manually by doing several simulations (Consider the MRMR analysis
% in random forest and work from there)
num_features_az = 6;
final_features_az = [1 3 4 7 8 9];
num_features_ti = 6;
final_features_ti = [2 3 5 6 7 9];

hidden_units = [128, 64, 32, 16]; % 4 hidden layers

% Training setup
dl_params = struct();
dl_params.MaxEpochs = 200;
dl_params.MiniBatchSize = 64;
dl_params.InitialLearnRate = 0.001;
dl_params.ValidationFrequency = 50;
dl_params.ValidationPatience = 20;
dl_params.Shuffle = 'every-epoch';
dl_params.Verbose = true;
dl_params.Plots = 'none'; % Switch to 'training-progress' if you want to see graphs

%% CREATE NETWORK ARCHITECTURE
layers_az = [
    featureInputLayer(num_features_az, 'Name', 'input')
    fullyConnectedLayer(hidden_units(1), 'Name', 'fc1')
    batchNormalizationLayer
    reluLayer('Name', 'relu1')
    dropoutLayer(0.2, 'Name', 'dropout1')
    fullyConnectedLayer(hidden_units(2), 'Name', 'fc2')
    batchNormalizationLayer
    reluLayer('Name', 'relu2')
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(hidden_units(3), 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.2, 'Name', 'dropout3')
    fullyConnectedLayer(hidden_units(4), 'Name', 'fc4')
    reluLayer('Name', 'relu4')    
    fullyConnectedLayer(1, 'Name', 'output')    
];
layers_ti = [
    featureInputLayer(num_features_ti, 'Name', 'input')
    fullyConnectedLayer(hidden_units(1), 'Name', 'fc1')
    batchNormalizationLayer
    reluLayer('Name', 'relu1')
    dropoutLayer(0.2, 'Name', 'dropout1')
    fullyConnectedLayer(hidden_units(2), 'Name', 'fc2')
    batchNormalizationLayer
    reluLayer('Name', 'relu2')
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(hidden_units(3), 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.2, 'Name', 'dropout3')
    fullyConnectedLayer(hidden_units(4), 'Name', 'fc4')
    reluLayer('Name', 'relu4')    
    fullyConnectedLayer(1, 'Name', 'output')    
];
%% SET UP TRAINING OPTIONS
training_options = trainingOptions('adam', ...
    'MaxEpochs', dl_params.MaxEpochs, ...
    'MiniBatchSize', dl_params.MiniBatchSize, ...
    'InitialLearnRate', dl_params.InitialLearnRate, ...    
    'ValidationFrequency', dl_params.ValidationFrequency, ...
    'ValidationPatience', dl_params.ValidationPatience, ...
    'Shuffle', dl_params.Shuffle, ...
    'Verbose', dl_params.Verbose, ...
    'Plots', dl_params.Plots);

%% TRAINING MODEL FOR AZIMUTH
fprintf('\nTraining DL for Azimuth...\n');
training_options.ValidationData = {X_val_norm(:,final_features_az), Y_val_az};
dl_model_az = trainnet(X_train_norm(:,final_features_az), Y_train_az, layers_az, "mse", training_options);

%% TRAIN MODEL FOR TILT
fprintf('\nTraining DL for Tilt...\n');
training_options.ValidationData = {X_val_norm(:,final_features_ti), Y_val_ti};
dl_model_ti = trainnet(X_train_norm(:,final_features_ti), Y_train_ti, layers_ti, "mse", training_options);

%% VALIDATION SET PREDICTIONS
Y_pred_val_az = predict(dl_model_az, X_val_norm(:,final_features_az));
Y_pred_val_ti = predict(dl_model_ti, X_val_norm(:,final_features_ti));

%% TEST SET PREDICTIONS
Y_pred_test_az = predict(dl_model_az, X_test_norm(:,final_features_az));
Y_pred_test_ti = predict(dl_model_ti, X_test_norm(:,final_features_ti));

%% CALCULATE METRICS
% Metrics for Azimuth - Validation
val_metrics_az = calculate_metrics(Y_val_az, Y_pred_val_az, 'DL_Azimuth_Val');

% Metrics for Tilt - Validation
val_metrics_ti = calculate_metrics(Y_val_ti, Y_pred_val_ti, 'DL_Tilt_Val');

% Metrics for Azimuth - Test
test_metrics_az = calculate_metrics(Y_test_az, Y_pred_test_az, 'DL_Azimuth_Test');

% Métricas para Tilt - Test
test_metrics_ti = calculate_metrics(Y_test_ti, Y_pred_test_ti, 'DL_Tilt_Test');

% SHOW RESULTS
fprintf('\n=== EVALUATION OF THE AZIMUTH MODEL ===\n');
fprintf('MAE (Mean Absolute Error): %.2f degrees\n', test_metrics_az.MAE);
fprintf('MSE (Mean Squared Error): %.2f degrees²\n', test_metrics_az.MSE);
fprintf('RMSE (Root Mean Squared Error): %.2f degrees\n', test_metrics_az.RMSE);
fprintf('R² (Coefficient of Determination): %.4f\n', test_metrics_az.R2);

fprintf('\n=== EVALUATION OF THE TILT MODEL ===\n');
fprintf('MAE (Mean Absolute Error): %.2f degrees\n', test_metrics_ti.MAE);
fprintf('MSE (Mean Squared Error): %.2f degrees²\n', test_metrics_ti.MSE);
fprintf('RMSE (Root Mean Squared Error): %.2f degrees\n', test_metrics_ti.RMSE);
fprintf('R² (Coefficient of Determination): %.4f\n', test_metrics_ti.R2);

%% CREATE RESULTS STRUCTURE
results = struct();
results.models.azimuth = dl_model_az;
results.models.tilt = dl_model_ti;
results.parameters = dl_params;
results.normalization.mu_az = train_mu(final_features_az);
results.normalization.mu_ti = train_mu(final_features_ti);
results.normalization.sigma_az = train_sigma(final_features_az);
results.normalization.sigma_ti = train_sigma(final_features_ti);
results.models.az_var_names = feature_names(final_features_az);
results.models.ti_var_names = feature_names(final_features_ti);

% Predictions
results.predictions.val_azimuth = Y_pred_val_az;
results.predictions.val_tilt = Y_pred_val_ti;
results.predictions.test_azimuth = Y_pred_test_az;
results.predictions.test_tilt = Y_pred_test_ti;

% Metrics
results.metrics.val_azimuth = val_metrics_az;
results.metrics.val_tilt = val_metrics_ti;
results.metrics.test_azimuth = test_metrics_az;
results.metrics.test_tilt = test_metrics_ti;

fprintf('\nDeep Learning successfully trained!\n');
fprintf('MSE Test Azimuth: %.4f\n', test_metrics_az.MSE);
fprintf('MSE Test Tilt: %.4f\n', test_metrics_ti.MSE);

end

%% AUXILIARY FUNCTION TO CALCULATE METRICS
function metrics = calculate_metrics(y_true, y_pred, name)
    % Calculate metrics
    mse = mean((y_true - y_pred).^2);
    rmse = sqrt(mse);
    mae = mean(abs(y_true - y_pred));
    
    % R-squared
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2 = 1 - (ss_res / ss_tot);
    
    % Create structure
    metrics = struct();
    metrics.MSE = mse;
    metrics.RMSE = rmse;
    metrics.MAE = mae;
    metrics.R2 = r2;
    metrics.name = name;    
end