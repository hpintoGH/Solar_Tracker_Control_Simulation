function results = Random_Forest_MRMR(X_train, Y_train_az, Y_train_ti, ...
                                   X_val, Y_val_az, Y_val_ti, ...
                                   X_test, Y_test_az, Y_test_ti, feature_names)
%% FUNCTION: train random forest
% Trains Random Forest models for azimuth and tilt
%
% Entradas:
%   X_train, Y_train_az, Y_train_ti: Trainning data
%   X_val, Y_val_az, Y_val_ti: Validation data
%   X_test, Y_test_az, Y_test_ti: Test data
%   feature_names: Variable names
%
% Output:
%   results: Structure with results and metrics

%% REDUNDANCY ANALYSIS (MRMR)
% APPLY MRMR ONLY WITH TRAINING DATA
fprintf('Applying MRMR with time validation...\n');

% MRMR with training data only
[idx_az, scores_az] = fsrmrmr(X_train, Y_train_az);
[idx_ti, scores_ti] = fsrmrmr(X_train, Y_train_ti);

% Show feature ranking for azimuth
fprintf('\n=== RANKING AZIMUTH FEATURES (TEMPORAL) ===\n');
for i = 1:length(idx_az)
    fprintf('%2d. %s (Score: %.4f)\n', i, feature_names{idx_az(i)}, scores_az(idx_az(i)));
end

% Show feature ranking for tilt
fprintf('\n=== RANKING TILT FEATURES (TEMPORAL) ===\n');
for i = 1:length(idx_ti)
    fprintf('%2d. %s (Score: %.4f)\n', i, feature_names{idx_ti(i)}, scores_ti(idx_ti(i)));
end
fprintf('\n');
%% NESTED TEMPORAL VALIDATION
num_features_to_test = 3:min(9, size(X_train,2));

rmse_az_vs_nfeatures = zeros(length(num_features_to_test), 1);
rmse_ti_vs_nfeatures = zeros(length(num_features_to_test), 1);

for nf_idx = 1:length(num_features_to_test)
    num_features = num_features_to_test(nf_idx);
    
    % Select features according to MRMR ranking
    selected_features_az = idx_az(1:num_features);
    selected_features_ti = idx_ti(1:num_features);
    
    X_selected_az = X_train(:, selected_features_az);
    X_selected_ti = X_train(:, selected_features_ti);
    
     % Train with 2018-2021 data
    model_az = TreeBagger(100, X_selected_az, Y_train_az, ...
        'Method', 'regression', 'OOBPrediction', 'off');
    model_ti = TreeBagger(100, X_selected_ti, Y_train_ti, ...
        'Method', 'regression', 'OOBPrediction', 'off');
    
    % Validate with 2022 data
    pred_az = predict(model_az, X_val(:, selected_features_az));
    pred_ti = predict(model_ti, X_val(:, selected_features_ti));
    
    rmse_az_vs_nfeatures(nf_idx) = sqrt(mean((Y_val_az - pred_az).^2));
    rmse_ti_vs_nfeatures(nf_idx) = sqrt(mean((Y_val_ti - pred_ti).^2));
    
    fprintf('Features: %d, RMSE_Az: %.4f°, RMSE_TI: %.4f°\n', ...
        num_features, rmse_az_vs_nfeatures(nf_idx), rmse_ti_vs_nfeatures(nf_idx));
end

% Select optimal number of features based on validation performance
[~, optimal_nf_az] = min(rmse_az_vs_nfeatures);
[~, optimal_nf_ti] = min(rmse_ti_vs_nfeatures);

optimal_features_az = num_features_to_test(optimal_nf_az);
optimal_features_ti = num_features_to_test(optimal_nf_ti);

fprintf('\n=== OPTIMAL FEATURE SELECTION RESULTS ===\n');
fprintf('Optimal features for Azimuth: %d\n', optimal_features_az);
fprintf('Optimal features for Tilt: %d\n', optimal_features_ti);
fprintf('Best RMSE Az: %.4f°\n', min(rmse_az_vs_nfeatures));
fprintf('Best RMSE Ti: %.4f°\n', min(rmse_ti_vs_nfeatures));

%% ENTRENAMIENTO FINAL CON FEATURES ÓPTIMAS
final_features_az = idx_az(1:optimal_features_az);
final_features_ti = idx_ti(1:optimal_features_ti);
final_features_az_names = feature_names(final_features_az);
final_features_ti_names = feature_names(final_features_ti);

% Entrenar con todos los datos de entrenamiento (2018-2021)
X_final_train_az = X_train(:, final_features_az);
X_final_train_ti = X_train(:, final_features_ti);

model_azimuth_final = TreeBagger(200, X_final_train_az, Y_train_az, ...
    'Method', 'regression', 'OOBPrediction', 'on');
model_tilt_final = TreeBagger(200, X_final_train_ti, Y_train_ti, ...
    'Method', 'regression', 'OOBPrediction', 'on');

%% EVALUACIÓN FINAL EN DATOS NO VISTOS (2022)
pred_val_az = predict(model_azimuth_final, X_val(:, final_features_az));
pred_val_ti = predict(model_tilt_final, X_val(:, final_features_ti));

% Métricas para Azimuth - Validación
val_metrics_az = calculate_metrics(Y_val_az, pred_val_az, 'RF_Azimuth_Val');

% Métricas para Tilt - Validación
val_metrics_ti = calculate_metrics(Y_val_ti, pred_val_ti, 'RF_Tilt_Val');

fprintf('\n=== FINAL EVALUATION (2022) ===\n');
fprintf('RMSE Azimut: %.4f°\n', val_metrics_az.RMSE);
fprintf('RMSE Elevación: %.4f°\n', val_metrics_ti.RMSE);

%% FINAL TEST IN 2023 (DATA COMPLETELY UNSEEN)
pred_test_az = predict(model_azimuth_final, X_test(:, final_features_az));
pred_test_ti = predict(model_tilt_final, X_test(:, final_features_ti));

% Metrics for Azimuth - Test
test_metrics_az = calculate_metrics(Y_test_az, pred_test_az, 'RF_Azimuth_Test');

% Metrics for Tilt - Test
test_metrics_ti = calculate_metrics(Y_test_ti, pred_test_ti, 'RF_Tilt_Test');

% SHOW RESULTS
fprintf('\n=== EVALUATION OF THE AZIMUTH MODEL ===\n');
fprintf('MAE (Mean Absolute Error): %.2f grados\n', test_metrics_az.MAE);
fprintf('MSE (Mean Square Error): %.2f grados\n', test_metrics_az.MSE);
fprintf('RMSE (Root Mean Square Error): %.2f grados\n', test_metrics_az.RMSE);
fprintf('R² (Coefficient of determination): %.4f\n', test_metrics_az.R2);

fprintf('\n=== EVALUACIÓN DEL MODELO DE INCLINACIÓN ===\n');
fprintf('MAE (Mean Absolute Error): %.2f grados\n', test_metrics_ti.MAE);
fprintf('MSE (Mean Square Error): %.2f grados\n', test_metrics_ti.MSE);
fprintf('RMSE (Root Mean Square Error): %.2f grados\n', test_metrics_ti.RMSE);
fprintf('R² (Coefficient of determination): %.4f\n', test_metrics_ti.R2);

%% ERROR DISTRIBUTION ANALYSIS
figure('Name', 'Error Analysis', 'Position', [100, 100, 1200, 600]);

% Error Histogram - Azimuth
subplot(2, 3, 1);
error_dist_az = pred_test_az - Y_test_az;
histogram(error_dist_az, 30, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
title('Error Distribution - Azimuth');
xlabel('Error (degrees)');
ylabel('Frequency');
grid on;

% Error Histogram - Elevation
subplot(2, 3, 2);
error_dist_ti = pred_test_ti - Y_test_ti;
histogram(error_dist_ti, 30, 'FaceColor', 'red', 'FaceAlpha', 0.7);
title('Error Distribution - Tilt');
xlabel('Error (degrees)');
ylabel('Frequency');
grid on;

% Scatter Plot - Predicted vs Actual (Azimuth)
subplot(2, 3, 3);
h1 = scatter(Y_test_az, pred_test_az, 'blue', 'filled');
h1.MarkerFaceAlpha = 0.6;
hold on;
plot([min(Y_test_az), max(Y_test_az)], [min(Y_test_az), max(Y_test_az)], 'r--', 'LineWidth', 2);
title('Predicted vs Actual - Azimuth');
xlabel('Actual Value (grados)');
ylabel('Predicted Value (grados)');
grid on;
legend('Data', 'Ideal Line', 'Location', 'best');

% Scatter Plot - Predicted vs Actual (Tilt)
subplot(2, 3, 4);
h2 = scatter(Y_test_ti, pred_test_ti, 'red', 'filled');
h2.MarkerFaceAlpha = 0.6;
hold on;
plot([min(Y_test_ti), max(Y_test_ti)], [min(Y_test_ti), max(Y_test_ti)], 'r--', 'LineWidth', 2);
title('Predicted vs Actual - Tilt');
xlabel('Actual Value (grados)');
ylabel('Predicted Value (grados)');
grid on;
legend('Data', 'Ideal Line', 'Location', 'best');

% Residuals vs Predictions (Azimuth)
subplot(2, 3, 5);
h3 = scatter(pred_test_az, error_dist_az, 'blue', 'filled');
h3.MarkerFaceAlpha = 0.6;
hold on;
yline(0, 'r--', 'LineWidth', 2);
title('Residuals vs Predictions - Azimuth');
xlabel('Predictions (grados)');
ylabel('Residuals (grados)');
grid on;

% Residuals vs Predictions (Tilt)
subplot(2, 3, 6);
h4 = scatter(pred_test_ti, error_dist_ti, 'red', 'filled');
h4.MarkerFaceAlpha = 0.6;
hold on;
yline(0, 'r--', 'LineWidth', 2);
title('Residuals vs Predictions - Tilt');
xlabel('Predictions (grados)');
ylabel('Residuals (grados)');
grid on;

%% ADDITIONAL STATISTICS
fprintf('\n=== ADDITIONAL STATISTICS ===\n');
fprintf('Standard deviation of azimuth errors: %.2f grados\n', std(error_dist_az));
fprintf('Standard deviation of tilt errors: %.2f grados\n', std(error_dist_ti));

% Absolute error percentiles
percentiles = [50, 75, 90, 95];
fprintf('\nAbsolute error percentiles:\n');
for p = percentiles
    az_perc = prctile(abs(error_dist_az), p);
    ti_perc = prctile(abs(error_dist_ti), p);
    fprintf('P%d - Azimuth: %.2f°, Tilt: %.2f°\n', p, az_perc, ti_perc);
end

%% CREATE RESULTS STRUCTURE
results = struct();
results.models.azimuth = model_azimuth_final;
results.models.az_var_names = final_features_az_names;
results.models.tilt = model_tilt_final;
results.models.ti_var_names = final_features_ti_names;

% Predictions
results.predictions.val_azimuth = pred_val_az;
results.predictions.val_tilt = pred_val_ti;
results.predictions.test_azimuth = pred_test_az;
results.predictions.test_tilt = pred_test_ti;

% Metrics
results.metrics.val_azimuth = val_metrics_az;
results.metrics.val_tilt = val_metrics_ti;
results.metrics.test_azimuth = test_metrics_az;
results.metrics.test_tilt = test_metrics_ti;

% Figure
savefig("rf_statistics.fig")

end

%% AUXILIARY FUNCTION TO CALCULATE METRICS
function metrics = calculate_metrics(y_true, y_pred, name)
        
    % Error Cuadrático Medio (MSE)
    mse = mean((y_true - y_pred).^2);
    % Raíz del Error Cuadrático Medio (RMSE)
    rmse = sqrt(mse);
    % Error Absoluto Medio (MAE)
    mae = mean(abs(y_true - y_pred));

    % Coeficiente de Determinación (R²)
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