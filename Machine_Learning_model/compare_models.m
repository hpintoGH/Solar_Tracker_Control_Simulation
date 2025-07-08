function compare_models(rf_results, dl_results)
%% FUNCTION: compare_models
% Compares the results of Random Forest vs. Deep Learning
%
% Inputs:
% rf_results: Random Forest results
% dl_results: Deep Learning results

fprintf('\n=== MODEL COMPARISON ===\n');

%% COMPARATIVE TABLE OF METRICS - AZIMUTH
fprintf('\n--- AZIMUTH - TEST SET ---\n');
azimuth_table = table();
azimuth_table.Modelo = {'Random Forest'; 'Deep Learning'};
azimuth_table.MSE = [rf_results.metrics.test_azimuth.MSE; dl_results.metrics.test_azimuth.MSE];
azimuth_table.RMSE = [rf_results.metrics.test_azimuth.RMSE; dl_results.metrics.test_azimuth.RMSE];
azimuth_table.MAE = [rf_results.metrics.test_azimuth.MAE; dl_results.metrics.test_azimuth.MAE];
azimuth_table.R2 = [rf_results.metrics.test_azimuth.R2; dl_results.metrics.test_azimuth.R2];

disp(azimuth_table);

%% COMPARATIVE TABLE OF METRICS - TILT
fprintf('\n--- TILT - TEST SET ---\n');
tilt_table = table();
tilt_table.Modelo = {'Random Forest'; 'Deep Learning'};
tilt_table.MSE = [rf_results.metrics.test_tilt.MSE; dl_results.metrics.test_tilt.MSE];
tilt_table.RMSE = [rf_results.metrics.test_tilt.RMSE; dl_results.metrics.test_tilt.RMSE];
tilt_table.MAE = [rf_results.metrics.test_tilt.MAE; dl_results.metrics.test_tilt.MAE];
tilt_table.R2 = [rf_results.metrics.test_tilt.R2; dl_results.metrics.test_tilt.R2];

disp(tilt_table);

%% DETERMINE BEST MODEL
fprintf('\n--- COMPARATIVE ANALYSIS ---\n');

% Azimuth
if rf_results.metrics.test_azimuth.MSE < dl_results.metrics.test_azimuth.MSE
    fprintf('AZIMUTH: Random Forest has lower MSE (%.4f vs %.4f)\n', ...
            rf_results.metrics.test_azimuth.MSE, dl_results.metrics.test_azimuth.MSE);
    best_az = 'Random Forest';
else
    fprintf('AZIMUTH: Deep Learning has lower MSE (%.4f vs %.4f)\n', ...
            dl_results.metrics.test_azimuth.MSE, rf_results.metrics.test_azimuth.MSE);
    best_az = 'Deep Learning';
end

% Tilt
if rf_results.metrics.test_tilt.MSE < dl_results.metrics.test_tilt.MSE
    fprintf('TILT: Random Forest has lower MSE (%.4f vs %.4f)\n', ...
            rf_results.metrics.test_tilt.MSE, dl_results.metrics.test_tilt.MSE);
    best_ti = 'Random Forest';
else
    fprintf('TILT: Deep Learning has lower MSE (%.4f vs %.4f)\n', ...
            dl_results.metrics.test_tilt.MSE, rf_results.metrics.test_tilt.MSE);
    best_ti = 'Deep Learning';
end

fprintf('\nBest model for AZIMUTH: %s\n', best_az);
fprintf('Best model for TILT: %s\n', best_ti);

%% COMPARATIVE GRAPHICS
create_comparison_plots(rf_results, dl_results);

end

%% FUNCTION TO CREATE COMPARATIVE GRAPHICS
function create_comparison_plots(rf_results, dl_results)

fprintf('\nCreating comparison charts...\n');

%% GRAPH 1: MSE Comparison
figure('Name', 'Comparison MSE', 'Position', [100, 100, 800, 400]);

subplot(1, 2, 1);
mse_az = [rf_results.metrics.test_azimuth.MSE, dl_results.metrics.test_azimuth.MSE];
bar(mse_az);
set(gca, 'XTickLabel', {'Random Forest', 'Deep Learning'});
title('MSE - Azimuth');
ylabel('MSE');
grid on;

subplot(1, 2, 2);
mse_ti = [rf_results.metrics.test_tilt.MSE, dl_results.metrics.test_tilt.MSE];
bar(mse_ti);
set(gca, 'XTickLabel', {'Random Forest', 'Deep Learning'});
title('MSE - Tilt');
ylabel('MSE');
grid on;

%% GRAPH 2: Scatter plots of predictions vs actual values
figure('Name', 'Predictions vs actual values', 'Position', [200, 200, 1200, 800]);

% Get test data (assuming it is available in the base workspace)
try
    Y_test_az = evalin('base', 'Y_test_az');
    Y_test_ti = evalin('base', 'Y_test_ti');
    
    % Azimuth - Random Forest
    subplot(2, 2, 1);
    s1 = scatter(Y_test_az, rf_results.predictions.test_azimuth, 20, 'b', 'filled');
    s1.MarkerFaceAlpha = 0.6;
    hold on;
    plot([min(Y_test_az), max(Y_test_az)], [min(Y_test_az), max(Y_test_az)], 'r--', 'LineWidth', 2);
    xlabel('Actual Values');
    ylabel('Predictions');
    title(sprintf('RF - Azimuth (R² = %.3f)', rf_results.metrics.test_azimuth.R2));
    grid on;
    axis equal;
    
    % Azimuth - Deep Learning
    subplot(2, 2, 2);
    s2 = scatter(Y_test_az, dl_results.predictions.test_azimuth, 20, 'g', 'filled');
    s2.MarkerFaceAlpha = 0.6;
    hold on;
    plot([min(Y_test_az), max(Y_test_az)], [min(Y_test_az), max(Y_test_az)], 'r--', 'LineWidth', 2);
    xlabel('Actual Values');
    ylabel('Predictions');
    title(sprintf('DL - Azimuth (R² = %.3f)', dl_results.metrics.test_azimuth.R2));
    grid on;
    axis equal;
    
    % Tilt - Random Forest
    subplot(2, 2, 3);
    s3 = scatter(Y_test_ti, rf_results.predictions.test_tilt, 20, 'b', 'filled');
    s3.MarkerFaceAlpha = 0.6;
    hold on;
    plot([min(Y_test_ti), max(Y_test_ti)], [min(Y_test_ti), max(Y_test_ti)], 'r--', 'LineWidth', 2);
    xlabel('Actual Values');
    ylabel('Predictions');
    title(sprintf('RF - Tilt (R² = %.3f)', rf_results.metrics.test_tilt.R2));
    grid on;
    axis equal;
    
    % Tilt - Deep Learning
    subplot(2, 2, 4);
    s4 = scatter(Y_test_ti, dl_results.predictions.test_tilt, 20, 'g', 'filled');
    s4.MarkerFaceAlpha = 0.6;
    hold on;
    plot([min(Y_test_ti), max(Y_test_ti)], [min(Y_test_ti), max(Y_test_ti)], 'r--', 'LineWidth', 2);
    xlabel('Actual Values');
    ylabel('Predictions');
    title(sprintf('DL - Tilt (R² = %.3f)', dl_results.metrics.test_tilt.R2));
    grid on;
    axis equal;
    
catch
    fprintf('The scatter plots could not be created. Make sure Y_test_az and Y_test_ti are in the workspace.\n');
end

%% GRÁFICO 3: Comparison of all metrics
figure('Name', 'Metric Comparison', 'Position', [300, 300, 1000, 600]);

% Preparar datos para el gráfico
metrics_names = {'MSE', 'RMSE', 'MAE', 'R²'};
rf_metrics_az = [rf_results.metrics.test_azimuth.MSE, rf_results.metrics.test_azimuth.RMSE, ...
                 rf_results.metrics.test_azimuth.MAE, rf_results.metrics.test_azimuth.R2];
dl_metrics_az = [dl_results.metrics.test_azimuth.MSE, dl_results.metrics.test_azimuth.RMSE, ...
                 dl_results.metrics.test_azimuth.MAE, dl_results.metrics.test_azimuth.R2];

rf_metrics_ti = [rf_results.metrics.test_tilt.MSE, rf_results.metrics.test_tilt.RMSE, ...
                 rf_results.metrics.test_tilt.MAE, rf_results.metrics.test_tilt.R2];
dl_metrics_ti = [dl_results.metrics.test_tilt.MSE, dl_results.metrics.test_tilt.RMSE, ...
                 dl_results.metrics.test_tilt.MAE, dl_results.metrics.test_tilt.R2];

% Azimuth
subplot(1, 2, 1);
x = 1:length(metrics_names);
width = 0.35;
bar(x - width/2, rf_metrics_az, width, 'DisplayName', 'Random Forest');
hold on;
bar(x + width/2, dl_metrics_az, width, 'DisplayName', 'Deep Learning');
set(gca, 'XTickLabel', metrics_names);
title('Metrics - Azimuth');
legend('Location', 'best');
grid on;

% Tilt
subplot(1, 2, 2);
bar(x - width/2, rf_metrics_ti, width, 'DisplayName', 'Random Forest');
hold on;
bar(x + width/2, dl_metrics_ti, width, 'DisplayName', 'Deep Learning');
set(gca, 'XTickLabel', metrics_names);
title('Metrics - Tilt');
legend('Location', 'best');
grid on;

fprintf('Charts created successfully!\n');

end