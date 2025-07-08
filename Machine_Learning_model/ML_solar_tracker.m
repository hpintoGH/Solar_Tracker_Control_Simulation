%% Creating Machine Learning Models for Solar Trackers
% Uses solar trajectory and meteorological data to control dual-axis trackers
clc
clear
%% 1. DATA PREPARATION
fprintf('Preparing data...\n');
data_preparation; % Call the data preparation script

%% 2. TRAIN RANDOM FOREST
fprintf('\nTraining Random Forest...\n');
tic;
rf_results = Random_Forest_MRMR(X_train, Y_train_az, Y_train_ti, ...
                                   X_val, Y_val_az, Y_val_ti, ...
                                   X_test, Y_test_az, Y_test_ti, feature_names);

rf_time = toc;
fprintf('\nRandom Forest completed in %.2f seconds\n', rf_time);

%% 3. TRAINING DEEP LEARNING MLP

fprintf('\nTraining Deep Learning...\n');
tic;
dl_results = MLP_dl_model(X_train, Y_train_az, Y_train_ti, ...
                               X_val, Y_val_az, Y_val_ti, ...
                               X_test, Y_test_az, Y_test_ti, idx_norm, feature_names);
dl_time = toc;
fprintf('\nDeep Learning completed in %.2f seconds\n', dl_time);

%% 4. COMPARE RESULTS
fprintf('Comparing results...\n');
compare_models(rf_results, dl_results);

%% 5. SAVE RESULTS
save('model_comparison_results.mat', 'rf_results', 'dl_results', 'rf_time', 'dl_time');