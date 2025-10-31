clear
clc

data_years = 2018:2023;
tz = "America/New_York";

% Read data
disp("Loading data...")
for l = 1:length(data_years)
    file_path = "spa_solar_position_dataset_" + string(data_years(l)) + ".csv";
    opts = detectImportOptions(file_path);
    opts = setvaropts(opts, 'datetime', 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    data_table = readtimetable(file_path, opts); % File with solar positions

    % Filter only valid daytime data (zenith < 90 and GHI > 0)
    data_table = data_table(data_table.zenith_deg < 90 & data_table.GHI > 0, :);
    fprintf("Valid daytime data: %d records\n", size(data_table,1))

    % Extraterrestrial DNI calculation for the Perez model
    fprintf("\nCalculating extraterrestrial DNI...")
    dates = data_table.datetime;
    day_of_year = day(dates, 'dayofyear');
    years = unique(year(dates));
    fprintf("\nYears in the data: %d - %d", min(years), max(years));
    data_table.dni_extra = pvl_extraradiation(day_of_year);
    % Relative airmass
    data_table.AM = pvl_relativeairmass(data_table.zenith_deg, 'pickering2002');

    % Define search range for tilt and azimuth
    tilt_values = 0:1:90;           % from 0° to 90° every 1°
    azimuth_values = 0:1:360;      % from 0° to 360° every 1°

    fprintf("\nEvaluating %d tilt angles and %d azimuth angles", length(tilt_values), length(azimuth_values))
    fprintf("\nTotal combinations per record: %d\n", length(tilt_values) * length(azimuth_values))
   
    % Close any previous pool
    delete(gcp('nocreate'));
    % Create a pool with 2 cores
    parpool('local', 2);

    % Process data with a progress bar
    fprintf("\nProcessing data...")
    start_time = tic;

    n = height(data_table);

    % Prellocating
    datetime_values = data_table.datetime;
    results_cell = cell(n, 1);

    parfor i = 1:n
        row = data_table(i, :);
        result = find_optimal_angles_vectorized(row, tilt_values, azimuth_values);
        result = struct2table(result);
        results_cell{i} = result;
    end

    elapsed_time = toc;
    fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

    % Combine the results into a single table
    results_table = vertcat(results_cell{:});

    % Add datetime column
    results_table.datetime = datetime_values;

    writetable(results_table, "optimized_tilt_azimuth_results_" + string(data_years(l)) + ".csv")
    fprintf("\nResults saved in 'optimized_tilt_azimuth_results" + string(data_years(l)) + ".csv'\n")

    fprintf("\nDNI_extra range: %.0f - %.0f W/m²", min(data_table.dni_extra), max(data_table.dni_extra))
    fprintf("\nDNI_extra average: %.0f W/m²\n", mean(data_table.dni_extra, "omitnan"))
    
    delete(gcp('nocreate'));
end

% Vectorize the calculation for better efficiency
function result = find_optimal_angles_vectorized(row, tilt_values, azimuth_values)
    % Find optimal angles for a specific row using vectorization
    
    % Create a meshgrid for all combinations
    [tilt_grid, azimuth_grid] = ndgrid(tilt_values, azimuth_values);
    tilt_flat = tilt_grid(:);
    azimuth_flat = azimuth_grid(:);
    
    % Compute irradiance for all combinations at once
    
    [SkyDiffuse,SkyDiffuse_Iso,SkyDiffuse_Cir,SkyDiffuse_Hor] = pvl_perez(tilt_flat, azimuth_flat, ...
        row.DHI, row.DNI, row.dni_extra, row.zenith_deg, row.azimuth_deg, row.AM);
    % Angle of incidence
    aoi = pvl_getaoi(tilt_flat, azimuth_flat, row.zenith_deg, row.azimuth_deg);
    I_beam = row.DNI .* cos(aoi .* (pi/180));
    % Estimation of diffuse irradiance from ground reflections
    GR = pvl_grounddiffuse(tilt_flat, row.GHI, row.Surface_Albedo);
    
    poa_values = SkyDiffuse + I_beam + GR;
    
    % Find the index of the maximum value
    [~, max_idx] = max(poa_values);
    
    result.best_tilt = tilt_flat(max_idx);
    result.best_azimuth = azimuth_flat(max_idx);
    result.poa = poa_values(max_idx);
end