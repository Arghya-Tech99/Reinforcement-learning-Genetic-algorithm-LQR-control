%% MONTE CARLO SIMULATION FOR RL-GA OPTIMIZED CO-LQR CONTROLLER
% Using Eurocode 8 drift criterion (h/200 = 0.015 m for 3m story height)

clear; clc; close all;

%% ==================== 1. LOAD OPTIMIZED CONTROLLER ====================
fprintf('Loading optimized controller from Optimal_QR.mat...\n');
if ~exist('Optimal_QR.mat', 'file')
    error('Optimal_QR.mat not found. Run main_RLGA_COLQR.m first.');
end
load('Optimal_QR.mat', 'best_Q', 'best_R');

% Display the optimized weights
fprintf('Optimized Q factor: %.2e\n', best_Q(1,1));
fprintf('Optimized R factor: %.2e\n', best_R);

%% ==================== 2. SYSTEM PARAMETERS ====================
% Building parameters
n = 10;                      % number of stories
m_nom = 3.5e4;               % nominal story mass (kg)
k_nom = 6.5e7;               % nominal story stiffness (N/m)
c_nom = 6.0e5;               % nominal story damping (N·s/m)
storyHeight = 3.0;            % story height (m)

% Eurocode 8 drift criterion
% Eurocode 8: Inter-story drift limit = h/200 for buildings with ductile behavior
% For 3m story height: 3.0/200 = 0.015 m
driftLimit = storyHeight / 200;  % 0.015 m (15 mm)
fprintf('\nEurocode 8 drift limit: %.4f m (%.1f mm)\n', driftLimit, driftLimit*1000);

% MR damper parameters
mr_max_force = 200000;        % maximum MR damper force (N) = 200 kN

% Time parameters
dt = 0.002;                   % time step (s)
T_total = 10;                  % total simulation time (s)
t = (0:dt:T_total)';
nSteps = length(t);

%% ==================== 3. COMPUTE NOMINAL LQR GAIN ====================
% Build nominal system matrices first
Ms_nom = m_nom * eye(n);

% Stiffness matrix (shear building)
main_diag_K = [2*k_nom * ones(n-1, 1); k_nom];
off_diag_K = -k_nom * ones(n-1, 1);
Ks_nom = diag(main_diag_K) + diag(off_diag_K, 1) + diag(off_diag_K, -1);

% Damping matrix
main_diag_C = [2*c_nom * ones(n-1, 1); c_nom];
off_diag_C = -c_nom * ones(n-1, 1);
Cs_nom = diag(main_diag_C) + diag(off_diag_C, 1) + diag(off_diag_C, -1);

% State-space matrices for nominal system
A_nom = [zeros(n), eye(n); -Ms_nom\Ks_nom, -Ms_nom\Cs_nom];
B_nom = [zeros(n,1); -Ms_nom \ [1; zeros(n-1,1)]];

% Compute nominal LQR gain (this is what RL-GA optimized)
K_lqr_nominal = lqr(A_nom, B_nom, best_Q, best_R);
fprintf('\nNominal LQR gain computed (size %d×%d)\n', size(K_lqr_nominal));

%% ==================== 4. UNCERTAINTY DEFINITION ====================
fprintf('\nDefining uncertainty distributions...\n');

% Function to sample uncertainties - store parameters for heatmap
sample_uncertainties = @() struct(...
    'mass_scale',      1 + 0.10*randn, ...                % ±10% normal
    'stiffness_scale', 1 + 0.15*randn, ...                % ±15% normal
    'damping_scale',   1 + 0.20*randn, ...                % ±20% normal
    'PGA_scale',       0.5 + 1.0*rand, ...                % uniform [0.5, 1.5]
    'seed',            randi(1e6) ...                     % random seed
);

%% ==================== 5. MONTE CARLO SETTINGS ====================
Nsim = 6;                   % Number of Monte Carlo runs
results = NaN(Nsim, 2);        % [peak_drift, peak_force]
successful = 0;

% Storage for all runs data
all_runs_displacements = cell(Nsim, 1);
all_runs_peak_drifts = zeros(Nsim, n);  % Peak drift per floor for each run
all_runs_params = zeros(Nsim, 4);        % [mass_scale, stiffness_scale, damping_scale, PGA_scale]
all_runs_peak_forces = zeros(Nsim, 1);

fprintf('\nStarting Monte Carlo simulations (%d runs)...\n', Nsim);

%% ==================== 6. MAIN MONTE CARLO LOOP ====================
for sim = 1:Nsim
    fprintf('Run %d/%d: ', sim, Nsim);
    
    try
        % --- Sample uncertainties ---
        params = sample_uncertainties();
        
        % Store parameters for heatmap
        all_runs_params(sim, :) = [params.mass_scale, params.stiffness_scale, ...
                                    params.damping_scale, params.PGA_scale];
        
        % --- Generate earthquake with given seed ---
        rng(params.seed);
        [ag, ~] = generate_nsae_kanai_tajimi_2023Brandao();
        ag = params.PGA_scale * ag;
        ag = interp1(linspace(0, T_total, length(ag)), ag, t)';
        
        % --- Scale structural parameters ---
        m = m_nom * params.mass_scale;
        k = k_nom * params.stiffness_scale;
        c = c_nom * params.damping_scale;
        
        % --- Build system matrices with scaled parameters ---
        Ms = m * eye(n);
        
        % Stiffness matrix
        main_diag_K = [2*k * ones(n-1, 1); k];
        off_diag_K = -k * ones(n-1, 1);
        Ks = diag(main_diag_K) + diag(off_diag_K, 1) + diag(off_diag_K, -1);
        
        % Damping matrix
        main_diag_C = [2*c * ones(n-1, 1); c];
        off_diag_C = -c * ones(n-1, 1);
        Cs = diag(main_diag_C) + diag(off_diag_C, 1) + diag(off_diag_C, -1);
        
        % State-space matrices
        A = [zeros(n), eye(n); -Ms\Ks, -Ms\Cs];
        B = [zeros(n,1); -Ms \ [1; zeros(n-1,1)]];
        E = [zeros(n,1); -ones(n,1)];
        
        % Use fixed LQR gain
        K_lqr = K_lqr_nominal;
        
        % Initialize state
        x = zeros(2*n, 1);
        u_control = 0;
        
        % Storage
        displacements = zeros(nSteps, n);
        control_force = zeros(nSteps, 1);
        
        % --- Numerical integration ---
        for i = 1:nSteps
            % Store current state
            displacements(i, :) = x(1:n)';
            control_force(i) = u_control;
            
            % Get earthquake acceleration
            xg_ddot = ag(i);
            
            % Compute LQR optimal force
            f_optimal = -K_lqr * x;
            
            % Apply MR damper constraints
            if abs(f_optimal) <= mr_max_force
                u_control = f_optimal;
            else
                u_control = sign(f_optimal) * mr_max_force;
            end
            
            % Runge-Kutta 4th order
            k1 = A * x + B * u_control + E * xg_ddot;
            x_mid = x + 0.5 * dt * k1;
            k2 = A * x_mid + B * u_control + E * xg_ddot;
            x_mid = x + 0.5 * dt * k2;
            k3 = A * x_mid + B * u_control + E * xg_ddot;
            x_end = x + dt * k3;
            k4 = A * x_end + B * u_control + E * xg_ddot;
            
            x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
        end
        
        % --- CORRECTED: Compute story drifts (relative displacements) ---
        story_drifts = zeros(nSteps, n);
        for floor = 1:n
            if floor == 1
                story_drifts(:, floor) = displacements(:, floor);
            else
                story_drifts(:, floor) = displacements(:, floor) - displacements(:, floor-1);
            end
        end
        
        % --- Performance metrics ---
        peak_drift = max(abs(story_drifts(:)));           % maximum absolute drift
        peak_force = max(abs(control_force));              % peak control force
        
        % Store peak drift per floor
        for floor = 1:n
            all_runs_peak_drifts(sim, floor) = max(abs(story_drifts(:, floor)));
        end
        
        % Store results
        results(sim, :) = [peak_drift, peak_force];
        all_runs_peak_forces(sim) = peak_force;
        
        successful = successful + 1;
        
        % Status update
        drift_ratio = peak_drift / driftLimit * 100;
        if peak_drift <= driftLimit
            fprintf('✓ peak drift = %.4f m (%.1f%% of Eurocode limit)\n', peak_drift, drift_ratio);
        else
            fprintf('⚠ peak drift = %.4f m (%.1f%% of Eurocode limit) - EXCEEDS LIMIT\n', peak_drift, drift_ratio);
        end
        
    catch ME
        fprintf('✗ failed: %s\n', ME.message);
    end
end

%% ==================== 7. RESULTS ANALYSIS ====================
fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Successful runs: %d/%d (%.1f%%)\n', successful, Nsim, 100*successful/Nsim);

if successful == 0
    error('No successful simulations to analyze.');
end

% Remove failed runs
valid_idx = ~any(isnan(results), 2);
valid_results = results(valid_idx, :);
valid_peak_drifts = valid_results(:,1);
valid_peak_forces = valid_results(:,2) / 1000;  % Convert to kN
valid_params = all_runs_params(valid_idx, :);
valid_floor_drifts = all_runs_peak_drifts(valid_idx, :);

% Calculate success rate for Eurocode 8 criterion
success_rate = 100 * sum(valid_peak_drifts <= driftLimit) / length(valid_peak_drifts);
fprintf('\n===== PERFORMANCE AGAINST EUROCODE 8 =====\n');
fprintf('Eurocode 8 drift limit (h/200): %.4f m\n', driftLimit);
fprintf('Success rate: %.1f%% (%d/%d runs)\n', success_rate, sum(valid_peak_drifts <= driftLimit), length(valid_peak_drifts));

%% ==================== 8. REQUIRED OUTPUT PLOTS ====================

%--------------------------------------------------------------------------
% FIGURE 1: Peak Drift Distribution Histogram (Eurocode 8)
%--------------------------------------------------------------------------
figure('Position', [100, 100, 900, 600]);

% Create histogram
histogram(valid_peak_drifts, 30, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

% Add vertical line for Eurocode 8 limit
xline(driftLimit, 'r--', 'LineWidth', 3);

% Add vertical line for mean value
xline(mean(valid_peak_drifts), 'g-', 'LineWidth', 2);

xlabel('Peak Drift (m)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 13, 'FontWeight', 'bold');
title('Peak Drift Distribution - Eurocode 8 Criterion', 'FontSize', 14, 'FontWeight', 'bold');
legend('Simulations', sprintf('Eurocode 8 Limit (%.4f m)', driftLimit), ...
       sprintf('Mean (%.4f m)', mean(valid_peak_drifts)), 'Location', 'best');
grid on;

% Add statistics text box
text_str = {sprintf('Eurocode 8 Limit: %.4f m', driftLimit);
            sprintf('Mean: %.4f m', mean(valid_peak_drifts));
            sprintf('Std Dev: %.4f m', std(valid_peak_drifts));
            sprintf('95%% CI: [%.4f, %.4f] m', prctile(valid_peak_drifts,2.5), prctile(valid_peak_drifts,97.5));
            sprintf('Success Rate: %.1f%%', success_rate)};
annotation('textbox', [0.6, 0.6, 0.25, 0.2], 'String', text_str, ...
           'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 11, ...
           'HorizontalAlignment', 'left');

% Save figure
saveas(gcf, 'Eurocode8_Peak_Drift_Distribution.png');
fprintf('✓ Saved: Eurocode8_Peak_Drift_Distribution.png\n');

%--------------------------------------------------------------------------
% FIGURE 2: Story Drift Profile (Mean ± Std for each floor) with Eurocode 8 limit
%--------------------------------------------------------------------------
figure('Position', [100, 100, 900, 600]);

% Calculate mean and standard deviation for each floor
mean_floor_drifts = mean(valid_floor_drifts, 1);
std_floor_drifts = std(valid_floor_drifts, 0, 1);
max_floor_drifts = max(valid_floor_drifts, [], 1);

% Create error bar plot
errorbar(1:n, mean_floor_drifts, std_floor_drifts, 'o-', 'LineWidth', 2.5, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0.2 0.6 0.8], 'Color', [0 0 0.5]);
hold on;

% Add horizontal line for Eurocode 8 limit
yline(driftLimit, 'r--', 'LineWidth', 3);

% Add max values as stars
plot(1:n, max_floor_drifts, 'r*', 'MarkerSize', 10, 'LineWidth', 1.5);

xlabel('Floor Number', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Peak Drift (m)', 'FontSize', 13, 'FontWeight', 'bold');
title('Story Drift Profile - Eurocode 8 Compliance', 'FontSize', 14, 'FontWeight', 'bold');
legend('Mean ± Std Dev', sprintf('Eurocode 8 Limit (%.4f m)', driftLimit), ...
       'Maximum Observed', 'Location', 'best');
grid on;
xlim([0.5, 10.5]);
ylim([0, max(max_floor_drifts)*1.1]);

% Add text showing if all floors meet criteria
floors_below_limit = all(mean_floor_drifts + std_floor_drifts < driftLimit);
if floors_below_limit
    text_str = sprintf('✓ All floors meet Eurocode 8 criterion\n(Mean + 1σ < %.4f m)', driftLimit);
    text_color = 'green';
else
    text_str = sprintf('⚠ Some floors exceed Eurocode 8 criterion\n(Mean + 1σ > %.4f m)', driftLimit);
    text_color = 'red';
end
annotation('textbox', [0.15, 0.8, 0.35, 0.08], 'String', text_str, ...
           'BackgroundColor', 'white', 'EdgeColor', text_color, ...
           'FontSize', 11, 'Color', text_color, 'FontWeight', 'bold');

% Add floor-by-floor statistics table
fprintf('\n--- Floor-wise Eurocode 8 Compliance ---\n');
fprintf('Floor\tMean (m)\tStd (m)\tMax (m)\tCompliance\n');
for floor = 1:n
    compliant = mean_floor_drifts(floor) + std_floor_drifts(floor) < driftLimit;
    if compliant
        status = '✓';
    else
        status = '✗';
    end
    fprintf('%d\t%.4f\t\t%.4f\t%.4f\t%s\n', floor, mean_floor_drifts(floor), ...
            std_floor_drifts(floor), max_floor_drifts(floor), status);
end

% Save figure
saveas(gcf, 'Eurocode8_Story_Drift_Profile.png');
fprintf('✓ Saved: Eurocode8_Story_Drift_Profile.png\n');

%--------------------------------------------------------------------------
% FIGURE 3: Control Force Distribution
%--------------------------------------------------------------------------
figure('Position', [100, 100, 900, 600]);

% Create histogram of peak forces
histogram(valid_peak_forces, 30, 'FaceColor', [0.3 0.7 0.3], 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;

% Add line for MR damper capacity
xline(mr_max_force/1000, 'r--', 'LineWidth', 3);

% Add line for mean force
xline(mean(valid_peak_forces), 'b-', 'LineWidth', 2);

xlabel('Peak MR Damper Force (kN)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 13, 'FontWeight', 'bold');
title('Control Force Distribution', 'FontSize', 14, 'FontWeight', 'bold');
legend('Simulations', sprintf('MR Capacity (%.0f kN)', mr_max_force/1000), ...
       sprintf('Mean (%.1f kN)', mean(valid_peak_forces)), 'Location', 'best');
grid on;

% Add statistics text box
force_stats = {sprintf('Mean: %.1f kN', mean(valid_peak_forces));
               sprintf('Std Dev: %.1f kN', std(valid_peak_forces));
               sprintf('Max: %.1f kN', max(valid_peak_forces));
               sprintf('Min: %.1f kN', min(valid_peak_forces));
               sprintf('%% of Capacity: %.1f%%', 100*mean(valid_peak_forces)/(mr_max_force/1000))};
annotation('textbox', [0.6, 0.6, 0.25, 0.18], 'String', force_stats, ...
           'BackgroundColor', 'white', 'EdgeColor', 'black', 'FontSize', 11, ...
           'HorizontalAlignment', 'left');

% Save figure
saveas(gcf, 'Eurocode8_Control_Force_Distribution.png');
fprintf('✓ Saved: Eurocode8_Control_Force_Distribution.png\n');

%--------------------------------------------------------------------------
% FIGURE 4: Robustness Heatmaps (2D histograms of drift vs uncertain parameters)
%--------------------------------------------------------------------------
figure('Position', [100, 100, 1400, 1000]);

% Parameter names for plotting
param_names = {'Mass Scale', 'Stiffness Scale', 'Damping Scale', 'PGA Scale (Earthquake Intensity)'};
param_ranges = {'0.7-1.3', '0.6-1.4', '0.5-1.5', '0.5-1.5'};
param_units = {'', '', '', ''};

for param_idx = 1:4
    subplot(2, 2, param_idx);
    
    % Create 2D histogram (heatmap)
    x_data = valid_params(:, param_idx);
    y_data = valid_peak_drifts;
    
    % Create bins
    n_bins_x = 25;
    n_bins_y = 25;
    
    % Define bin edges based on parameter ranges
    if param_idx == 1  % Mass scale
        x_edges = linspace(0.7, 1.3, n_bins_x+1);
    elseif param_idx == 2  % Stiffness scale
        x_edges = linspace(0.6, 1.4, n_bins_x+1);
    elseif param_idx == 3  % Damping scale
        x_edges = linspace(0.5, 1.5, n_bins_x+1);
    else  % PGA Scale
        x_edges = linspace(0.5, 1.5, n_bins_x+1);
    end
    y_edges = linspace(0, max(valid_peak_drifts)*1.05, n_bins_y+1);
    
    % Compute 2D histogram
    N = histcounts2(x_data, y_data, x_edges, y_edges);
    
    % Plot as heatmap
    imagesc(x_edges(1:end-1) + diff(x_edges(1:2))/2, ...
            y_edges(1:end-1) + diff(y_edges(1:2))/2, N');
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(gca, 'jet');
    
    hold on;
    
    % Add horizontal line for Eurocode 8 drift limit
    yline(driftLimit, 'w--', 'LineWidth', 3);
    
    % Add vertical line for nominal value (1.0)
    xline(1.0, 'g--', 'LineWidth', 2);
    
    xlabel(sprintf('%s %s', param_names{param_idx}, param_ranges{param_idx}), 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Peak Drift (m)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Robustness: Drift vs %s', param_names{param_idx}), 'FontSize', 13, 'FontWeight', 'bold');
    
    % Calculate failure probability in different regions
    above_limit_idx = y_data > driftLimit;
    below_limit_idx = y_data <= driftLimit;
    
    % Calculate correlation
    [corr_coef, p_value] = corrcoef(x_data, y_data);
    
    % Calculate failure rate in nominal region (0.95-1.05)
    nominal_idx = x_data >= 0.95 & x_data <= 1.05;
    if sum(nominal_idx) > 0
        nominal_failure = 100 * sum(y_data(nominal_idx) > driftLimit) / sum(nominal_idx);
    else
        nominal_failure = NaN;
    end
    
    % Add statistics
    text_str = {sprintf('Correlation: ρ = %.3f', corr_coef(1,2));
                sprintf('p-value: %.3e', p_value(1,2));
                sprintf('Overall failure: %.1f%%', 100*(1-success_rate/100));
                sprintf('Nominal failure: %.1f%%', nominal_failure)};
    
    % Position text based on data range
    text_x = x_edges(1) + 0.02*(x_edges(end)-x_edges(1));
    text_y = y_edges(end-5);
    text(text_x, text_y, text_str, 'Color', 'white', 'FontSize', 10, ...
         'BackgroundColor', 'black', 'EdgeColor', 'white', 'FontWeight', 'bold');
    
    % Mark safe and failure regions
    fill([x_edges(1), x_edges(end), x_edges(end), x_edges(1)], ...
         [0, 0, driftLimit, driftLimit], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    fill([x_edges(1), x_edges(end), x_edges(end), x_edges(1)], ...
         [driftLimit, driftLimit, y_edges(end), y_edges(end)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    
    text(x_edges(1)+0.1, driftLimit/2, 'SAFE', 'Color', 'green', 'FontSize', 12, 'FontWeight', 'bold');
    text(x_edges(1)+0.1, driftLimit*1.5, 'FAILURE', 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');
end

sgtitle('Robustness Heatmaps: Effect of Parameter Variations on Eurocode 8 Compliance', ...
        'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'Eurocode8_Robustness_Heatmaps.png');
fprintf('✓ Saved: Eurocode8_Robustness_Heatmaps.png\n');

%% ==================== 9. EUROCODE 8 COMPLIANCE SUMMARY ====================
fprintf('\n========== EUROCODE 8 COMPLIANCE SUMMARY ==========\n');
fprintf('Building: %d-story shear building\n', n);
fprintf('Story height: %.1f m\n', storyHeight);
fprintf('Eurocode 8 drift limit (h/200): %.4f m (%.1f mm)\n', driftLimit, driftLimit*1000);
fprintf('Total Monte Carlo runs: %d\n', Nsim);
fprintf('Successful simulations: %d\n', successful);
fprintf('\n--- Compliance Statistics ---\n');
fprintf('Runs meeting Eurocode 8 limit: %d/%d (%.1f%%)\n', ...
        sum(valid_peak_drifts <= driftLimit), length(valid_peak_drifts), success_rate);
fprintf('Runs exceeding Eurocode 8 limit: %d/%d (%.1f%%)\n', ...
        sum(valid_peak_drifts > driftLimit), length(valid_peak_drifts), 100-success_rate);

fprintf('\n--- Drift Statistics Relative to Eurocode Limit ---\n');
fprintf('Mean drift as %% of limit: %.1f%%\n', 100*mean(valid_peak_drifts)/driftLimit);
fprintf('Max drift as %% of limit: %.1f%%\n', 100*max(valid_peak_drifts)/driftLimit);
fprintf('95th percentile as %% of limit: %.1f%%\n', 100*prctile(valid_peak_drifts,95)/driftLimit);

fprintf('\n--- Control Force Summary ---\n');
fprintf('Mean peak force: %.1f kN (%.1f%% of capacity)\n', mean(valid_peak_forces), 100*mean(valid_peak_forces)/(mr_max_force/1000));
fprintf('Max peak force: %.1f kN (%.1f%% of capacity)\n', max(valid_peak_forces), 100*max(valid_peak_forces)/(mr_max_force/1000));

%% ==================== 10. SAVE RESULTS ====================
% Save only the essential data for the four plots
save('Eurocode8_MC_Results.mat', 'valid_peak_drifts', 'valid_peak_forces', ...
     'valid_floor_drifts', 'valid_params', 'driftLimit', 'mr_max_force', ...
     'n', 'successful', 'success_rate', 'storyHeight');

fprintf('\n✅ Results saved to Eurocode8_MC_Results.mat\n');
fprintf('✅ Four required figures saved as PNG files:\n');
fprintf('   1. Eurocode8_Peak_Drift_Distribution.png\n');
fprintf('   2. Eurocode8_Story_Drift_Profile.png\n');
fprintf('   3. Eurocode8_Control_Force_Distribution.png\n');
fprintf('   4. Eurocode8_Robustness_Heatmaps.png\n');
fprintf('\n✅ Eurocode 8 compliance summary printed above\n');