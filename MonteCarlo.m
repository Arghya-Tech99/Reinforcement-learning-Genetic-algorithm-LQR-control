%% MONTE CARLO SIMULATION FOR RL-GA OPTIMIZED CO-LQR CONTROLLER
% Fixed version with correct floor drift calculations

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
driftLimit = storyHeight / 400;   % design drift limit = 0.0075 m

% MR damper parameters
mr_max_force = 200000;        % maximum MR damper force (N)

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

% Function to sample uncertainties
sample_uncertainties = @() struct(...
    'mass_scale',      1 + 0.10*randn, ...                % ±10% normal
    'stiffness_scale', 1 + 0.15*randn, ...                % ±15% normal
    'damping_scale',   1 + 0.20*randn, ...                % ±20% normal
    'PGA_scale',       0.5 + 1.0*rand, ...                % uniform [0.5, 1.5]
    'seed',            randi(1e6) ...                     % random seed
);

%% ==================== 5. MONTE CARLO SETTINGS ====================
Nsim = 6;                   % Number of Monte Carlo runs
results = NaN(Nsim, 8);        % [peak_drift, rms_drift, peak_force, rms_force, 
                               %  max_interstory, max_floor_disp, max_floor_vel, max_accel]
successful = 0;

% Storage for all runs data
all_runs_displacements = cell(Nsim, 1);
all_runs_velocities = cell(Nsim, 1);
all_runs_control_force = cell(Nsim, 1);
all_runs_params = cell(Nsim, 1);

fprintf('\nStarting Monte Carlo simulations (%d runs)...\n', Nsim);

%% ==================== 6. MAIN MONTE CARLO LOOP ====================
for sim = 1:Nsim
    fprintf('Run %d/%d: ', sim, Nsim);
    
    try
        % --- Sample uncertainties ---
        params = sample_uncertainties();
        
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
        velocities = zeros(nSteps, n);
        accelerations = zeros(nSteps, n);
        control_force = zeros(nSteps, 1);
        
        % --- Numerical integration ---
        for i = 1:nSteps
            % Store current state
            displacements(i, :) = x(1:n)';
            velocities(i, :) = x(n+1:end)';
            control_force(i) = u_control;
            
            % Get earthquake acceleration
            xg_ddot = ag(i);
            
            % Compute state derivative for acceleration
            x_dot_current = A * x + B * u_control + E * xg_ddot;
            accelerations(i, :) = x_dot_current(n+1:end)';
            
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
                % First floor drift = its absolute displacement
                story_drifts(:, floor) = displacements(:, floor);
            else
                % For floors 2-10: drift = displacement relative to floor below
                story_drifts(:, floor) = displacements(:, floor) - displacements(:, floor-1);
            end
        end
        
        % Also compute inter-story drifts (for verification)
        interstory_drifts = zeros(nSteps, n-1);
        for floor = 1:n-1
            interstory_drifts(:, floor) = displacements(:, floor+1) - displacements(:, floor);
        end
        
        % --- Performance metrics ---
        peak_drift = max(abs(story_drifts(:)));           % maximum absolute drift
        rms_drift = sqrt(mean(story_drifts(:).^2));       % RMS drift
        peak_force = max(abs(control_force));              % peak control force
        rms_force = sqrt(mean(control_force.^2));          % RMS control force
        max_interstory = max(max(abs(interstory_drifts))); % max inter-story drift
        max_floor_disp = max(max(abs(displacements)));     % max floor displacement
        max_floor_vel = max(max(abs(velocities)));         % max floor velocity
        max_accel = max(max(abs(accelerations)));          % max floor acceleration
        
        % Store results
        results(sim, :) = [peak_drift, rms_drift, peak_force, rms_force, ...
                           max_interstory, max_floor_disp, max_floor_vel, max_accel];
        
        % Store full data for this run
        all_runs_displacements{sim} = displacements;
        all_runs_velocities{sim} = velocities;
        all_runs_control_force{sim} = control_force;
        all_runs_params{sim} = params;
        
        successful = successful + 1;
        
        % Status update
        drift_ratio = peak_drift / driftLimit * 100;
        status = '✓';
        if peak_drift > driftLimit
            status = '⚠';
        end
        fprintf('%s peak drift = %.4f m (%.1f%% of limit)\n', status, peak_drift, drift_ratio);
        
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

% Remove any NaN rows
valid_idx = ~any(isnan(results), 2);
valid_results = results(valid_idx, :);
peak_drifts = valid_results(:,1);
rms_drifts = valid_results(:,2);
peak_forces = valid_results(:,3) / 1000;
rms_forces = valid_results(:,4) / 1000;
max_interstory = valid_results(:,5);
max_floor_disp = valid_results(:,6);
max_floor_vel = valid_results(:,7);
max_accel = valid_results(:,8);

%% ==================== 8. STATISTICS ====================
fprintf('\n===== MONTE CARLO STATISTICS =====\n');
fprintf('Peak Drift (m) - Target: < %.4f m\n', driftLimit);
fprintf('  Mean: %.6f (%.1f%% of limit)\n', mean(peak_drifts), 100*mean(peak_drifts)/driftLimit);
fprintf('  Std:  %.6f\n', std(peak_drifts));
fprintf('  Min:  %.6f (%.1f%% of limit)\n', min(peak_drifts), 100*min(peak_drifts)/driftLimit);
fprintf('  Max:  %.6f (%.1f%% of limit)\n', max(peak_drifts), 100*max(peak_drifts)/driftLimit);
fprintf('  95%% CI: [%.6f, %.6f]\n', prctile(peak_drifts,2.5), prctile(peak_drifts,97.5));

% Success rate
success_rate = 100 * sum(peak_drifts <= driftLimit) / length(peak_drifts);
fprintf('\n✓ Controller SUCCESS rate: %.1f%% (drift < %.4f m)\n', success_rate, driftLimit);

%% ==================== 9. PLOTTING ====================
% Select the best, worst, and median runs for detailed plotting
[~, best_idx] = min(peak_drifts);
[~, worst_idx] = max(peak_drifts);
[~, median_idx] = min(abs(peak_drifts - median(peak_drifts)));

% Convert indices to actual run numbers
best_run = find(valid_idx, best_idx, 'first');
worst_run = find(valid_idx, worst_idx, 'first');
median_run = find(valid_idx, median_idx, 'first');

% Figure 1: All 10 Floor Drifts for Best, Worst, and Median Runs
figure('Position', [50, 50, 1800, 1000]);

for run_type = 1:3
    switch run_type
        case 1
            run_num = best_run;
            title_str = 'Best Performance';
            color = [0.2 0.8 0.2];
        case 2
            run_num = median_run;
            title_str = 'Median Performance';
            color = [0.2 0.2 0.8];
        case 3
            run_num = worst_run;
            title_str = 'Worst Performance';
            color = [0.8 0.2 0.2];
    end
    
    displacements = all_runs_displacements{run_num};
    
    % Compute drifts correctly for this run
    story_drifts_plot = zeros(nSteps, n);
    for floor = 1:n
        if floor == 1
            story_drifts_plot(:, floor) = displacements(:, floor);
        else
            story_drifts_plot(:, floor) = displacements(:, floor) - displacements(:, floor-1);
        end
    end
    
    for floor = 1:10
        subplot(3,10,(run_type-1)*10 + floor);
        plot(t, story_drifts_plot(:, floor), 'Color', color, 'LineWidth', 1);
        hold on;
        yline(driftLimit, 'r--', 'LineWidth', 1);
        yline(-driftLimit, 'r--', 'LineWidth', 1);
        
        if run_type == 3
            xlabel('Time (s)');
        end
        if floor == 1
            ylabel(sprintf('%s\nDrift (m)', title_str));
        end
        title(sprintf('Floor %d', floor));
        grid on;
        xlim([0 T_total]);
        ylim([-driftLimit*1.5, driftLimit*1.5]);
    end
end
sgtitle('Floor Drift Time Histories for Best, Median, and Worst Cases');

% Figure 2: Statistical Distribution of Drifts Across All Floors
figure('Position', [50, 50, 1400, 800]);

% Compute max drift for each floor across all successful runs
floor_max_drifts = zeros(successful, n);
floor_rms_drifts = zeros(successful, n);

for run_idx = 1:successful
    run_num = find(valid_idx, run_idx, 'first');
    displacements = all_runs_displacements{run_num};
    
    % Compute drifts correctly
    for floor = 1:n
        if floor == 1
            floor_drift = displacements(:, floor);
        else
            floor_drift = displacements(:, floor) - displacements(:, floor-1);
        end
        floor_max_drifts(run_idx, floor) = max(abs(floor_drift));
        floor_rms_drifts(run_idx, floor) = sqrt(mean(floor_drift.^2));
    end
end

% Box plot of max drifts per floor
subplot(2,2,1);
boxplot(floor_max_drifts, 'Labels', 1:n);
hold on;
yline(driftLimit, 'r--', 'LineWidth', 2);
xlabel('Floor Number'); ylabel('Max Drift (m)');
title('Distribution of Maximum Drift per Floor');
grid on;

% Box plot of RMS drifts per floor
subplot(2,2,2);
boxplot(floor_rms_drifts, 'Labels', 1:n);
xlabel('Floor Number'); ylabel('RMS Drift (m)');
title('Distribution of RMS Drift per Floor');
grid on;

subplot(2,2,3);
% Mean drift profile
mean_max_drifts = mean(floor_max_drifts);
std_max_drifts = std(floor_max_drifts);
errorbar(1:n, mean_max_drifts, std_max_drifts, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(driftLimit, 'r--', 'LineWidth', 2);
xlabel('Floor Number'); ylabel('Max Drift (m)');
title('Mean Maximum Drift Profile with ±1σ');
grid on;

subplot(2,2,4);
% Heatmap of drifts
imagesc(1:n, 1:successful, floor_max_drifts);
colorbar;
xlabel('Floor Number'); ylabel('Run Number');
title('Max Drift per Floor Across All Runs');
colormap('hot');

sgtitle('Statistical Analysis of Floor Drifts');

% Figure 3: Main Monte Carlo Statistics
figure('Position', [50, 50, 1600, 900]);

% Peak Drift Histogram
subplot(2,4,1);
histogram(peak_drifts, 25, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
hold on;
xline(driftLimit, 'r--', 'LineWidth', 2);
xlabel('Peak Drift (m)'); ylabel('Frequency');
title(sprintf('Peak Drift Distribution\nSuccess Rate: %.1f%%', success_rate));
legend('Simulations', 'Drift Limit', 'Location', 'best');
grid on;

% RMS Drift Histogram
subplot(2,4,2);
histogram(rms_drifts, 25, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
xlabel('RMS Drift (m)'); ylabel('Frequency');
title('RMS Drift Distribution');
grid on;

% Peak Control Force
subplot(2,4,3);
histogram(peak_forces, 25, 'FaceColor', [0.3 0.7 0.3], 'EdgeColor', 'k');
xlabel('Peak Force (kN)'); ylabel('Frequency');
title(sprintf('Peak MR Damper Force\nMean: %.1f kN', mean(peak_forces)));
grid on;

% RMS Control Force
subplot(2,4,4);
histogram(rms_forces, 25, 'FaceColor', [0.9 0.6 0.1], 'EdgeColor', 'k');
xlabel('RMS Force (kN)'); ylabel('Frequency');
title(sprintf('RMS Control Force\nMean: %.1f kN', mean(rms_forces)));
grid on;

% Maximum Inter-story Drift
subplot(2,4,5);
histogram(max_interstory, 25, 'FaceColor', [0.5 0.2 0.7], 'EdgeColor', 'k');
xlabel('Max Inter-story Drift (m)'); ylabel('Frequency');
title('Maximum Inter-story Drift');
grid on;

% CDF of Peak Drift
subplot(2,4,6);
cdfplot(peak_drifts);
hold on;
xline(driftLimit, 'r--', 'LineWidth', 2);
yline(0.95, 'g--', 'LineWidth', 1.5);
xlabel('Peak Drift (m)'); ylabel('Cumulative Probability');
title('CDF of Peak Drift');
legend('Simulated', 'Drift Limit', '95% Threshold', 'Location', 'best');
grid on;

% Scatter: Peak Drift vs Peak Force
subplot(2,4,7);
scatter(peak_drifts, peak_forces, 50, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Peak Drift (m)'); ylabel('Peak Force (kN)');
title('Peak Drift vs Peak Force');
grid on;

% Box plot summary
subplot(2,4,8);
boxplot(peak_drifts);
ylabel('Peak Drift (m)');
title('Distribution Summary');
grid on;

sgtitle('Monte Carlo Simulation Results: RL-GA Optimized CO-LQR Controller');

% Figure 4: MR Damper Force for Best, Worst, and Median Cases
figure('Position', [50, 50, 1400, 400]);

cases = {best_run, median_run, worst_run};
case_names = {'Best Case', 'Median Case', 'Worst Case'};
colors = {[0.2 0.8 0.2], [0.2 0.2 0.8], [0.8 0.2 0.2]};

for i = 1:3
    subplot(1,3,i);
    force = all_runs_control_force{cases{i}} / 1000;
    plot(t, force, 'Color', colors{i}, 'LineWidth', 1.5);
    hold on;
    yline(mr_max_force/1000, 'k--', 'LineWidth', 1);
    yline(-mr_max_force/1000, 'k--', 'LineWidth', 1);
    xlabel('Time (s)'); ylabel('Force (kN)');
    title(sprintf('MR Damper Force - %s\nPeak: %.1f kN', case_names{i}, max(abs(force))));
    grid on;
    xlim([0 T_total]);
end
sgtitle('MR Damper Control Force Time Histories');

% Figure 5: Floor Displacement Profiles at Peak Response
figure('Position', [50, 50, 1200, 400]);

for i = 1:3
    subplot(1,3,i);
    run_num = cases{i};
    displacements = all_runs_displacements{run_num};
    
    % Find time of peak top floor displacement
    [~, peak_time_idx] = max(abs(displacements(:, end)));
    
    % Plot displacement profile at peak
    plot(displacements(peak_time_idx, :), 1:n, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Displacement (m)'); ylabel('Floor Number');
    title(sprintf('Displacement Profile at Peak - %s', case_names{i}));
    grid on;
    set(gca, 'YDir', 'reverse');
end
sgtitle('Building Displacement Profiles at Peak Response');

%% ==================== 10. EXPORT RESULTS ====================
save('MC_Results_Complete.mat', 'results', 'valid_results', 'all_runs_displacements', ...
     'all_runs_velocities', 'all_runs_control_force', 'all_runs_params', ...
     'driftLimit', 't', 'success_rate', 'floor_max_drifts', 'floor_rms_drifts');

fprintf('\n✅ Complete results saved to MC_Results_Complete.mat\n');
fprintf('✅ All floor drift plots generated successfully\n');