%% MONTE CARLO SIMULATION FOR RL-GA OPTIMIZED CO-LQR CONTROLLER
% Using direct numerical integration (bypassing Simulink)

clear; clc; close all;

%% ==================== 1. LOAD OPTIMIZED CONTROLLER ====================
fprintf('Loading optimized controller from Optimal_QR.mat...\n');
if ~exist('Optimal_QR.mat', 'file')
    error('Optimal_QR.mat not found. Run main_RLGA_COLQR.m first.');
end
load('Optimal_QR.mat', 'best_Q', 'best_R');

%% ==================== 2. SYSTEM PARAMETERS ====================
% Building parameters
n = 10;                      % number of stories
m_nom = 3.5e4;               % nominal story mass (kg)
k_nom = 6.5e7;               % nominal story stiffness (N/m)
c_nom = 6.0e5;               % nominal story damping (N·s/m)
storyHeight = 3.0;            % story height (m)
driftLimit = storyHeight / 400;   % design drift limit (m)

% MR damper parameters
mr_max_force = 200000;        % maximum MR damper force (N)

% Time parameters
dt = 0.002;                   % time step (s)
T_total = 10;                  % total simulation time (s)
t = (0:dt:T_total)';
nSteps = length(t);

%% ==================== 3. UNCERTAINTY DEFINITION ====================
fprintf('\nDefining uncertainty distributions...\n');

% Function to sample uncertainties
sample_uncertainties = @() struct(...
    'mass_scale',      max(0.7, 1 + 0.10*randn), ...      % ±10% truncated normal
    'stiffness_scale', max(0.6, 1 + 0.15*randn), ...      % ±15%
    'damping_scale',   max(0.5, 1 + 0.20*randn), ...      % ±20%
    'PGA_scale',       0.5 + 1.0*rand, ...                % uniform [0.5, 1.5]
    'seed',            randi(1e6) ...                     % random seed
);

%% ==================== 4. MONTE CARLO SETTINGS ====================
Nsim = 6;                   % Number of Monte Carlo runs for final = 300
results = NaN(Nsim, 5);        % [peak_drift, rms_drift, peak_force, rms_force, max_interstory_drift]
successful = 0;

fprintf('\nStarting Monte Carlo simulations (%d runs)...\n', Nsim);

%% ==================== 5. MAIN MONTE CARLO LOOP ====================
for sim = 1:Nsim
    fprintf('Run %d/%d: ', sim, Nsim);
    
    try
        % --- Sample uncertainties ---
        params = sample_uncertainties();
        
        % --- Generate earthquake with given seed ---
        rng(params.seed);
        [ag, ~] = generate_nsae_kanai_tajimi_2023Brandao();  % returns acceleration and time
        ag = params.PGA_scale * ag;                           % scale PGA
        ag = interp1(linspace(0, T_total, length(ag)), ag, t)'; % resample to our time vector
        
        % --- Scale structural parameters ---
        m = m_nom * params.mass_scale;
        k = k_nom * params.stiffness_scale;
        c = c_nom * params.damping_scale;
        
        % --- Build system matrices with scaled parameters ---
        % Mass matrix
        Ms = m * eye(n);
        
        % Stiffness matrix (shear building)
        main_diag_K = [2*k * ones(n-1, 1); k];
        off_diag_K = -k * ones(n-1, 1);
        Ks = diag(main_diag_K) + diag(off_diag_K, 1) + diag(off_diag_K, -1);
        
        % Damping matrix
        main_diag_C = [2*c * ones(n-1, 1); c];
        off_diag_C = -c * ones(n-1, 1);
        Cs = diag(main_diag_C) + diag(off_diag_C, 1) + diag(off_diag_C, -1);
        
        % State-space matrices
        A = [zeros(n), eye(n); -Ms\Ks, -Ms\Cs];
        B = [zeros(n,1); -Ms \ [1; zeros(n-1,1)]];   % control input matrix
        E = [zeros(n,1); -ones(n,1)];                 % earthquake input matrix
        
        % --- Compute LQR gain using optimal Q and R ---
        K_lqr = lqr(A, B, best_Q, best_R);
        
        % --- Initialize state vectors ---
        x = zeros(2*n, 1);        % state vector [displacements; velocities]
        u_control = 0;             % control force
        
        % Storage for results
        displacements = zeros(nSteps, n);
        velocities = zeros(nSteps, n);
        control_force = zeros(nSteps, 1);
        
        % --- Numerical integration (Newmark-beta method) ---
        fprintf('Integrating... ');
        
        for i = 1:nSteps
            % Store current state
            displacements(i, :) = x(1:n)';
            velocities(i, :) = x(n+1:end)';
            control_force(i) = u_control;
            
            % Get earthquake acceleration at current time
            xg_ddot = ag(i);
            
            % Compute LQR optimal force
            f_optimal = -K_lqr * x;
            
            % Apply MR damper constraints (clipped optimal)
            if abs(f_optimal) <= mr_max_force
                u_control = f_optimal;
            else
                u_control = sign(f_optimal) * mr_max_force;
            end
            
            % Compute state derivative
            x_dot = A * x + B * u_control + E * xg_ddot;
            
            % Simple Euler integration (use smaller dt if needed)
            x = x + dt * x_dot;
        end
        
        % --- Compute story drifts (relative displacements between floors) ---
        story_drifts = zeros(nSteps, n);
        story_drifts(:, 1) = displacements(:, 1);  % first floor drift = its displacement
        for j = 2:n
            story_drifts(:, j) = displacements(:, j) - displacements(:, j-1);
        end
        
        % --- Performance metrics ---
        peak_drift = max(abs(story_drifts(:)));           % maximum absolute drift
        rms_drift = sqrt(mean(story_drifts(:).^2));       % RMS drift
        peak_force = max(abs(control_force));              % peak control force
        rms_force = sqrt(mean(control_force.^2));          % RMS control force
        max_interstory = max(max(abs(story_drifts)));      % max inter-story drift
        
        % Store results
        results(sim, :) = [peak_drift, rms_drift, peak_force, rms_force, max_interstory];
        successful = successful + 1;
        
        fprintf('✓ peak drift = %.4f m\n', peak_drift);
        
    catch ME
        fprintf('✗ failed: %s\n', ME.message);
    end
end

%% ==================== 6. RESULTS ANALYSIS ====================
fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Successful runs: %d/%d (%.1f%%)\n', successful, Nsim, 100*successful/Nsim);

if successful == 0
    error('No successful simulations to analyze.');
end

% Remove any NaN rows
valid_results = results(~any(isnan(results), 2), :);
peak_drifts = valid_results(:,1);
rms_drifts = valid_results(:,2);
peak_forces = valid_results(:,3);
rms_forces = valid_results(:,4);
max_interstory = valid_results(:,5);

%% ==================== 7. STATISTICS ====================
fprintf('\n===== MONTE CARLO STATISTICS =====\n');
fprintf('Peak Drift (m):\n');
fprintf('  Mean: %.6f\n', mean(peak_drifts));
fprintf('  Std:  %.6f\n', std(peak_drifts));
fprintf('  Min:  %.6f\n', min(peak_drifts));
fprintf('  Max:  %.6f\n', max(peak_drifts));
fprintf('  95%% CI: [%.6f, %.6f]\n', prctile(peak_drifts,2.5), prctile(peak_drifts,97.5));
fprintf('  Exceedance probability (drift > %.4f m): %.1f%%\n', ...
    driftLimit, 100 * sum(peak_drifts > driftLimit) / length(peak_drifts));

fprintf('\nRMS Drift (m):\n');
fprintf('  Mean: %.6f\n', mean(rms_drifts));
fprintf('  Std:  %.6f\n', std(rms_drifts));

fprintf('\nPeak Control Force (kN):\n');
fprintf('  Mean: %.2f\n', mean(peak_forces)/1000);
fprintf('  Std:  %.2f\n', std(peak_forces)/1000);

fprintf('\nRMS Control Force (kN):\n');
fprintf('  Mean: %.2f\n', mean(rms_forces)/1000);
fprintf('  Std:  %.2f\n', std(rms_forces)/1000);

%% ==================== 8. PLOTTING ====================
figure('Position', [50, 50, 1400, 900]);

% 1. Peak Drift Histogram
subplot(2,3,1);
histogram(peak_drifts, 25, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k');
hold on;
xline(driftLimit, 'r--', 'LineWidth', 2);
xlabel('Peak Drift (m)'); ylabel('Frequency');
title('Peak Story Drift Distribution');
legend('Simulations', 'Drift Limit', 'Location', 'best');
grid on;

% 2. RMS Drift Histogram
subplot(2,3,2);
histogram(rms_drifts, 25, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k');
xlabel('RMS Drift (m)'); ylabel('Frequency');
title('RMS Story Drift Distribution');
grid on;

% 3. Peak Control Force
subplot(2,3,3);
histogram(peak_forces/1000, 25, 'FaceColor', [0.3 0.7 0.3], 'EdgeColor', 'k');
xlabel('Peak Force (kN)'); ylabel('Frequency');
title('Peak MR Damper Force');
grid on;

% 4. RMS Control Force
subplot(2,3,4);
histogram(rms_forces/1000, 25, 'FaceColor', [0.9 0.6 0.1], 'EdgeColor', 'k');
xlabel('RMS Force (kN)'); ylabel('Frequency');
title('RMS MR Damper Force');
grid on;

% 5. Maximum Inter-story Drift
subplot(2,3,5);
histogram(max_interstory, 25, 'FaceColor', [0.5 0.2 0.7], 'EdgeColor', 'k');
xlabel('Max Inter-story Drift (m)'); ylabel('Frequency');
title('Maximum Inter-story Drift');
grid on;

% 6. Cumulative Distribution of Peak Drift
subplot(2,3,6);
cdfplot(peak_drifts);
hold on;
xline(driftLimit, 'r--', 'LineWidth', 2);
xlabel('Peak Drift (m)'); ylabel('Cumulative Probability');
title('CDF of Peak Drift');
legend('Simulated', 'Drift Limit', 'Location', 'best');
grid on;

sgtitle(sprintf('Monte Carlo Results (N = %d successful runs)', successful));

% 7. Additional: Time history of a representative run
figure('Position', [50, 50, 1200, 800]);
 representative_run = randi(successful);
 
subplot(2,2,1);
plot(t, displacements(:, 1:5:end));  % plot every 5th floor for clarity
xlabel('Time (s)'); ylabel('Displacement (m)');
title('Floor Displacements (Representative Run)');
legend(arrayfun(@(x) sprintf('Floor %d', x), 1:5:n, 'UniformOutput', false));
grid on;

subplot(2,2,2);
plot(t, story_drifts(:, 1:5:end));
xlabel('Time (s)'); ylabel('Drift (m)');
title('Story Drifts (Representative Run)');
legend(arrayfun(@(x) sprintf('Story %d', x), 1:5:n, 'UniformOutput', false));
grid on;

subplot(2,2,3);
plot(t, control_force/1000);
xlabel('Time (s)'); ylabel('Force (kN)');
title('MR Damper Control Force');
grid on;

subplot(2,2,4);
plot(t, ag);
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
title('Earthquake Ground Acceleration');
grid on;

%% ==================== 9. SAVE RESULTS ====================
save('MC_Results.mat', 'results', 'valid_results', 'params', 'driftLimit', 't');
fprintf('\n✓ Results saved to MC_Results.mat\n');

%% ==================== 10. SUMMARY REPORT ====================
fprintf('\n========== SUMMARY REPORT ==========\n');
fprintf('Monte Carlo Simulation Complete\n');
fprintf('Controller: RL-GA Optimized CO-LQR\n');
fprintf('Number of stories: %d\n', n);
fprintf('Drift limit: %.4f m\n', driftLimit);
fprintf('Success rate: %.1f%% (%d/%d)\n', 100*successful/Nsim, successful, Nsim);
fprintf('\nKey Findings:\n');
fprintf('- The controller maintains peak drift below %.4f m in %.1f%% of cases\n', ...
    driftLimit, 100 * (1 - sum(peak_drifts > driftLimit)/length(peak_drifts)));
fprintf('- Mean peak drift: %.4f m (%.1f%% of limit)\n', ...
    mean(peak_drifts), 100*mean(peak_drifts)/driftLimit);
fprintf('- Worst-case peak drift: %.4f m (%.1f%% of limit)\n', ...
    max(peak_drifts), 100*max(peak_drifts)/driftLimit);