%% Main Script for RLGA-CO-LQR Control of Building with MR Damper
% Purpose: Orchestrate the setup, run the two-stage RL-GA optimization 
% (which opens the RL Training Monitor), execute the final controlled simulation, 
% and analyze the results.

clear; close all; clc;

disp('================================================================');
disp('   RL-GA Co-LQR Optimization for MR Damper Control (Main Script)');
disp('================================================================');

%% 1. Configuration Parameters
% --- Building & Damper Parameters ---
config.nStories = 10;                    
config.storyMass = 3.5e4;               
config.storyStiffness = 6.5e7;          
config.storyDamping = 6.0e5;            
config.storyHeight = 3.0;               
config.totalMass = config.storyMass * config.nStories;
config.driftLimit = config.storyHeight / 400;
config.mrMaxForce = 200000;             
config.mrMaxCurrent = 2.0;              

% --- Genetic Algorithm parameters (CRITICAL CHANGES) ---
config.gaPopSize = 60;                  % Increased from 20 to 50
config.gaMaxGenerations = 30;           % Increased from 10 to 30
config.gaCrossoverProb = 0.85;          % Slightly increased
config.gaMutationProb = 0.15;           % Increased for more exploration
config.Q_min_log = 4;                   
config.Q_max_log = 10;                  
config.R_min_log = -4;                  
config.R_max_log = 2;                   

% --- Reinforcement Learning parameters (MAJOR CHANGES) ---
config.rlMaxEpisodes = 300;             % Increased from 25 to 100 (MINIMUM)
config.rlMaxSteps = 3000;               % Reduced from 5000 to 2000 (faster episodes)
config.rlLearningRate = 0.1;            % Increased from 0.01 to 0.1 (faster learning)
config.rlGamma = 0.99;                  % Keep high for long-term planning

% --- Seismic excitation parameters ---
config.pga = 0.3;                       
config.duration = 10;                   
config.dt = 0.002; 

% --- Simulink Model Name ---
ModelName = 'WOA_vs_RLGA'; 

fprintf('Building: %d stories, Total mass: %.1f kg\n', ...
        config.nStories, config.totalMass);
fprintf('MR Damper: Max force %.0f kN\n', config.mrMaxForce/1000);

%% 2. System Dynamics and Earthquake Setup
fprintf('2. Generating Earthquake and System Dynamics...\n');

% Generate earthquake data
[ag, t] = generate_nsae_kanai_tajimi_2023Brandao();
config.time = t;
config.earthquake = ag;

% Create TimeSeries object for Simulink
input_excite = timeseries(ag, t);

% Run dynamic model to get system matrices
Dynamic_model; 

% Package state-space matrices into a struct
state_space = struct('A', A, 'Bc', Bc, 'E', E, 'Ms', Ms, 'Ks', Ks, 'Cs', Cs);
building = struct('Ms', Ms, 'Ks', Ks, 'Cs', Cs, 'nStories', config.nStories, ...
                 'storyHeight', config.storyHeight);

% Load necessary configuration into the Base Workspace
assignin('base', 'config', config);
assignin('base', 'state_space', state_space);
assignin('base', 'building', building);
assignin('base', 'input_excite', input_excite);

%% 3. Run RL-GA Optimization
fprintf('3. Starting RL-GA Optimization...\n');

% Use the simplified RLGA optimization
[best_Q, best_R, training_info] = RLGA_Optimization_Simple(config, building, state_space);

% Compute optimal LQR gain
[K_RLGA, P] = computeLQRGain(state_space.A, state_space.Bc, best_Q, best_R);

fprintf('Optimization Complete. Final LQR Gain K computed.\n');
fprintf('Optimal Q factor: %.6e\n', best_Q(1,1) / state_space.Ks(1,1));
fprintf('Optimal R factor: %.6e\n', best_R);

% Store episode rewards for plotting
if i == 1  % Only store for first individual to save memory
    if exist('episode_rewards_temp', 'var')
        training_info.rl_episode_rewards = episode_rewards_temp;
    end
end

%% 4. Plot Training Progress
fprintf('4. Plotting training progress...\n');

% Create separate figures for RL and GA progress
if isfield(training_info, 'rl_episode_rewards')
    figure('Position', [100, 100, 600, 400]);
    
    % Ensure rewards are positive and increasing
    rewards = training_info.rl_episode_rewards;
    
    % Plot actual rewards
    plot(1:length(rewards), rewards, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Episode Reward');
    hold on;
    
    % Add trend line
    if length(rewards) > 5
        x = 1:length(rewards);
        
        % Use moving average for smoother trend
        window_size = min(5, floor(length(rewards)/4));
        smoothed_rewards = movmean(rewards, window_size);
        
        title('RL Training: Trends')
        plot(x, smoothed_rewards, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Trend (Smoothed)');
        
        % Add performance indicators
        max_reward = max(rewards);
        min_reward = min(rewards);
        final_reward = rewards(end);
        
        % % Plot target line for good performance
        % target_level = 0.7 * max_reward;
        % plot([1, length(rewards)], [target_level, target_level], 'g--', ...
        %      'LineWidth', 1, 'DisplayName', 'Good Performance Level');
    end
    
    title('RL Training');
    xlabel('Episode number');
    ylabel('Total Reward');
    grid on;
    legend('Location', 'best');
    
    hold off;
end

%% 5. Export to Simulink and Run Final Simulation
fprintf('5. Exporting to Simulink and running final simulation...\n');

% Export all variables to base workspace for Simulink
assignin('base', 'K_RLGA', K_RLGA);
assignin('base', 'K_optimal', K_RLGA);
assignin('base', 'F_MR_MAX', config.mrMaxForce);
assignin('base', 'F_MR_MIN', 0);

% Save results
save('RLGA_Results.mat', 'K_RLGA', 'best_Q', 'best_R', 'training_info', ...
     'config', 'state_space', 'building');

% Run Simulink simulation
try
    if bdIsLoaded(ModelName)
        % Update gain block if model is open
        set_param([ModelName '/LQR_Gain'], 'Gain', 'K_RLGA');
        fprintf('LQR_Gain block updated with K_RLGA\n');
    end
    
    % Run simulation
    simOut = sim(ModelName, 'SimulationMode', 'normal', 'SrcWorkspace', 'base');
    fprintf('Simulation completed successfully.\n');
    
catch ME
    fprintf('Simulink simulation failed: %s\n', ME.message);
    fprintf('K_RLGA is available in workspace for manual simulation.\n');
end

%% 6. Display Complete Results
fprintf('\n=== RLGA-CO-LQR Optimization Complete ===\n');
fprintf('Optimal LQR Gain Matrix (K_RLGA):\n');
disp(K_RLGA);
fprintf('Matrix size: %dx%d\n', size(K_RLGA));
fprintf('Matrix statistics:\n');
fprintf('  Min: %.6e, Max: %.6e, Mean: %.6e\n', ...
        min(K_RLGA(:)), max(K_RLGA(:)), mean(K_RLGA(:)));

disp('----------------------------------------------------------------');
disp('   Execution Finished. Check figures for training progress.');
disp('----------------------------------------------------------------');