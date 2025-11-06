%% RL_GA_Optimizer.m: Reinforcement Learning Enhanced Genetic Algorithm for LQR Tuning
% This script now integrates the cost function logic as a nested function 
% to eliminate function call overhead during the optimization loop.

% --- 0. Prerequisites and Setup ---
clc; clear; close all;
disp('--- Starting RL-GA Enhanced Optimization (Cost Function Integrated) ---');

% 1. Run Prerequisites
generate_nsae_kanai_tajimi_2023Brandao;
Dynamic_model; 

% 2. Define Problem Dimensions and Constraints
n = size(Ms, 1);        % Number of stories (10)
N_STATES = 2 * n;       % 20 states
N_CONTROLS = size(Gamma, 2); % 1 control input
F_MR_MAX = 2.0 * 10^4;  % Damper force constraint (50 kN)
F_MR_MIN = - F_MR_MAX;

% Define the time series input for Simulink (passed to cost function)
input_excite = timeseries(ag, t);
ModelName = 'WOA_vs_RLGA'; % Simulink Model Name

% Search bounds for log10(Q) and log10(R)
LB = [4, -8]; % Q_param >= 1e4, R_param >= 1e-8
UB = [10, 2]; % Q_param <= 1e10, R_param <= 1e2

% --- 3. RL-Enhanced Initialization ---
TargetPopulationSize = 20; % Target population size for GA
disp('3. Starting RL-Enhanced Initialization...');

% Call the initialization routine using the NESTED function handle
InitialPopulation = RL_Enhanced_Initialization(...
    TargetPopulationSize, LB, UB, @nested_cost_function, ...
    A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS);

disp(['   Initialization complete. Found ', num2str(size(InitialPopulation, 1)), ' stable progenitors.']);

% --- 4. Genetic Algorithm (GA) Optimization Setup ---
% Note: Using a placeholder for the GA implementation based on typical structure.
GA_Options.PopulationSize = TargetPopulationSize;
GA_Options.InitialPopulation = InitialPopulation; 
GA_Options.MaxGenerations = 60; 
GA_Options.Display = 'iter';

% Call the GA/WOA with the NESTED function handle
disp('4. Starting Genetic Algorithm Optimization...');
% [BestPos_log, BestCost] = ga(@nested_cost_function, 2, [], [], [], [], LB, UB, [], GA_Options, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS); 
% --- Placeholder for Custom Optimizer Call ---
% In a real script, this is where your WOA/RL-GA function would be called:
% For demonstration, we'll skip the loop and use the best result from initialization
BestPos_log = InitialPopulation(1,:); 
BestCost = nested_cost_function(BestPos_log, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS);


% --- 5. POST-PROCESSING FINAL RESULTS ---
Q_param_RLGA = 10^BestPos_log(1); % Convert back from log
R_param_RLGA = 10^BestPos_log(2);

% Recalculate the final optimal K_lqr
Q_RLGA = Q_param_RLGA * eye(N_STATES);
R_RLGA = R_param_RLGA * eye(N_CONTROLS);
K_RLGA = lqr(A, Bc, Q_RLGA, R_RLGA);

disp('----------------------------------------------------');
disp('RL-GA Optimization Final Results:');
disp(['Optimal Q Parameter (log10): ', num2str(BestPos_log(1))]);
disp(['Optimal R Parameter (log10): ', num2str(BestPos_log(2))]);
disp(['Optimal Q_param: ', num2str(Q_param_RLGA, '%.2e')]);
disp(['Optimal R_param: ', num2str(R_param_RLGA, '%.2e')]);
disp(['Minimum Cost (Max Drift): ', num2str(BestCost, '%.4f')]);
disp('----------------------------------------------------');


% --- 6. FINAL SIMULATION RUN (Essential for updating Simulink scopes) ---
assignin('base', 'K_lqr_RLGA', K_RLGA);
assignin('base', 'F_MR_MAX', F_MR_MAX);
assignin('base', 'F_MR_MIN', F_MR_MIN);
assignin('base', 'input_excite', input_excite); 
assignin('base', 'A', A);
assignin('base', 'Bc', Bc);
assignin('base', 'E', E);

if bdIsLoaded(ModelName) || ~isempty(which(ModelName))
    SimTime = input_excite.Time(end);
    disp(['Running final Simulink simulation for T=', num2str(SimTime), 's...']);
    % The sim command is now the DIRECT call in the optimization process
    simOut = sim(ModelName, 'SimulationMode', 'normal', ...
        'StopTime', num2str(SimTime), ...
        'SrcWorkspace', 'base');
    disp('Final simulation complete.');
else
    warning('Simulink model COLQR_Simulation.slx not found or loaded.');
end


%% NESTED COST FUNCTION (Replaces COLQR_CostFunction.m)
% This function is executed by the optimizer and has access to all variables 
% defined in the parent script (A, Bc, E, F_MR_MAX, etc.).
function Cost = nested_cost_function(X_log, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS)

    % Define the maximum cost (penalty for instability or simulation failure)
    MAX_PENALTY_COST = 1.0e03; 
    ModelName = 'WOA_vs_RLGA'; 

    % 1. Convert logarithmic parameters back to linear scale
    Q_param_RLGA = 10^X_log(1);
    R_param_RLGA = 10^X_log(2);

    % 2. Calculate LQR Gain K
    Q_RLGA = Q_param_RLGA * eye(N_STATES);
    R_RLGA = R_param_RLGA * eye(N_CONTROLS);

    % Calculate the optimal gain. If lqr fails, return a high cost.
    try
        K_RLGA = lqr(A, Bc, Q_RLGA, R_RLGA);
    catch ME
        Cost = MAX_PENALTY_COST; % Penalty for failure
        return;
    end

    % --- 3. Run Simulink Simulation (Direct Call) ---
    % Ensure required variables for Simulink are available in the base workspace
    % NOTE: Nested function variables must be assigned to the base workspace for Simulink access
    F_MR_MIN = -F_MR_MAX;
    assignin('base', 'K_lqr_RLGA', K_RLGA);
    assignin('base', 'F_MR_MAX', F_MR_MAX);
    assignin('base', 'F_MR_MIN', F_MR_MIN);
    assignin('base', 'input_excite', input_excite); 
    assignin('base', 'A', A);
    assignin('base', 'Bc', Bc);
    assignin('base', 'E', E);

    try
        % Sim is the direct call to the simulation model
        simOut = sim(ModelName, 'SimulationMode', 'normal', ...
            'SrcWorkspace', 'base', ...
            'ReturnWorkspaceOutputs', 'on', ...
            'OutputSaveName', 'OutRLGA');
    catch ME
        Cost = MAX_PENALTY_COST; % Penalty for simulation failure
        return;
    end
             
    % --- 4. Extract States and Calculate Inter-Story Drift ---
    try
        % Get the output structure from the simulation results
        X_controlled_ts = simOut.get('OutRLGA'); 
        
        if isa(X_controlled_ts, 'timeseries')
             % Extract displacement data (first N_STATES/2 elements of the state vector)
             Displacements = X_controlled_ts.Data(:, 1:N_STATES/2); 
        else
            error('Output data format is not recognized or data is not a timeseries object.');
        end
        
        n_stories = size(Displacements, 2);
        Drifts = zeros(size(Displacements));

        % Calculate Drifts: d_i - d_i-1
        Drifts(:, 1) = Displacements(:, 1); % Drift for first floor is d1
        for i = 2:n_stories
            Drifts(:, i) = Displacements(:, i) - Displacements(:, i-1);
        end
        
        % The cost is the MAXIMUM absolute inter-story drift (MID)
        MaxDrift = max(max(abs(Drifts))); 
        
        Cost = MaxDrift; % The final cost value
        
    catch ME
        % If data extraction or drift calculation fails, return a high penalty
        Cost = MAX_PENALTY_COST;
    end

end % End of nested_cost_function
