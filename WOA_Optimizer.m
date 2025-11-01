%% WOA_Optimizer.m: Whale Optimization Algorithm for COLQR Tuning

% Implements the entire optimization process:
% 1. Runs prerequisites (NSAE and Dynamic Model).
% 2. Executes WOA to find optimal Q and R parameters.
% 3. Calculates the final optimal K_lqr.
% 4. Prepares the time-series input required by the Simulink model.

disp('--- Starting Whale Optimization Algorithm for COLQR Tuning ---');

%%  1. Run NSAE and Dynamic Model Setup
% Ensures system matrices (A, Bc, E) and seismic input (ag, t) are available
generate_nsae_kanai_tajimi_2023Brandao;
Dynamic_model; 

%% 2. Define Problem Dimensions and Constraints
n = size(Ms, 1);        % Number of stories (10)
N_STATES = 2 * n;       % 20 states
N_CONTROLS = size(Gamma, 2); % 1 control input
F_MR_MAX = 5.0 * 10^4;  % Damper force constraint (50 kN)
F_MR_MIN = -F_MR_MAX;   % Symmetry for the damper (can push and pull)

input_excite = timeseries(ag, t); % The time series input for Simulink

%% 3. WOA PARAMETERS 
WOA_Params.PopulationSize = 10;   % Number of search agents (whales) 
WOA_Params.MaxIterations = 20;    % Maximum number of optimization cycles 
WOA_Params.Dim = 2;               % Dimension of the search space (log10(Q_param), log10(R_param))

%% 4. DEFINING SEARCH BOUNDS
% Bounds: [log10(Q_param)_min, log10(R_param)_min] to [log10(Q_param)_max, log10(R_param)_max]
WOA_Params.LowerBound = [4, -4]; % Q_param >= 1e4, R_param >= 1e-4
WOA_Params.UpperBound = [10, 2]; % Q_param <= 1e10, R_param <= 1e2

disp(['WOA initialized: PopSize=', num2str(WOA_Params.PopulationSize), ...
      ', Iterations=', num2str(WOA_Params.MaxIterations)]);

%% 5. LAUNCH WOA
% COLQR_CostFunction is passed as a function handle.
[BestPos_log, BestCost, ~] = WhaleOptimizationAlgorithm(@COLQR_CostFunction, WOA_Params);

%% 6. POST-PROCESSING RESULTS & FINAL LQR GAIN
Q_opt_param = 10^BestPos_log(1); % Convert back from log
R_opt_param = 10^BestPos_log(2);

% Recalculate the final optimal K_lqr using the best found parameters
Q_opt = Q_opt_param * eye(N_STATES);
R_opt = R_opt_param * eye(N_CONTROLS);
K_lqr = lqr(A, Bc, Q_opt, R_opt);

disp('----------------------------------------------------');
disp('WOA Optimization Final Results:');
disp(['Optimal Q Parameter (log10): ', num2str(BestPos_log(1))]);
disp(['Optimal R Parameter (log10): ', num2str(BestPos_log(2))]);
disp(['Optimal Q_param: ', num2str(Q_opt_param, '%.2e')]);
disp(['Optimal R_param: ', num2str(R_opt_param, '%.2e')]);
disp(['Minimum Cost (Max Drift): ', num2str(BestCost, '%.4f')]);
disp('Optimal K_lqr, F_MR_MAX/MIN, and input_excite ready for Simulink.');
disp('----------------------------------------------------');

% Clean up temporary variables
clear WOA_Params BestPos_log BestCost n N_STATES N_CONTROLS Q_opt R_opt;