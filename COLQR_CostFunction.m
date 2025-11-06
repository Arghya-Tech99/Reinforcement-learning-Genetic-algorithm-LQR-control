function Cost = COLQR_CostFunction(X_log, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS)
% COLQR_CostFunction: Calculates the cost (Max Inter-Story Drift) for a given Q and R.

    % Define the maximum cost (penalty for instability or simulation failure)
    MAX_PENALTY_COST = 1.0e10; 

    % 1. Convert logarithmic parameters back to linear scale
    Q_param_WOA = 10^X_log(1);
    R_param_WOA = 10^X_log(2);

    % 2. Calculate LQR Gain K
    Q_WOA = Q_param_WOA * eye(N_STATES);
    R_WOA = R_param_WOA * eye(N_CONTROLS);

    % Calculate the optimal gain. If lqr fails, return a high cost.
    try
        K_WOA = lqr(A, Bc, Q_WOA, R_WOA);
    catch ME
        Cost = MAX_PENALTY_COST; % Penalty for failure (e.g., instability or numerical issue)
        warning('COLQR_CostFunction: LQR gain calculation failed. Q=1e%f, R=1e%f. Error: %s', X_log(1), X_log(2), ME.message);
        return;
    end

    % --- 3. Run Simulink Simulation ---
    ModelName = 'COLQR_WOA'; 
    
    % Ensure required variables for Simulink are available in the base workspace
    F_MR_MIN = 0.0;
    assignin('base', 'K_WOA', K_WOA);
    assignin('base', 'F_MR_MAX', F_MR_MAX);
    assignin('base', 'F_MR_MIN', F_MR_MIN);
    assignin('base', 'input_excite', input_excite); 
    
    % Run simulation
    try
        % The simulation must use the 'base' workspace for parameters defined above
        sim(ModelName, 'SaveTime', 'off', 'SaveState', 'off', 'SaveOutput', 'off', ... 
             'LoadExternalInput', 'off', 'SrcWorkspace', 'base');
    catch ME
        warning('COLQR_CostFunction: Simulink simulation failed. Penalty cost applied. Error: %s', ME.message);
        Cost = MAX_PENALTY_COST;
        return;
    end
             
    % --- 4. Extract States and Calculate Inter-Story Drift ---
    try
        % 4.1. Check if the output variable 'Output' exists in the base workspace
        if ~evalin('base', 'exist(''Output'', ''var'')')
            error('Simulink output variable ''Output'' was not created in the base workspace. Please check the To Workspace block settings in COLQR_Simulation.');
        end

        % 4.2. Load the controlled state vector X_controlled = [d1...d10, v1...v10]
        X_controlled_ts = evalin('base', 'Output');
        
        % Assuming the 'To Workspace' block is set to save as a timeseries
        if isa(X_controlled_ts, 'timeseries')
             % Extract displacement data (first N_STATES/2 elements of the state vector)
             % N_STATES/2 is the number of stories (displacements)
             Displacements = X_controlled_ts.Data(:, 1:N_STATES/2); 
        else
            % Fallback/Error for unexpected format
            error('Output data format is not recognized or data is not a timeseries object. Please set the To Workspace block format to "Time Series".');
        end
        
        n_stories = size(Displacements, 2);
        
        % Initialize Inter-Story Drift matrix
        Drifts = zeros(size(Displacements));

        % Drift for the first floor is d1 (since x0 = 0)
        Drifts(:, 1) = Displacements(:, 1);
        
        % Drifts for floors 2 through n_stories (di - d_i-1)
        for i = 2:n_stories
            Drifts(:, i) = Displacements(:, i) - Displacements(:, i-1);
        end
        
        % The cost is the MAXIMUM absolute inter-story drift (MID) across all floors and all time steps
        MaxDrift = max(max(abs(Drifts))); 
        
        Cost = MaxDrift; % The final cost value
        
        % Clean up the base workspace variable (good practice)
        evalin('base', 'clear Output');

    catch ME
        % If data extraction or drift calculation fails, return a high penalty
        warning('COLQR_CostFunction: Data extraction or calculation failed. Penalty cost applied. Error: %s', ME.message);
        Cost = MAX_PENALTY_COST;
        % Clean up if the simulation somehow created the variable
        if evalin('base', 'exist(''Output'', ''var'')')
             evalin('base', 'clear Output');
        end
        return;
    end
end