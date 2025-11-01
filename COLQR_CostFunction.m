function Cost = COLQR_CostFunction(X_log, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS)
% COLQR_CostFunction: Calculates the cost (Max Inter-Story Drift) for a given Q and R.

    % 1. Convert logarithmic parameters back to linear scale
    Q_param = 10^X_log(1);
    R_param = 10^X_log(2);

    % 2. Calculate LQR Gain K
    Q_opt = Q_param * eye(N_STATES);
    R_opt = R_param * eye(N_CONTROLS);

    % Calculate the optimal gain. If lqr fails, return a high cost.
    try
        K_lqr = lqr(A, Bc, Q_opt, R_opt);
    catch
        Cost = 1e15; % Penalty for failure (e.g., instability or numerical issue)
        return;
    end

    % --- 3. Run Simulink Simulation ---
    ModelName = 'COLQR_Simulation'; 
    
    % Ensure required variables for Simulink are available in the base workspace
    F_MR_MIN = -F_MR_MAX;
    assignin('base', 'K_lqr', K_lqr);
    assignin('base', 'F_MR_MAX', F_MR_MAX);
    assignin('base', 'F_MR_MIN', F_MR_MIN);
    assignin('base', 'input_excite', input_excite); 
    assignin('base', 'A', A);
    assignin('base', 'Bc', Bc);
    assignin('base', 'E', E);

    % Load the model if it's not open (prevents opening/closing during loop)
    if ~bdIsLoaded(ModelName); 
        try
            load_system(ModelName); 
        catch
            error('Simulink model "%s" not found. Please ensure the model file is in the current directory.', ModelName);
        end
    end
    
    % Define simulation time from the seismic input data
    SimTime = input_excite.Time(end); 
    
    % Run simulation and suppress output
    sim(ModelName, 'StartTime', '0', 'StopTime', num2str(SimTime), ...
        'SaveOutput', 'on', 'OutputSaveName', 'SimOut', ...
        'ReturnWorkspaceOutputs', 'on', 'SrcWorkspace', 'base');
             
    % --- 4. Extract States and Calculate Inter-Story Drift ---
    try
        % Load the controlled state vector X_controlled = [d1...d10, v1...v10]
        % This assumes the 'Controlled_States' To Workspace block exists and is working.
        X_controlled_ts = evalin('base', 'Controlled_States');
        
        % Extract displacement data (first 10 elements of the state vector)
        % Data is usually stored as (TimeSteps x States)
        Displacements = X_controlled_ts.signals.values(:, 1:10); 
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
    
    catch ME
        % If data extraction fails, return a very high penalty to guide WOA away
        warning('COLQR_CostFunction: Data extraction failed during iteration. Returning max cost. Error: %s', ME.message);
        MaxDrift = 1e10; 
    end
    
    % 5. Return Cost
    Cost = MaxDrift; 
end