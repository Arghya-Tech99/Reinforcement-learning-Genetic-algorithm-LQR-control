function InitialPopulation = RL_DQL_Initialization(TargetSize, LB, UB, CostFunction, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS)
% RL_DQL_Initialization: Generates a progenitor population for the GA 
% by using a simplified Q-Learning agent to guide intelligent exploration.
%
% This function replaces the Gaussian sampling approach with an epsilon-greedy 
% policy to search for stable, low-cost LQR parameters.

    % --- 1. DQL Parameters ---
    N_SAMPLES = TargetSize * 30; % Total number of exploration steps/episodes
    N_DIM = 2;                   % [log10(Q), log10(R)]
    
    % Q-Learning Hyperparameters
    GAMMA = 0.9;      % Discount Factor
    ALPHA = 0.3;      % Learning Rate
    EPSILON_START = 1.0; % Initial exploration rate
    EPSILON_END = 0.1;  % Final exploration rate
    
    % --- 2. Discretized Action Space (DQL Actions) ---
    % The agent can choose one of 5 actions: 
    % 1-4: Step by a fixed amount (0.5 in log-space)
    % 5: Jump randomly (Exploration)
    
    STEP_SIZE = 0.21;
    Actions = [
        STEP_SIZE, 0;       % 1: Increase log10(Q)
        -STEP_SIZE, 0;      % 2: Decrease log10(Q)
        0, STEP_SIZE;       % 3: Increase log10(R)
        0, -STEP_SIZE;      % 4: Decrease log10(R)
        0, 0                % 5: Placeholder for random jump (handled below)
    ];
    N_ACTIONS = size(Actions, 1);
    
    % --- 3. DQL State/Q-Value Setup (Simplified) ---
    % State is defined by its index in the search space.
    % We need to discretize the continuous log-space into "States" (Bins).
    N_BINS = 10; % Discretize Q and R log-space into 10 bins each (10x10 state space)
    Q_BINS = linspace(LB(1), UB(1), N_BINS+1);
    R_BINS = linspace(LB(2), UB(2), N_BINS+1);
    
    % Q-Table: Stores the Q-value for each State-Action pair.
    % Q_Table(Q_bin, R_bin, Action_index)
    Q_Table = zeros(N_BINS, N_BINS, N_ACTIONS);
    
    % Store all successful samples and their costs
    Samples = zeros(N_SAMPLES, N_DIM);
    Costs = zeros(N_SAMPLES, 1);
    
    % Initialize starting position (center of the search space)
    CurrentSample = (LB + UB) / 2; 
    
    disp(['   Exploration Phase: Running ', num2str(N_SAMPLES), ' Q-Learning trials.']);
    
    % Helper function to convert continuous log-Q/R into a discrete State index (bin)
    function state = get_state(log_qr)
        q_idx = max(1, min(N_BINS, sum(log_qr(1) >= Q_BINS(1:N_BINS))));
        r_idx = max(1, min(N_BINS, sum(log_qr(2) >= R_BINS(1:N_BINS))));
        state = [q_idx, r_idx];
    end

    % --- 4. DQL Exploration Loop ---
    for i = 1:N_SAMPLES
        % Calculate exploration rate decay (from start down to end)
        EPSILON = EPSILON_START * exp(-i / (N_SAMPLES / 4)) + EPSILON_END;
        
        % Current state index (Q_bin, R_bin)
        S_idx = get_state(CurrentSample);
        
        % 4.1. Epsilon-Greedy Action Selection
        if rand() < EPSILON
            % Explore: Choose a random action (1 to N_ACTIONS) OR a random jump
            Action_index = randi([1, N_ACTIONS]);
            
            if Action_index == N_ACTIONS % Use the 5th action to trigger a random jump
                Action_vector = LB + (UB - LB) .* rand(1, N_DIM); % Full random jump
                NewSample = Action_vector;
            else
                Action_vector = Actions(Action_index, :);
                NewSample = CurrentSample + Action_vector;
            end
        else
            % Exploit: Choose the best action from the Q-Table
            [~, Action_index] = max(Q_Table(S_idx(1), S_idx(2), 1:N_ACTIONS-1)); % Exclude random jump
            Action_vector = Actions(Action_index, :);
            NewSample = CurrentSample + Action_vector;
        end
        
        % 4.2. Enforce Bounds
        NewSample = max(LB, min(UB, NewSample));
        
        % 4.3. Execute Action & Observe Reward/Cost
        CurrentCost = CostFunction(NewSample, A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS);
        
        % Calculate Reward (R): Negative of the Cost (Max Drift), capped for stability
        MAX_PENALTY = 1.0e03;
        if CurrentCost >= MAX_PENALTY
            Reward = -MAX_PENALTY; % Large negative reward for instability
        else
            Reward = -CurrentCost; % Negative cost is the reward (we want to maximize reward/minimize cost)
        end
        
        % 4.4. Update Q-Table (Bellman Equation)
        S_prime_idx = get_state(NewSample);
        
        % The Q-update is only performed for directional steps (1 to 4)
        if Action_index <= N_ACTIONS - 1 
            % Q(S,A) = Q(S,A) + alpha * [R + gamma * max(Q(S',a')) - Q(S,A)]
            Q_s_a = Q_Table(S_idx(1), S_idx(2), Action_index);
            
            % Max Q-value of the next state S'
            Max_Q_s_prime_a_prime = max(Q_Table(S_prime_idx(1), S_prime_idx(2), 1:N_ACTIONS-1));
            
            % Update
            Q_Table(S_idx(1), S_idx(2), Action_index) = Q_s_a + ALPHA * (Reward + GAMMA * Max_Q_s_prime_a_prime - Q_s_a);
        end
        
        % 4.5. Transition to Next State (Step/Sample)
        CurrentSample = NewSample;

        % 4.6. Store Sample
        Samples(i, :) = NewSample;
        Costs(i) = CurrentCost;
        
        % Optional: Display progress
        if mod(i, floor(N_SAMPLES/10)) == 0 || i == N_SAMPLES
            disp(['     Trial ', num2str(i), '/', num2str(N_SAMPLES), '. Epsilon: ', num2str(EPSILON, '%.2f')]);
        end
    end
    
    % --- 5. Progenitor Selection (GA Initialization) ---
    
    % 1. Filter out failed simulations (Cost = 1e9 or more)
    ValidIndices = Costs < MAX_PENALTY;
    FilteredSamples = Samples(ValidIndices, :);
    FilteredCosts = Costs(ValidIndices);
    
    if length(FilteredCosts) < TargetSize
        warning('RL-DQL: Insufficient valid samples found (%d). The GA population will be padded with random samples.', length(FilteredCosts));
        % Fallback: Pad with random samples
        RandomSamples = LB + (UB - LB) .* rand(TargetSize - length(FilteredSamples), N_DIM);
        InitialPopulation = [FilteredSamples; RandomSamples];
        
        % If we still don't have enough (TargetSize is large), pad with any random samples
        if size(InitialPopulation, 1) < TargetSize
            RandomFill = LB + (UB - LB) .* rand(TargetSize - size(InitialPopulation, 1), N_DIM);
            InitialPopulation = [InitialPopulation; RandomFill];
        end
        
    else
        % 2. Sort valid samples by cost (best performing first)
        [~, SortedIndices] = sort(FilteredCosts, 'ascend');
        
        % 3. Select the top 'TargetSize' samples as the progenitor population
        InitialPopulation = FilteredSamples(SortedIndices(1:TargetSize), :);
    end
    
    disp(['   DQL-Enhanced Initialization complete. Selected top ', num2str(size(InitialPopulation, 1)), ' progenitors.']);

end