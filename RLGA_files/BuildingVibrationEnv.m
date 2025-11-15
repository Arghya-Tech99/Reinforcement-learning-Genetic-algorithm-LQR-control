classdef BuildingVibrationEnv < rl.env.MATLABEnvironment
    %BUILDINGVIBRATIONENV Custom environment for building vibration control
    
    properties
        % Configuration
        Config
        StateSpace
        Earthquake
        CurrentStep
        CurrentState
        TotalReward
        MaxDrift
        
        % Constants for Reward Shaping
        MaxDriftPenalty = 10000;  % Large penalty for unstable simulation/exceeding max drift
        DriftLimitReward = 100;   % Max reward for a safe step (drift < limit)
    end
    
    methods
        function this = BuildingVibrationEnv(config, state_space, earthquake)
            % Initialize environment
            ObservationInfo = rlNumericSpec([2*config.nStories 1]);
            % Action space: log10(Q) and log10(R) factors
            ActionInfo = rlNumericSpec([2 1], 'LowerLimit', [-6; -6], 'UpperLimit', [10; 6]);
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
            
            % Store configuration
            this.Config = config;
            this.StateSpace = state_space;
            this.Earthquake = earthquake;
            
            % Initialize episode data
            this.reset();
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % Step function for the environment
            
            this.CurrentStep = this.CurrentStep + 1;
            
            % 1. State/Action Mapping
            % Action is the *change* in log10(Q) and log10(R)
            step_size = 0.5; % Define a moderate step size
            next_log_Q = this.CurrentState(1) + step_size * Action(1);
            next_log_R = this.CurrentState(2) + step_size * Action(2);
            
            % Clip parameters to search bounds
            next_log_Q = max(log10(this.Config.storyMass * 100), min(log10(1e12), next_log_Q));
            next_log_R = max(log10(1e-12), min(log10(1e6), next_log_R));
            
            Observation = [next_log_Q; next_log_R];
            
            % 2. Evaluate Cost (Run Simulation)
            [cost, is_stable] = this.evaluateLQR(10^next_log_Q, 10^next_log_R);
            
            % 3. === REWARD CALCULATION (Dense Hyperbolic Reward) ===
            isDone = false;
            
            if ~is_stable
                % Hard penalty for instability (simulation failed or generated NaN/Inf cost)
                Reward = -this.MaxDriftPenalty;
                isDone = true;
            else
                % Use max_drift as the primary reward signal (cost is max_drift)
                max_drift = cost;
                this.MaxDrift = max(this.MaxDrift, max_drift);
                
                % Normalized Drift (0 at drift limit, 1 at 0 drift)
                % Hyperbolic function: R = R_max * (1 - normalized_drift)
                % Normalized to the drift limit (e.g., 0.006 m)
                
                if max_drift > this.Config.driftLimit
                    % Heavy penalty for exceeding the acceptable drift limit
                    Reward = -100 * (max_drift / this.Config.driftLimit);
                else
                    % Dense positive reward based on how far below the limit it is
                    % R = R_max * (1 - (current_drift / max_acceptable_drift))
                    normalized_drift_ratio = max_drift / this.Config.driftLimit;
                    Reward = this.DriftLimitReward * (1 - normalized_drift_ratio);
                    
                    % Small bonus for finding a new record
                    if max_drift < this.Config.driftLimit * 0.95 
                        Reward = Reward + 10; 
                    end
                end
            end
            
            this.CurrentState = Observation;
            this.TotalReward = this.TotalReward + Reward;

            % 4. Check Termination
            if this.CurrentStep >= this.Config.rlMaxSteps
                isDone = true;
            end
            
            LoggedSignals.MaxDrift = this.MaxDrift;
            LoggedSignals.Cost = cost;
            LoggedSignals.TotalReward = this.TotalReward;
        end
        
        function InitialObservation = reset(this)
            % Reset environment to a random state within stability bounds
            
            % Initialize state: log10(Q) and log10(R)
            % Use a stable, mid-range point to start
            q_mid = log10(this.Config.storyStiffness * 10); % e.g., 7.8
            r_mid = log10(1); % e.g., 0
            
            this.CurrentState = [q_mid; r_mid] + rand(2, 1) .* 0.5; % Add small randomness
            
            this.CurrentStep = 0;
            this.TotalReward = 0;
            this.MaxDrift = 0;
            
            InitialObservation = this.CurrentState;
        end
    end
    
    methods (Access = private)
        function [cost, is_stable] = evaluateLQR(this, Q_factor, R_factor)
            % Q_factor and R_factor are linear scale (not log)
            is_stable = true;
            MAX_PENALTY = this.MaxDriftPenalty;
            cost = MAX_PENALTY;
            
            % 1. Compute LQR Gain
            Q_matrix = Q_factor * eye(size(this.StateSpace.A, 1));
            R_matrix = R_factor * eye(size(this.StateSpace.Bc, 2));
            
            try
                K_gain = lqr(this.StateSpace.A, this.StateSpace.Bc, Q_matrix, R_matrix);
            catch
                is_stable = false; return;
            end
            
            % 2. Setup Simulation Workspace
            % The variables A, Bc, E, input_excite, F_MR_MAX, F_MR_MIN are already 
            % assigned to the 'base' workspace in main_RLGA_COLQR.m.
            
            % We only need to assign the new gain and damper parameters to the base
            % workspace for this single evaluation.
            
            assignin('base', 'K_RLGA', K_gain);
            assignin('base', 'K_optimal', K_gain); % For computeCOLQRForce consistency
            
            mr_params.maxForce = this.Config.mrMaxForce;
            mr_params.maxCurrent = this.Config.mrMaxCurrent;
            assignin('base', 'mr_params', mr_params);
            
            % 3. Run Simulation
            try
                simOut = sim('WOA_vs_RLGA', ...
                    'StopTime', num2str(this.Earthquake.time(end)), ...
                    'SrcWorkspace', 'base', ...
                    'ReturnWorkspaceOutputs', 'on');
            catch
                is_stable = false; return;
            end
            
            % 4. Calculate Cost (Max Story Drift)
            try
                story_drifts = simOut.get('story_drifts');
                
                if ~isa(story_drifts, 'timeseries') || isempty(story_drifts.Data)
                    is_stable = false; cost = MAX_PENALTY; return;
                end
                
                % Cost is the maximum absolute story drift
                cost = max(max(abs(story_drifts.Data)));
                
                if isnan(cost) || isinf(cost) || cost < 0 || ~isreal(cost)
                    is_stable = false; cost = MAX_PENALTY;
                end
                
            catch
                is_stable = false; cost = MAX_PENALTY;
            end
        end
    end
end