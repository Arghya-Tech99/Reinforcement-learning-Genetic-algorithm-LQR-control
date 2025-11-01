function [BestPosition, BestCost, ConvergenceCurve] = WhaleOptimizationAlgorithm(CostFunction, Params)
% WhaleOptimizationAlgorithm: Implements the WOA metaheuristic.

% Inputs:
%   CostFunction: A handle to the function to be minimized (@COLQR_CostFunction)
%   Params: Structure containing optimization parameters (PopSize, MaxIter, Bounds, etc.)

% Outputs:
%   BestPosition: The best [log10(Q), log10(R)] found
%   BestCost: The minimum cost (Max Inter-Story Drift) achieved
%   ConvergenceCurve: History of the best cost over iterations

    % Initialization
    PopSize = Params.PopulationSize;
    MaxIter = Params.MaxIterations;
    Dim = Params.Dim;
    Lb = Params.LowerBound;
    Ub = Params.UpperBound;

    % Dynamic Model variables passed from the main script
    A = evalin('base', 'A');
    Bc = evalin('base', 'Bc');
    E = evalin('base', 'E');
    F_MR_MAX = evalin('base', 'F_MR_MAX');
    input_excite = evalin('base', 'input_excite');
    N_STATES = evalin('base', 'N_STATES');
    N_CONTROLS = evalin('base', 'N_CONTROLS');
    
    % Initialize Whales (Search Agents)
    Positions = zeros(PopSize, Dim);
    for i = 1:PopSize
        Positions(i, :) = Lb + (Ub - Lb) .* rand(1, Dim);
    end

    % Initial Evaluation
    Costs = zeros(PopSize, 1);
    for i = 1:PopSize
        Costs(i) = CostFunction(Positions(i, :), A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS);
    end

    [BestCost, BestIndex] = min(Costs);
    BestPosition = Positions(BestIndex, :);
    
    ConvergenceCurve = zeros(MaxIter, 1);

    % Main Loop
    for t = 1:MaxIter
        a = 2 - t * (2/MaxIter); % a linearly decreases from 2 to 0
        
        for i = 1:PopSize
            r1 = rand(); r2 = rand();
            A_vec = 2*a*r1 - a; % A vector (controls exploration/exploitation)
            C_vec = 2*r2;      % C vector
            
            l = rand() * 2 - 1; % Random number in [-1, 1] for spiral movement
            p = rand();         % Probability of switching between search and attack

            % Update Position
            if p < 0.5 % Encircling or Search for Prey
                if abs(A_vec) < 1 % Encircling Prey (Exploitation)
                    D_abs = abs(C_vec * BestPosition - Positions(i, :));
                    Positions(i, :) = BestPosition - A_vec * D_abs;
                else % Search for Prey (Exploration)
                    RandIndex = randi(PopSize);
                    RandPosition = Positions(RandIndex, :);
                    D_abs = abs(C_vec * RandPosition - Positions(i, :));
                    Positions(i, :) = RandPosition - A_vec * D_abs;
                end
            else % Bubble-net Attacking (Exploitation)
                D_prime = abs(BestPosition - Positions(i, :));
                Positions(i, :) = D_prime * exp(l * 1) .* cos(l * 2 * pi) + BestPosition;
            end
            
            % Check bounds and calculate cost
            Positions(i, :) = max(Positions(i, :), Lb);
            Positions(i, :) = min(Positions(i, :), Ub);
            
            NewCost = CostFunction(Positions(i, :), A, Bc, E, F_MR_MAX, input_excite, N_STATES, N_CONTROLS);
            
            if NewCost < Costs(i)
                Costs(i) = NewCost;
                if NewCost < BestCost
                    BestCost = NewCost;
                    BestPosition = Positions(i, :);
                end
            end
        end
        
        ConvergenceCurve(t) = BestCost;
        disp(['Iteration ', num2str(t), ': Best Cost = ', num2str(BestCost)]);
    end
end