function [control_force, current_command] = computeCOLQRForce(states, K_optimal, mr_params)
    % CO-LQR control force computation for Simulink
    
    persistent mr_force_prev;
    
    if isempty(mr_force_prev)
        mr_force_prev = 0;
    end
    
    % Optimal LQR force
    optimal_force = -K_optimal * states(:);
    
    % MR damper force (improved model)
    max_force = mr_params.maxForce;
    
    % IMPROVED Clipped optimal control logic
    force_demand = optimal_force;
    
    % Check if we can provide the demanded force
    if abs(force_demand) <= max_force
        % We can provide the exact force
        if sign(force_demand) == sign(mr_force_prev) || mr_force_prev == 0
            % Same direction or starting from zero - use maximum current
            current_command = mr_params.maxCurrent;
        else
            % Direction change - need to overcome damper
            current_command = mr_params.maxCurrent;
        end
        mr_force = force_demand;
    else
        % Force demand exceeds damper capacity - use maximum available
        current_command = mr_params.maxCurrent;
        mr_force = sign(force_demand) * max_force;
    end
    
    % Add hysteresis compensation
    if sign(optimal_force) ~= sign(mr_force_prev) && mr_force_prev ~= 0
        % Direction change - apply brief boost
        mr_force = 1.2 * mr_force; % 20% boost for direction changes
        mr_force = sign(mr_force) * min(abs(mr_force), max_force);
    end
    
    % Smooth force transitions
    force_change = abs(mr_force - mr_force_prev);
    if force_change > 0.5 * max_force
        % Limit rapid force changes
        max_change = 0.3 * max_force;
        if mr_force > mr_force_prev
            mr_force = mr_force_prev + max_change;
        else
            mr_force = mr_force_prev - max_change;
        end
    end
    
    control_force = mr_force;
    mr_force_prev = control_force;
    
    % Ensure force limits
    control_force = max(min(control_force, max_force), -max_force);
end