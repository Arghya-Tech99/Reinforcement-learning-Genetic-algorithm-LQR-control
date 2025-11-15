function [control_force, current_command] = computeCOLQRForce(states, K_optimal, mr_params)
    % CO-LQR control force computation for Simulink
    
    persistent mr_force_prev;
    
    if isempty(mr_force_prev)
        mr_force_prev = 0;
    end
    
    % Optimal LQR force
    optimal_force = -K_optimal * states(:);
    
    % MR damper force (simplified model)
    max_force = mr_params.maxForce;
    
    % Clipped optimal control logic
    if abs(optimal_force) > abs(mr_force_prev) && sign(optimal_force) == sign(mr_force_prev)
        % Increase current to match optimal force
        current_command = mr_params.maxCurrent;
    else
        % Zero current
        current_command = 0;
    end
    
    % Simplified MR force model (replace with Bouc-Wen if needed)
    mr_force = current_command * max_force / mr_params.maxCurrent;
    
    % Clip to maximum force
    control_force = sign(optimal_force) * min(abs(optimal_force), abs(mr_force));
    mr_force_prev = control_force;
end