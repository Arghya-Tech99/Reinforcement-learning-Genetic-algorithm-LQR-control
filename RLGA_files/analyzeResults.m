function analyzeResults(simOut, config, training_info)
    % Analyze and plot simulation results
    
    % Extract data from simulation
    time = simOut.tout;
    displacements = simOut.displacements.Data;
    story_drifts = simOut.story_drifts.Data;
    control_force = simOut.control_force.Data;
    
    % Plot training progress
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    if isfield(training_info, 'rl_stats')
        plot(training_info.rl_stats.EpisodeIndex, training_info.rl_stats.EpisodeReward);
        title('RL Training: Episode Reward');
        xlabel('Episode');
        ylabel('Reward');
        grid on;
    end
    
    subplot(2, 3, 2);
    plot(training_info.ga_fitness);
    title('GA Optimization: Best Fitness');
    xlabel('Generation');
    ylabel('Fitness');
    grid on;
    
    subplot(2, 3, 3);
    plot(time, displacements);
    title('Story Displacements');
    xlabel('Time (s)');
    ylabel('Displacement (m)');
    legend(arrayfun(@(x) sprintf('Story %d', x), 1:config.nStories, 'UniformOutput', false));
    grid on;
    
    subplot(2, 3, 4);
    plot(time, story_drifts);
    hold on;
    plot([time(1), time(end)], [config.driftLimit, config.driftLimit], 'r--', 'LineWidth', 2);
    title('Story Drifts vs Limit');
    xlabel('Time (s)');
    ylabel('Drift (m)');
    legend([arrayfun(@(x) sprintf('Story %d', x), 1:config.nStories, 'UniformOutput', false), {'Drift Limit'}]);
    grid on;
    
    subplot(2, 3, 5);
    plot(time, control_force);
    title('MR Damper Control Force');
    xlabel('Time (s)');
    ylabel('Force (N)');
    grid on;
    
    subplot(2, 3, 6);
    max_drifts = max(abs(story_drifts), [], 1);
    bar(1:config.nStories, max_drifts);
    hold on;
    plot([0, config.nStories+1], [config.driftLimit, config.driftLimit], 'r--', 'LineWidth', 2);
    title('Maximum Story Drifts');
    xlabel('Story');
    ylabel('Maximum Drift (m)');
    grid on;
    
    % Performance metrics
    max_absolute_drift = max(max_drifts);
    rms_control_force = rms(control_force);
    
    fprintf('\n=== Performance Metrics ===\n');
    fprintf('Maximum Story Drift: %.6f m (Limit: %.6f m)\n', max_absolute_drift, config.driftLimit);
    fprintf('RMS Control Force: %.2f N\n', rms_control_force);
    
    if max_absolute_drift < config.driftLimit
        fprintf('✓ Control criterion SATISFIED\n');
    else
        fprintf('✗ Control criterion NOT satisfied\n');
    end
end