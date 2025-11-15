function [K, P] = computeLQRGain(A, B, Q, R)
    % Compute LQR gain matrix
    try
        [K, P] = lqr(A, B, Q, R);
    catch ME
        warning('LQR computation failed: %s. Using backup controller.', ME.message);
        % Backup: simple pole placement
        poles = -0.1 * ones(size(A, 1), 1);
        K = place(A, B, poles);
        P = [];
    end
end