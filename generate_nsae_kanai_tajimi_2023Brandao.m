function [ag, t] = generate_nsae_kanai_tajimi() 
% Defining Parameters
T = 10.000;
dt = 0.002;
g = 9.81;
PGA_target = 0.3*g;

% Kanai-Tajimi Spectrum Parameters
xi_g = 0.6; % Soil damping
omega_g = 4*pi; % Frequency

% Envelope Function Parameters 
a1 = 1.35;           
a2 = 0.5;

% Frequency Vector Setup (Eq. 10)
Delta_omega = 2 * pi / T; % Frequency increment 

% Max frequency for the summation
omega_max = 30 * 2 * pi; 
    
% Create time vector
t = 0:dt:T;

% Create discrete frequency vector (omega_j)
omega_j = Delta_omega:Delta_omega:omega_max;
N_omega = length(omega_j); % Number of frequency intervals 

% S0 constant spectral density (Equation 9) 
S0 = (0.03 * xi_g) / (pi * omega_g * (4 * xi_g^2 + 1));

% PSD function S(omega_j) (Equation 9) [cite: 999, 1000]
omega2 = omega_j.^2;
omega4 = omega_j.^4;

numerator = omega_g^4 + 4 * omega_g^2 * xi_g^2 * omega2;
denominator = (omega2 - omega_g^2).^2 + 4 * omega_g^2 * xi_g^2 * omega2;

S_omega_j = S0 * (numerator ./ denominator);
S_omega_j_PLOT = S_omega_j*(1/g)^2 * (2 * pi);

% Generate random phase angles (phi_j) uniformly distributed from 0 to 2*pi 
phi_j = 2 * pi * rand(1, N_omega); 

% Pre-calculate the amplitude term: sqrt(S(omega_j) * Delta_omega)
amplitude_term = sqrt(S_omega_j * Delta_omega);

% Initialize stationary acceleration vector
ag_stationary = zeros(size(t));

    % Summation loop (Equation 10) [cite: 1005]
    for j = 1:N_omega
        % Calculating the summation
        ag_stationary = ag_stationary + amplitude_term(j) * cos(omega_j(j) * t + phi_j(j));
    end
    ag_stationary = sqrt(2) * ag_stationary;

% Envelope function g(t)
g_t = a1 * t .* exp(-a2 * t);

% Nonstationary Artificial Earthquake (NSAE)
ag_nsae = ag_stationary .* g_t;

% Find the current PGA
PGA_current = max(abs(ag_nsae));

% Scale factor
scale_factor = PGA_target / PGA_current;

% Final scaled acceleration (ag)
ag = scale_factor * ag_nsae;

% Plotting the NSAE
figure;
plot(t, ag, 'b-', 'LineWidth', 1.0);
title('Generated Nonstationary Artificial Earthquake (NSAE)', 'FontSize', 14);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Acceleration (m/s^2)', 'FontSize', 12);
grid on;

% Plotting the PSD of Kanai-Taniji Spectrum 
figure;
plot(omega_j,S_omega_j_PLOT*(10^3), 'b-', 'LineWidth', 1.0);
title('Kanai-Tajimi Power Spectral Density (PSD)', 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
grid on;
xlim([0 150]);

% Save the variables 'ag' and 't' to a file named 'seismic_input.mat'
save('seismic_input.mat', 'ag', 't'); 

disp('Seismic acceleration data saved successfully.');

end
