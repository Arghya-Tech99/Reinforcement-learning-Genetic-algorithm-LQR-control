%% 1. SYSTEM SETUP AND OPTIMIZATION

% 1.1 Run NSAE Generation 
disp('1. Generating Nonstationary Artificial Earthquake (NSAE)...');
[ag, t] = generate_nsae_kanai_tajimi_2023Brandao(); % MATLAB script containing NSAE function
disp('NSAE data generated and saved to seismic_input.mat.');

% 1.2 Define Dynamic Model Variables
disp('2. Defining Building Dynamic Model and System Matrices...');
Dynamic_model; % Parameters defining the dynamic model of building 

% 1.3 LQR Optimization 
n = 10; % number of stories
N_STATES = 2 * n; % 20 states (10 displacements and 10 velocities)
N_CONTROLS = size(Gamma, 2); % 1 control input (MR Damper force)

% === Q and R Tuning Parameters ===

% ****** HERE WE RUN A SCRIPT WHICH TUNES THESE WEIGHING MATRICES ******

% Using typical values for demonstration:
Q_param = 1e7; % General penalization factor for states
R_param = 1e-1;  % General penalization factor for control effort (force)

% Construct Q Matrix (Penalize all states equally)
Q_opt = Q_param * eye(N_STATES);

% Construct R Matrix
R_opt = R_param * eye(N_CONTROLS); 

disp(['3. Calculating Optimal LQR Gain K (Q=', num2str(Q_param), ', R=', num2str(R_param), ')...']);

% The LQR function finds the optimal steady-state feedback gain K.
% K: mr x 2n (1 x 20 in this case)
% u = -Kx
[K_lqr, P, eig_CL] = lqr(A, Bc, Q_opt, R_opt);
disp('Optimal LQR Gain K_lqr calculated successfully.');

% --- 1.4 Define Constraints ---
% Define the physical constraints of the MR Damper (max force output)
F_MR_MAX = 5.0 * 10^4; % 50 kN, using a realistic damper max force constraint
F_MR_MIN = -F_MR_MAX;  % Symmetry for the damper (can push and pull)

%% 2. SIMULATION SETUP

% --- 2.1 Prepare Time-Series Input ---
% Extract time step from the generated data
dt_sim = t(2) - t(1);

% Prepare the ground acceleration input for Simulink (time series object)
% Note: The input_excite variable from Dynamic_model.m is already defined 
% as a 2-column matrix [t, ag], but for time-domain simulation, 
% using a timeseries object is often cleaner.
input_excite = timeseries(ag, t);

% --- 2.2 System Definition (Required for COLQR Simulation) ---
% Define the LQR system equation (Continuous-Time State-Space)
% The system is defined in Dynamic_model.m, but we ensure the A, Bc, E matrices 
% are available for the simulation loop or Simulink.
% The relevant matrices are A (2n x 2n), Bc (2n x 1), E (2n x 1).

disp('4. Workspace variables finalized for Simulink model...');
disp('----------------------------------------------------');
disp('LQR Optimization Complete!');
disp('Variables K_lqr, F_MR_MAX, F_MR_MIN, and input_excite are now in the MATLAB Workspace.');

% Note: The COLQR logic (the clipping) is typically implemented inside the Simulink model 
% where the control signal (u = -Kx) is passed through a saturation block 
% limited by F_MR_MAX and F_MR_MIN before being applied to the damper force input.