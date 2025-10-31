% The 'load' command imports 'ag' and 't' from the .mat file 
% and places them directly into this script's workspace.
load('seismic_input.mat', 'ag', 't');
t_column = t(:);
ag_column = ag(:);

input_excite = [t_column, ag_column];

n = 10; % 10-storied shear building

% DEFINING THE MASS MATRIX
m = 3.5 * (10^4);
Ms = m * eye(10);

% DEFINING THE STIFFNESS MATRIX
k = 6.0 * (10^5);
% 1. Create the main diagonal (k1 + k2 = 2k)
% The main diagonal is 2k for floors 1 to n-1 (9 floors) and k for the top floor (10)
main_diag_K = [2*k * ones(n-1, 1); k];
% 2. Create the upper and lower off-diagonals (-k)
off_diag_K = -k * ones(n-1, 1);
% 3. Assemble the matrix using the diag function
% diag(A, k) creates a matrix with vector A on the kth diagonal.
Ks = diag(main_diag_K, 0) + ...  % Main diagonal (k=0)
     diag(off_diag_K, 1) + ...   % Upper off-diagonal (k=1)
     diag(off_diag_K, -1);       % Lower off-diagonal (k=-1)
     
% disp('The 10x10 Stiffness Matrix Ks is:');
% disp(Ks);

% DEFINING THE DAMPING MATRIX
c = 6.5 * (10^7);
% 1. Create the main diagonal (k1 + k2 = 2k)
% The main diagonal is 2k for floors 1 to n-1 (9 floors) and k for the top floor (10)
main_diag_C = [2*c * ones(n-1, 1); c];
% 2. Create the upper and lower off-diagonals (-k)
off_diag_C = -c * ones(n-1, 1);
% 3. Assemble the matrix using the diag function
% diag(A, k) creates a matrix with vector A on the kth diagonal.
Cs = diag(main_diag_C, 0) + ...  % Main diagonal (k=0)
     diag(off_diag_C, 1) + ...   % Upper off-diagonal (k=1)
     diag(off_diag_C, -1);       % Lower off-diagonal (k=-1)
     
% disp('The 10x10 Stiffness Matrix Cs is:');
% disp(Cs);

% DEFINING Lambda - location vector of the seismic forces
Lambda = ones(n, 1);

% DEFINING Gamma - Control force location matrix
mr = 1; % Number of MR Dampers used = mr = 1
Gamma = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0];

% System Matrix A (2n x 2n)
Zero_n = zeros(n, n);
Eye_n = eye(n);
A_top = [Zero_n, Eye_n];
A_bottom = [-Ms \ Ks, -Ms \ Cs]; % Ms \ Ks is equivalent to inv(Ms) * Ks
A = [A_top; A_bottom];

% Seismic Input Matrix E (2n x 1)
E_top = zeros(n, 1);
E_bottom = -Lambda; 
E = [E_top; E_bottom];

% Control Input Matrix Bc (2n x mr)
Bc_top = zeros(n, mr);
Bc_bottom = -Ms \ Gamma;
Bc = [Bc_top; Bc_bottom];

% Concatenate E and Bc into a single input matrix (2n x (mr+1))
Bce = [Bc, E];

% Positions of the vectors of seismic acclerations F (2n x 1)
F_top = zeros(n, 1);
F_bottom = -Lambda; 
F = [F_top; F_bottom];

% Positions of the vectors of control forces Dc (2n x mr)
Dc_top = zeros(n, mr);
Dc_bottom = -Ms \ Gamma;
Dc = [Dc_top; Dc_bottom];

% Concatenate F and Dc into a single matrix (2n x (mr+1))
Dcf = [Dc, F];

% Defining the input state u(t)
fmr = 0 ; 
% fmr - Control force applied by MR Damper
% fmr = 0 when uncontrolled
% fmr -> Computed using COLQR algorithm
xg_Ddot = ag ; % xg_Ddot - double dot of xg
u = [fmr,xg_Ddot];

% Output Matrix C (2n x 2n)
Identity_n = eye(n, n);
Zero_n = zeros(n, n);
C_top = [Zero_n, Identity_n];
C_bottom = [-Ms \ Ks, -Ms \ Cs]; % Ms \ Ks is equivalent to inv(Ms) * Ks
C = [C_top; C_bottom];

% DEFINING THE STATE-SPACE MODEL OF THE BUILDING
% The 'ss' function creates a continuous-time state-space object:
% ss(A, B, C, D)
sys_ss = ss(A, Bce, C, Dcf);

% disp('Continuous-time State-Space Model (sys_ss) defined.');
% disp(['System States (2n): ', num2str(size(A, 1))]);
% disp(['System Inputs: ', num2str(size(Bce, 2))]);
% disp(['System Outputs: ', num2str(size(C, 1))]);