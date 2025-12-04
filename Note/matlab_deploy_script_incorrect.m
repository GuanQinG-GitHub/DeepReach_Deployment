%% DeepReach MATLAB Deployment Script (Corrected)
clear; clc;

%% 1. Load Model
data = load('deepreach_model.mat');

%% 2. Define Grid
% Define grid matching the problem domain
% Dubins3D: x, y, theta
grid_min = [-0.9; -0.9; -pi];
grid_max = [0.9; 0.9; pi];
N = [101; 101; 101]; % Resolution

x = linspace(grid_min(1), grid_max(1), N(1));
y = linspace(grid_min(2), grid_max(2), N(2));
theta = linspace(grid_min(3), grid_max(3), N(3));

[X, Y, TH] = ndgrid(x, y, theta);

num_points = numel(X);

%% 3. Prepare Input
% We will evaluate at t=0 (Target Set) and t=1 (Final Reachable Set)
t_eval = 0.0; % Change to 1.0 to see evolved set

% Flatten grid
X_flat = X(:)';
Y_flat = Y(:)';
TH_flat = TH(:)';
T_flat = t_eval * ones(1, num_points);

% Input: [t; x; y; theta] (4 x N)
model_in = [T_flat; X_flat; Y_flat; TH_flat];

%% 4. Normalize Input
% state_mean/var are 3x1 (for x, y, theta)
state_mean = double(data.state_mean);
state_var = double(data.state_var);

% Ensure column vectors
if size(state_mean, 2) > 1; state_mean = state_mean'; end
if size(state_var, 2) > 1; state_var = state_var'; end

% Normalize spatial dimensions (indices 2, 3, 4)
% input = (coord - mean) ./ var
model_in(2:end, :) = bsxfun(@rdivide, bsxfun(@minus, model_in(2:end, :), state_mean), state_var);

%% 5. Forward Pass (5 Layers)
z = model_in;

% Layer 1
z = data.W1 * z + data.b1;
z = sin(30 * z);

% Layer 2
z = data.W2 * z + data.b2;
z = sin(30 * z);

% Layer 3
z = data.W3 * z + data.b3;
z = sin(30 * z);

% Layer 4
%% DeepReach MATLAB Deployment Script (Corrected)
clear; clc;

%% 1. Load Model
data = load('deepreach_model.mat');

%% 2. Define Grid
% Define grid matching the problem domain
% Dubins3D: x, y, theta
grid_min = [-0.9; -0.9; -pi];
grid_max = [0.9; 0.9; pi];
N = [101; 101; 101]; % Resolution

x = linspace(grid_min(1), grid_max(1), N(1));
y = linspace(grid_min(2), grid_max(2), N(2));
theta = linspace(grid_min(3), grid_max(3), N(3));

[X, Y, TH] = ndgrid(x, y, theta);

num_points = numel(X);

%% 3. Prepare Input
% We will evaluate at t=0 (Target Set) and t=1 (Final Reachable Set)
t_eval = 0.0; % Change to 1.0 to see evolved set

% Flatten grid
X_flat = X(:)';
Y_flat = Y(:)';
TH_flat = TH(:)';
T_flat = t_eval * ones(1, num_points);

% Input: [t; x; y; theta] (4 x N)
model_in = [T_flat; X_flat; Y_flat; TH_flat];

%% 4. Normalize Input
% state_mean/var are 3x1 (for x, y, theta)
state_mean = double(data.state_mean);
state_var = double(data.state_var);

% Ensure column vectors
if size(state_mean, 2) > 1; state_mean = state_mean'; end
if size(state_var, 2) > 1; state_var = state_var'; end

% Normalize spatial dimensions (indices 2, 3, 4)
% input = (coord - mean) ./ var
model_in(2:end, :) = bsxfun(@rdivide, bsxfun(@minus, model_in(2:end, :), state_mean), state_var);

%% 5. Forward Pass (5 Layers)
z = model_in;

% Layer 1
z = data.W1 * z + data.b1;
z = sin(30 * z);

% Layer 2
z = data.W2 * z + data.b2;
z = sin(30 * z);

% Layer 3
z = data.W3 * z + data.b3;
z = sin(30 * z);

% Layer 4
z = data.W4 * z + data.b4;
z = sin(30 * z);

% Layer 5 (Output)
z = data.W5 * z + data.b5;

%% 6. Denormalize Output
value_mean = double(data.value_mean);
value_var = double(data.value_var);
value_normto = double(data.value_normto);

% Check model type (default to 'exact' if not specified, as per user context)
% In 'exact' mode: V = (NN * t * value_var / value_normto) + boundary_fn(x)
% In 'vanilla' mode: V = NN * value_var + value_mean

% Implement Boundary Function (Dubins3DDiscounted)
% g(x) = min{L-|x1|, L-|x2|, dist² - r²}
% L = 0.9, r = 0.3, Cx = 0, Cy = 0
L = 0.9;
r = 0.3;
Cx = 0.0;
Cy = 0.0;

x1 = X_flat;
x2 = Y_flat;

% Box constraints
g1 = L - abs(x1);
g2 = L - abs(x2);

% Circular obstacle
dist_sq = (x1 - Cx).^2 + (x2 - Cy).^2;
g3 = dist_sq - r^2;

% Boundary value l(x)
l_x = min(min(g1, g2), g3);

% Compute Final Value
% Note: z is the raw NN output
% T_flat is the time vector
V_flat = (z .* T_flat * value_var / value_normto) + l_x;

% Reshape to grid
V = reshape(V_flat, N');

%% 7. Visualization
figure(1); clf;
% Plot isosurface at V=0 (Boundary)
% Negative = Unsafe, Positive = Safe
p = patch(isosurface(X, Y, TH, V, 0));
p.FaceColor = 'red';
p.EdgeColor = 'none';
p.FaceAlpha = 0.5;
camlight; lighting phong;
axis equal;
xlabel('x'); ylabel('y'); zlabel('theta');
title(['Zero Level Set at t = ', num2str(t_eval)]);
grid on;
xlim([grid_min(1) grid_max(1)]);
ylim([grid_min(2) grid_max(2)]);
zlim([grid_min(3) grid_max(3)]);

disp('Done.');