% prediction of COVID-19 daily infections in Israel
clear; close all; clc;


%% Load data

raw_data = readtable('corona_new_cases.csv');

%% Preprocess data

n_days = 14;


X_all   = buffer(raw_data{:, 'New_verified_cases'}, n_days, n_days - 1);
X_all   = X_all(:, n_days:(end - 1));
Y0_all  = raw_data{(n_days + 1):end, 'New_verified_cases'}';

n_samples = length(Y0_all);

%% Apply a logarithmic transformation to all data

X_all   = log(X_all);
Y0_all  = log(Y0_all);

%% Split data into train and validation sets

valid_ratio = 0.2;

n_train_samples = floor((1 - valid_ratio)*n_samples);
X_train         = X_all(:, 1:n_train_samples);
Y0_train        = Y0_all(1:n_train_samples);
X_valid         = X_all(:, (n_train_samples + 1):end);
Y0_valid        = Y0_all((n_train_samples + 1):end);

%% Define the network

% Set dimensions
n_input = n_days;
n_per_hidden_layer = 31;
L = 4; % number of layers including the input & output layers
n_output = 1;

% Initialize weights
W{1} = randn(n_per_hidden_layer + 1, n_input + 1); % extra weight for bias
for i=2:L-2
    W{i} = randn(n_per_hidden_layer + 1, n_per_hidden_layer + 1); % extra weight for bias
end
W{L-1} = randn(n_output, n_per_hidden_layer + 1); % extra weight for bias

% Set an activation function for each layer
%TODO 2: examine the activation functions code and understand their output
for i=1:L-2
    g{i} = @ReLU;
end
g{L-1} = @Linear;

%% Declare the learning parameters
eta      	= 1e-6;   %TODO 3: Set Learning rate
n_epochs    = 11000;  %TODO 4: Set Number of learning epochs

%% Learn

% Loop over learning epochs
for ep = 1:n_epochs
    
    % random order of samples
    samp_order = randperm(n_train_samples);
    
    % Loop over all samples
    for samp = 1:n_train_samples
    
        % Choose a random sample
        s    = samp_order(samp);
        x{1} = X_train(:, s);
        x{1}(end + 1) = 1;
        y0   = Y0_train(s);
        
        % Forward pass
        for i=2:L
            
            [x{i}, xp{i}] = g{i-1}(W{i-1}*x{i-1});
            % add bias neuron with constant activation
            if i ~= L
                x{i}(end) = 1;
                xp{i}(end) = 0;
            end
        end
        
        % Backward pass        
        delta{L-1} = (x{L}-y0)*xp{L};
        d{L-1} = delta{L-1}*transpose(x{L-1});
        for i=2:L-1
            delta{L-i} = transpose(W{L-i+1})*delta{L-i+1}.*xp{L-i+1};
            d{L-i} = delta{L-i}*transpose(x{L-i});
        end
        
        % update weights
        for i=1:L-1
            W{i} = W{i} - eta*d{i};
        end
        
    end
        
end

%% Get output

Forward pass the whole training set
X_train_fp{1} = X_train;
% bias neuron for all examples
X_train_fp{1} = [X_train_fp{1};ones(1,size(X_train_fp{1},2))];
for i=2:L
    X_train_fp{i} = g{i-1}(W{i-1}*X_train_fp{i-1});
    % add bias neuron
    if i ~= L
        X_train_fp{i}(end,:) = 1;
    end
end
Y_train = X_train_fp{L};

Forward pass the whole validation set
X_valid_fp{1} = X_valid;
% bias neuron for all examples
X_valid_fp{1} = [X_valid_fp{1};ones(1,size(X_valid_fp{1},2))];

for i=2:L
    X_valid_fp{i} = g{i-1}(W{i-1}*X_valid_fp{i-1});
    % add bias neuron
    if i ~= L
        X_valid_fp{i}(end,:) = 1;
    end
end
Y_valid = X_valid_fp{L};

%% Print errors and R2

% Squared errors
train_err = mean((Y_train-Y0_train).^2); %TODO 9: calculate the mean squared error for the training set
valid_err = mean((Y_valid-Y0_valid).^2); %TODO 9: calculate the mean squared error for the validation set

fprintf('Training error:\t%g\nValidation error:\t%g\n', ...
    train_err, valid_err);

fprintf('\n');

% "Undo" the logarithmic transformation
exp_Y_train = exp(Y_train);
exp_Y0_train = exp(Y0_train);
exp_Y_valid = exp(Y_valid);
exp_Y0_valid = exp(Y0_valid);

% R2
train_R2 = corr(exp_Y_train', exp_Y0_train');
valid_R2 = corr(exp_Y_valid', exp_Y0_valid');
fprintf('Training R²:\t%g\nValidation R²:\t%g\n', ...
    train_R2, valid_R2);

%% Plot fit

%plot the predicted values of the train and validation vs. the
% true values. Use a scatter plot.  
% 1. use different colors for each set
% 2. create a figure legend with the labels for each set
% 3. create a dashed black line along the optimal distribution of the data
scatter(exp_Y0_train, exp_Y_train, 'r'); % plot train data
title("actual vs predicted amount of new daily Covid-19 cases")
hold on;
scatter(exp_Y0_valid, exp_Y_valid, 'b'); % plot validation data
% plot the line corresponding to perfect predictions
a = exp(min(Y0_all)); % low end of data range
b = exp(max(Y0_all)); % high end of data range
plot([a b], [a b], 'LineStyle','--', 'Color', 'black', 'LineWidth',2);
hold off;
xlabel("actual number of cases")
ylabel("predicted number of cases")
legend('training', 'validation')

