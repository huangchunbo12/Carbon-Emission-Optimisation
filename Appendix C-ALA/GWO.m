function [BestScore, BestPos, Convergence_curve] = GWO(N, Max_iter, lb, ub, dim, fobj, params)
% GWO: Grey Wolf Optimizer (for minimization)
% Outputs aligned with ALA/GA/PSO/SSA/WOA:
%   BestScore            -- Global best fitness (Alpha)
%   BestPos              -- Global best position (Alpha position)
%   Convergence_curve    -- Best fitness value per iteration (length = Max_iter)
%
% Dependency: initialization.m  (Note your implementation signature: initialization(N, dim, ub, lb))
% ===== Parameter parsing and default values (extensible) =====
if nargin < 7, params = struct(); end
a_scheme = getdef(params, 'a_scheme', 'linear');  % Decay strategy: 'linear' or 'exp'
use_greedy = getdef(params, 'use_greedy', false); % Whether to use greedy replacement for individuals (default is full population replacement)
% ===== Standardize bounds vectors =====
if isscalar(lb), lb = lb .* ones(1,dim); end
if isscalar(ub), ub = ub .* ones(1,dim); end
% ===== Initialize population and fitness =====
X   = initialization(N, dim, ub, lb);     % Call order consistent with ALA
fit = arrayfun(@(i) fobj(X(i,:)), 1:N);
% ===== Find Alpha / Beta / Delta =====
[alpha_score, a_idx] = min(fit);
alpha_pos = X(a_idx, :);
% Temporarily set to infinity to find the second and third best
tmp_fit = fit; tmp_fit(a_idx) = inf;
[beta_score,  b_idx] = min(tmp_fit);
beta_pos  = X(b_idx, :);
tmp_fit(b_idx) = inf;
[delta_score, d_idx] = min(tmp_fit);
delta_pos = X(d_idx, :);
BestScore = alpha_score;
BestPos   = alpha_pos;
Convergence_curve = zeros(1, Max_iter);
% ===== Main loop =====
for t = 1:Max_iter
    % 'a' decays from 2 to 0
    switch lower(a_scheme)
        case 'exp'
            a = 2 * exp(-(4 * (t/Max_iter))^2);
        otherwise % 'linear'
            a = 2 - 2 * (t-1) / max(1,(Max_iter-1));
    end
    Xnew = X;
    % -- Position update (for each agent, each dimension) --
    for i = 1:N
        for j = 1:dim
            % Alpha guidance
            r1 = rand(); r2 = rand();
            A1 = 2*a*r1 - a;
            C1 = 2*r2;
            D_alpha = abs(C1*alpha_pos(j) - X(i,j));
            X1 = alpha_pos(j) - A1*D_alpha;
            % Beta guidance
            r1 = rand(); r2 = rand();
            A2 = 2*a*r1 - a;
            C2 = 2*r2;
            D_beta = abs(C2*beta_pos(j) - X(i,j));
            X2 = beta_pos(j) - A2*D_beta;
            % Delta guidance
            r1 = rand(); r2 = rand();
            A3 = 2*a*r1 - a;
            C3 = 2*r2;
            D_delta = abs(C3*delta_pos(j) - X(i,j));
            X3 = delta_pos(j) - A3*D_delta;
            % Aggregate
            Xnew(i,j) = (X1 + X2 + X3) / 3;
        end
    end
    % Boundary clamping
    Xnew = min(max(Xnew, lb), ub);
    % Evaluate new generation
    fnew = arrayfun(@(i) fobj(Xnew(i,:)), 1:N);
    % Individual replacement strategy
    if use_greedy
        improve = fnew < fit;
        X(improve,:) = Xnew(improve,:);
        fit(improve) = fnew(improve);
    else
        X = Xnew;
        fit = fnew;
    end
    % Update Alpha / Beta / Delta
    [alpha_score, a_idx] = min(fit);
    alpha_pos = X(a_idx, :);
    tmp_fit = fit; tmp_fit(a_idx) = inf;
    [beta_score,  b_idx] = min(tmp_fit);
    beta_pos  = X(b_idx, :);
    tmp_fit(b_idx) = inf;
    [delta_score, d_idx] = min(tmp_fit);
    delta_pos = X(d_idx, :);
    % Record global best
    if alpha_score < BestScore
        BestScore = alpha_score;
        BestPos   = alpha_pos;
    end
    Convergence_curve(t) = BestScore;
end
end
% ===== Helper: Default parameter getter =====
function v = getdef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end