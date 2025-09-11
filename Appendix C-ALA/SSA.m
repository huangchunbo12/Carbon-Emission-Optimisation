function [BestScore, BestPos, Convergence_curve] = SSA(N, Max_iter, lb, ub, dim, fobj, params)
% SSA: Salp Swarm Algorithm (for minimization)
% Outputs are consistent with ALA/GA/PSO:
%   BestScore            -- Global best fitness
%   BestPos              -- Global best position
%   Convergence_curve    -- Best fitness value per iteration (length = Max_iter)
%
% Dependency: initialization.m  (Note your existing implementation's signature is initialization(N,dim,ub,lb))
% ===== Parameters and default values =====
if nargin < 7, params = struct(); end
leader_frac = getdef(params, 'leader_frac', 0.5);   % The first half are leaders
use_linear_c1 = true;                               % c1 update strategy (true=Gaussian-like, false=linear)
% ===== Standardize bounds vectors =====
if isscalar(lb), lb = lb .* ones(1,dim); end
if isscalar(ub), ub = ub .* ones(1,dim); end
range = ub - lb;
% ===== Initialize population =====
X   = initialization(N, dim, ub, lb);     % Same call order as ALA
fit = arrayfun(@(i) fobj(X(i,:)), 1:N);
% ===== Initialize global best (food) =====
[BestScore, idx] = min(fit);
BestPos = X(idx, :);
Convergence_curve = zeros(1, Max_iter);
n_leader = max(1, ceil(leader_frac * N));
% ===== Main loop =====
for t = 1:Max_iter
    % c1: decays from 2 to ~0 (Mirjalili's original form)
    if use_linear_c1
        c1 = 2 * exp(-(4 * (t/Max_iter))^2);
    else
        c1 = 2 - 2 * (t-1) / max(1,(Max_iter-1));
    end
    Xnew = X;
    % ---- Leaders (the first n_leader individuals) ----
    for i = 1:n_leader
        for j = 1:dim
            c2 = rand();
            c3 = rand();
            step = c1 * (range(j) * c2 + lb(j));    % Random step (scaled to the dimension's range)
            if c3 >= 0.5
                Xnew(i,j) = BestPos(j) + step;
            else
                Xnew(i,j) = BestPos(j) - step;
            end
        end
    end
    % ---- Followers (the rest) ----
    for i = n_leader+1:N
        % Use chain-like averaging update (using the just-updated previous individual)
        Xnew(i,:) = (Xnew(i-1,:) + X(i,:)) / 2;
    end
    % ---- Boundary clamping ----
    Xnew = min(max(Xnew, lb), ub);
    % ---- Evaluate and update ----
    fit_new = arrayfun(@(i) fobj(Xnew(i,:)), 1:N);
    % Individual replacement (greedy)
    improve = fit_new < fit;
    X(improve, :) = Xnew(improve, :);
    fit(improve)   = fit_new(improve);
    % Global best update
    [curBest, curIdx] = min(fit);
    if curBest < BestScore
        BestScore = curBest;
        BestPos   = X(curIdx, :);
    end
    Convergence_curve(t) = BestScore;
end
end
% ===== Helper: Get default value =====
function v = getdef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end