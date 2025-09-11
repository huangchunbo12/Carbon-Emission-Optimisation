function [BestScore, BestPos, Convergence_curve] = PSO(N, Max_iter, lb, ub, dim, fobj, params)
% PSO: Particle Swarm Optimization in real space (for minimization)
% Outputs are consistent with ALA/GA:
%   BestScore            -- Global best fitness
%   BestPos              -- Global best position
%   Convergence_curve    -- Best fitness value per iteration (length = Max_iter)
%
% Dependency: initialization.m  (Note your existing implementation's signature is initialization(N,dim,ub,lb))
% ====== Parameters and default values ======
if nargin < 7, params = struct(); end
w_start   = getdef(params, 'w_start',   0.9);
w_end     = getdef(params, 'w_end',     0.4);
c1        = getdef(params, 'c1',        1.7);
c2        = getdef(params, 'c2',        1.7);
vmax_frac = getdef(params, 'vmax_frac', 0.2);
% ====== Standardize bounds vectors ======
if isscalar(lb), lb = lb .* ones(1,dim); end
if isscalar(ub), ub = ub .* ones(1,dim); end
range = ub - lb;
% ====== Initialize population and velocity ======
X = initialization(N, dim, ub, lb);     % Call order consistent with ALA
V = zeros(N, dim);
vmax = vmax_frac * range;                % Velocity limit (per dimension)
% ====== Initialize personal/global bests ======
pbest_pos = X;
pbest_fit = arrayfun(@(i) fobj(X(i,:)), 1:N);
[BestScore, g_idx] = min(pbest_fit);
BestPos = pbest_pos(g_idx, :);
Convergence_curve = zeros(1, Max_iter);
% ====== Iteration loop ======
for it = 1:Max_iter
    % Linear decreasing inertia weight
    w = w_start - (w_start - w_end) * (it-1) / max(1, (Max_iter-1));
    % Velocity update (global best topology)
    r1 = rand(N, dim);
    r2 = rand(N, dim);
    V  = w .* V ...
        + c1 .* r1 .* (pbest_pos - X) ...
        + c2 .* r2 .* (BestPos    - X);
    % Clamp velocity
    V = max(min(V,  vmax), -vmax);
    % Position update
    X = X + V;
    % Boundary handling (clamp if out of bounds and zero out velocity in that dimension to suppress boundary oscillations)
    [X, hitMask] = clamp_with_zero_velocity(X, lb, ub);
    V(hitMask) = 0;
    % Fitness evaluation and personal best update
    fit = arrayfun(@(i) fobj(X(i,:)), 1:N);
    better = fit < pbest_fit;
    pbest_fit(better) = fit(better);
    pbest_pos(better, :) = X(better, :);
    % Global best update
    [curBest, curIdx] = min(pbest_fit);
    if curBest < BestScore
        BestScore = curBest;
        BestPos   = pbest_pos(curIdx, :);
    end
    Convergence_curve(it) = BestScore;
end
end
% ====== Helper: Get default value ======
function v = getdef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
% ====== Helper: Clamp to bounds and return hit mask (for zeroing velocity) ======
function [Xc, hitMask] = clamp_with_zero_velocity(X, lb, ub)
Xc = X;
Xc = min(max(Xc, lb), ub);
hitMask = (Xc ~= X);       % N x dim logical matrix: indicates positions that were clamped
end