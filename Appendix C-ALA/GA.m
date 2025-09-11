function [BestScore, BestPos, Convergence_curve] = GA(N, Max_iter, lb, ub, dim, fobj, params)
    % ===== Parameter initialization and default values =====
    if nargin < 7, params = struct(); end
    pc      = getfield_def(params, 'pc',      0.9);
    pm      = getfield_def(params, 'pm',      1/dim);
    eta_c   = getfield_def(params, 'eta_c',   15);
    eta_m   = getfield_def(params, 'eta_m',   20);
    tour_k  = getfield_def(params, 'tour_k',  2);
    elitism = getfield_def(params, 'elitism', true);
    % ==== Unify bounds shape (critical fix) ====
    if isscalar(lb), lb = lb*ones(1,dim); end
    if isscalar(ub), ub = ub*ones(1,dim); end
    % (Optional) safety check
    % assert(numel(lb)==dim && numel(ub)==dim, 'lb/ub dimensions must be 1xdim');
% ===== Initialize population =====
X = ga_initialization(N, dim, lb, ub);         % N x dim
fitness = arrayfun(@(i) fobj(X(i,:)), 1:N);    % 1 x N
[BestScore, bestIdx] = min(fitness);
BestPos = X(bestIdx, :);
Convergence_curve = zeros(1, Max_iter);
% ===== Main loop =====
for it = 1:Max_iter
    % ---- Generate offspring ----
    Offspring = zeros(N, dim);
    offCount  = 0;
    while offCount < N
        % Tournament selection for two parents
        p1 = tournament_select(fitness, tour_k);
        p2 = tournament_select(fitness, tour_k);
        parent1 = X(p1, :);
        parent2 = X(p2, :);
        % SBX crossover (produces 2 offspring)
        [c1, c2] = ga_sbx(parent1, parent2, lb, ub, eta_c, pc);
        % Polynomial mutation
        c1 = ga_poly_mutation(c1, lb, ub, eta_m, pm);
        c2 = ga_poly_mutation(c2, lb, ub, eta_m, pm);
        % Boundary handling
        c1 = min(max(c1, lb), ub);
        c2 = min(max(c2, lb), ub);
        % Fill offspring population
        offCount = offCount + 1;
        Offspring(offCount, :) = c1;
        if offCount < N
            offCount = offCount + 1;
            Offspring(offCount, :) = c2;
        end
    end
    % ---- Evaluate offspring ----
    offFit = arrayfun(@(i) fobj(Offspring(i,:)), 1:N);
    % ---- Elitism: transfer current best to next generation ----
    if elitism
        [~, worstIdx] = max(offFit);
        Offspring(worstIdx, :) = BestPos;
        offFit(worstIdx)       = fobj(BestPos);
    end
    % ---- Update population ----
    X = Offspring;
    fitness = offFit;
    % ---- Update global best ----
    [curBest, curIdx] = min(fitness);
    if curBest < BestScore
        BestScore = curBest;
        BestPos   = X(curIdx, :);
    end
    Convergence_curve(it) = BestScore;
end
end
% ========== Helper function: default parameter getter ==========
function v = getfield_def(s, f, default_v)
if isfield(s, f), v = s.(f); else, v = default_v; end
end
% ========== Helper function: tournament selection (minimization) ==========
function idx = tournament_select(fitness, k)
N = numel(fitness);
cands = randi(N, 1, k);
[~, bestLocal] = min(fitness(cands));
idx = cands(bestLocal);
end