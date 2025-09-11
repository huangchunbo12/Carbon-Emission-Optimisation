function [Leader_score, Leader_pos, Convergence_curve] = WOA(N, Max_iter, lb, ub, dim, fobj, params)
% WOA: Whale Optimization Algorithm (for minimization)
% Outputs are consistent with ALA/GA/PSO:
%   Leader_score         -- Global best fitness
%   Leader_pos           -- Global best position
%   Convergence_curve    -- Best fitness value per iteration (length = Max_iter)
%
% Dependency: initialization.m  (Note your existing implementation's signature is initialization(N,dim,ub,lb))
if nargin < 7, params = struct(); end
b = getdef(params, 'b', 1.0);   % Parameter for spiral shape, b=1 is common
% ===== Standardize bounds vectors =====
if isscalar(lb), lb = lb .* ones(1,dim); end
if isscalar(ub), ub = ub .* ones(1,dim); end
% ===== Initialize population =====
X = initialization(N, dim, ub, lb);   % Call order consistent with ALA
fit = arrayfun(@(i) fobj(X(i,:)), 1:N);
% ===== Initialize Leader (global best) =====
[Leader_score, idx] = min(fit);
Leader_pos = X(idx, :);
Convergence_curve = zeros(1, Max_iter);
% ===== Main loop =====
for t = 1:Max_iter
    a = 2 - 2 * (t-1) / max(1, (Max_iter-1)); % Linearly decreases from 2 to 0
    for i = 1:N
        r1 = rand();
        r2 = rand();
        A  = 2*a*r1 - a;     % Controls shrinking-expanding
        C  = 2*r2;           % Emphasizes the position of the leader or a random agent
        p  = rand();         % Decision (encircling/spiral) probability
        l  = (rand()*2) - 1; % Random number in [-1,1] for spiral
        if p < 0.5
            % -- Encircling prey phase -- (using leader or a random agent)
            if abs(A) < 1
                % Exploit using Leader
                D_Leader = abs(C.*Leader_pos - X(i,:));
                Xnew = Leader_pos - A.*D_Leader;
            else
                % Exploration phase: select a random agent
                rand_idx = randi(N);
                X_rand   = X(rand_idx, :);
                D_rand   = abs(C.*X_rand - X(i,:));
                Xnew     = X_rand - A.*D_rand;
            end
        else
            % -- Bubble-net attacking (spiral update) --
            D_Leader = abs(Leader_pos - X(i,:));
            Xnew = D_Leader .* exp(b*l) .* cos(2*pi*l) + Leader_pos;
        end
        % Boundary clamping
        Xnew = min(max(Xnew, lb), ub);
        % Evaluate and update individual
        fnew = fobj(Xnew);
        if fnew < fit(i)
            X(i,:) = Xnew;
            fit(i) = fnew;
        end
        % Update Leader
        if fit(i) < Leader_score
            Leader_score = fit(i);
            Leader_pos   = X(i,:);
        end
    end
    Convergence_curve(t) = Leader_score;
end
end
% ===== Helper: Get default value =====
function v = getdef(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end