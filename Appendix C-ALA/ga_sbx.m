function [c1, c2] = ga_sbx(p1, p2, lb, ub, eta_c, pc)
% Real-coded SBX crossover (classic Deb implementation)
% pc: crossover probability; eta_c: distribution index
d = numel(p1);
if rand > pc
    c1 = p1; c2 = p2;
    return;
end
c1 = zeros(1,d);
c2 = zeros(1,d);
for j = 1:d
    x1 = p1(j); x2 = p2(j);
    L  = min(x1,x2); U = max(x1,x2);
    if abs(x1 - x2) > eps
        % Calculate based on Deb's SBX formula
        lower = lb(j); upper = ub(j);
        randu = rand;
        beta = 1 + (2*(L - lower)/(U - L));
        alpha = 2 - beta^(-(eta_c+1));
        if randu <= 1/alpha
            betaq = (randu*alpha)^(1/(eta_c+1));
        else
            betaq = (1/(2 - randu*alpha))^(1/(eta_c+1));
        end
        c1j = 0.5*((x1 + x2) - betaq*(U - L));
        beta = 1 + (2*(upper - U)/(U - L));
        alpha = 2 - beta^(-(eta_c+1));
        if randu <= 1/alpha
            betaq = (randu*alpha)^(1/(eta_c+1));
        else
            betaq = (1/(2 - randu*alpha))^(1/(eta_c+1));
        end
        c2j = 0.5*((x1 + x2) + betaq*(U - L));
        % Boundary handling
        c1j = min(max(c1j, lower), upper);
        c2j = min(max(c2j, lower), upper);
    else
        c1j = x1; c2j = x2;
    end
    % Randomly swap
    if rand < 0.5
        c1(j) = c1j; c2(j) = c2j;
    else
        c1(j) = c2j; c2(j) = c1j;
    end
end
end