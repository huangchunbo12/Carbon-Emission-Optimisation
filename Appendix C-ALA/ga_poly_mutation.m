function y = ga_poly_mutation(x, lb, ub, eta_m, pm)
% Polynomial mutation (gene-by-gene, with mutation probability pm)
d = numel(x);
y = x;
for j = 1:d
    if rand < pm
        yl = lb(j); yu = ub(j);
        if yl == yu, continue; end
        delta1 = (y(j) - yl) / (yu - yl);
        delta2 = (yu - y(j)) / (yu - yl);
        randu  = rand;
        mut_pow = 1/(eta_m + 1);
        if randu <= 0.5
            xy = 1 - delta1;
            val = 2*randu + (1 - 2*randu)*(xy^(eta_m + 1));
            deltaq = val^mut_pow - 1;
        else
            xy = 1 - delta2;
            val = 2*(1 - randu) + 2*(randu - 0.5)*(xy^(eta_m + 1));
            deltaq = 1 - val^mut_pow;
        end
        y(j) = y(j) + deltaq*(yu - yl);
        y(j) = min(max(y(j), yl), yu);
    end
end
end