function X = ga_initialization(N, dim, lb, ub)
% 连续空间均匀初始化，支持 lb/ub 为标量或 1×dim 向量
if isscalar(lb), lb = lb .* ones(1,dim); end
if isscalar(ub), ub = ub .* ones(1,dim); end
X = rand(N, dim) .* (ub - lb) + lb;
end
