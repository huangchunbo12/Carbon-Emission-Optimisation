% ===== Run GWO multiple times and write each iteration process to Excel =====
close all; clear; clc
% ===== Basic parameters =====
N = 30;
Function_name = 'F29';
[lb,ub,dim,fobj] = CEC2017(Function_name);    % CEC2017
MaxFEs  = 30000;                               % Evaluation budget
% GWO: N evaluations for initialization; ~N evaluations per iteration -> N + N*Max_iter <= MaxFEs
Max_iter = max(1, floor((MaxFEs - N) / N));
nRuns   = 10;
% (Optional) GWO parameters; leave empty to use defaults
% a_scheme: 'linear' or 'exp'; use_greedy: true/false
params = struct('a_scheme','linear','use_greedy',false);
% ===== Excel file name (with timestamp) =====
ts = datestr(now,'yyyymmdd_HHMMSS');
xlsxFile = sprintf('GWO_%s_%s.xlsx', Function_name, ts);
if exist(xlsxFile,'file'), delete(xlsxFile); end  % Ensure it's a new file
% ===== Pre-allocate summary results =====
summaryTbl = table('Size',[nRuns 3], ...
    'VariableTypes', {'double','double','double'}, ...
    'VariableNames', {'Run','BestScore','Iterations'});
bestPosAll = nan(nRuns, dim);   % Record the best position for each run
curves     = cell(nRuns,1);     % Save convergence curves for plotting
% ===== Run multiple times and write out each iteration process =====
for r = 1:nRuns
    % For reproducibility: rng(2025 + r);
    [gwo_score, gwo_pos, gwo_curve] = GWO(N, Max_iter, lb, ub, dim, fobj, params);
    curves{r} = gwo_curve(:)';   % Store the curve
    % Write the current iteration process to a separate sheet
    T = table((1:numel(gwo_curve))', gwo_curve(:), ...
        'VariableNames', {'Iteration','BestScore'});
    sheetName = sprintf('Run%02d', r);
    writetable(T, xlsxFile, 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    % Summary information
    summaryTbl.Run(r)        = r;
    summaryTbl.BestScore(r)  = gwo_score;
    summaryTbl.Iterations(r) = numel(gwo_curve);
    bestPosAll(r,:)          = gwo_pos;
    fprintf('Run %02d | BestScore = %.6g | Iterations = %d\n', ...
        r, gwo_score, numel(gwo_curve));
end
% ===== Write out summary table =====
writetable(summaryTbl, xlsxFile, 'Sheet', 'Summary', 'WriteMode', 'overwritesheet');
% Write out the best position for each run (x1..x_dim)
posVarNames = compose('x%d', 1:dim);
posTbl = array2table(bestPosAll, 'VariableNames', posVarNames);
posTbl = addvars(posTbl, (1:nRuns)', 'Before', 1, 'NewVariableNames', 'Run');
writetable(posTbl, xlsxFile, 'Sheet', 'BestPosition', 'WriteMode', 'overwritesheet');
% (Optional) Record GWO parameters
paramNames = fieldnames(params);
paramVals  = struct2cell(params);
paramTbl   = table(paramNames, paramVals, 'VariableNames', {'Param','Value'});
writetable(paramTbl, xlsxFile, 'Sheet', 'GWO_Params', 'WriteMode', 'overwritesheet');
% ===== Visualization: function surface + overlay all GWO convergence curves =====
figure('Position',[454 445 900 360]);
subplot(1,2,1);
func_plot_cec2017(Function_name);
title(Function_name);
xlabel('x_1'); ylabel('x_2'); zlabel([Function_name,'(x_1, x_2)']); grid on;
subplot(1,2,2); hold on;
for r = 1:nRuns
    semilogy(curves{r}, 'LineWidth', 1);
end
title([Function_name, '  (GWO, ', num2str(nRuns), ' runs)']);
xlabel('Iteration#'); ylabel('Best score so far'); grid on; box on;
fprintf('Results have been written to: %s\n', xlsxFile);