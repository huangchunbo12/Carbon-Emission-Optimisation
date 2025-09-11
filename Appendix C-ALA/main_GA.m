% ===== Run GA multiple times and write each iteration process to Excel =====
close all; clear; clc
% ===== Basic parameters =====
N = 30;
Function_name = 'F29';
[lb,ub,dim,fobj] = CEC2017(Function_name);    % CEC2017
MaxFEs  = 30000;                               % Evaluation budget (consistent with original ALA)
Max_iter = max(1, floor(MaxFEs / N));          % Map evaluation budget to GA iterations
nRuns   = 10;
% (Optional) GA parameters, can also be left empty to use GA's defaults
params = struct('pc',0.9,'pm',1/dim,'eta_c',15,'eta_m',20,'tour_k',2,'elitism',true);
% ===== Excel file name (with timestamp) =====
ts = datestr(now,'yyyymmdd_HHMMSS');
xlsxFile = sprintf('GA_%s_%s.xlsx', Function_name, ts);
if exist(xlsxFile,'file'), delete(xlsxFile); end  % Ensure it's a new file
% ===== Pre-allocate summary results =====
summaryTbl = table('Size',[nRuns 3], ...
    'VariableTypes', {'double','double','double'}, ...
    'VariableNames', {'Run','BestScore','Iterations'});
bestPosAll = nan(nRuns, dim);     % Record the best position for each run
curves     = cell(nRuns,1);       % Save convergence curves for plotting
% ===== Run multiple times and write out each iteration process =====
for r = 1:nRuns
    % For reproducibility: rng(2025 + r);
    [ga_score, ga_pos, ga_curve] = GA(N, Max_iter, lb, ub, dim, fobj, params);
    curves{r} = ga_curve(:)';   % Store the curve
    % Write the current iteration process to a separate sheet
    T = table((1:numel(ga_curve))', ga_curve(:), ...
        'VariableNames', {'Iteration','BestScore'});
    sheetName = sprintf('Run%02d', r);
    writetable(T, xlsxFile, 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    % Summary information
    summaryTbl.Run(r)        = r;
    summaryTbl.BestScore(r)  = ga_score;
    summaryTbl.Iterations(r) = numel(ga_curve);
    bestPosAll(r,:)          = ga_pos;
    fprintf('Run %02d | BestScore = %.6g | Iterations = %d\n', ...
        r, ga_score, numel(ga_curve));
end
% ===== Write out summary table =====
writetable(summaryTbl, xlsxFile, 'Sheet', 'Summary', 'WriteMode', 'overwritesheet');
% Write out the best position for each run (x1..x_dim)
posVarNames = compose('x%d', 1:dim);
posTbl = array2table(bestPosAll, 'VariableNames', posVarNames);
posTbl = addvars(posTbl, (1:nRuns)', 'Before', 1, 'NewVariableNames', 'Run');
writetable(posTbl, xlsxFile, 'Sheet', 'BestPosition', 'WriteMode', 'overwritesheet');
% (Optional) Record GA parameters
paramNames = fieldnames(params);
paramVals  = struct2cell(params);
paramTbl   = table(paramNames, paramVals, 'VariableNames', {'Param','Value'});
writetable(paramTbl, xlsxFile, 'Sheet', 'GA_Params', 'WriteMode', 'overwritesheet');
% ===== Visualization: function surface + overlay all GA convergence curves =====
figure('Position',[454 445 900 360]);
subplot(1,2,1);
func_plot_cec2017(Function_name);
title(Function_name);
xlabel('x_1'); ylabel('x_2'); zlabel([Function_name,'(x_1, x_2)']); grid on;
subplot(1,2,2); hold on;
for r = 1:nRuns
    semilogy(curves{r}, 'LineWidth', 1);
end
title([Function_name, '  (GA, ', num2str(nRuns), ' runs)']);
xlabel('Iteration#'); ylabel('Best score so far');
grid on; box on;
fprintf('Results have been written to: %s\n', xlsxFile);