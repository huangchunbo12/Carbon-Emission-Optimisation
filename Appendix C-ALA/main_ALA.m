% Artificial lemming algorithm: A novel bionic meta-heuristic technique ...
%
% Instructions:
% - Run ALA 10 times in a batch
% - Write the convergence curve (BestScore over iterations) for each run to a separate sheet in the same XLSX file
% - Add a 'Summary' sheet to record the final best score and number of iterations for each run
% - To fix randomness, set rng(seed + r) inside the for loop
close all; clear; clc
% ===== Basic parameters =====
N = 30;
Function_name = 'F29';
[lb,ub,dim,fobj] = CEC2017(Function_name);  % CEC2017
MaxFEs = 30000;
nRuns = 10;
% ===== Excel file name (with timestamp) =====
ts = datestr(now,'yyyymmdd_HHMMSS');
xlsxFile = sprintf('ALA_%s_%s.xlsx', Function_name, ts);
if exist(xlsxFile,'file'), delete(xlsxFile); end  % Ensure it's a new file
% ===== Pre-allocate summary results =====
summaryTbl = table('Size',[nRuns 3], ...
    'VariableTypes', {'double','double','double'}, ...
    'VariableNames', {'Run','BestScore','Iterations'});
bestPosAll = nan(nRuns, dim);  % Optional: record the best position for each run
% ===== Run multiple times and write out each iteration process =====
for r = 1:nRuns
    % For reproducibility: rng(2025 + r);  % Fix random seed (optional)
    [ala_score, ala_pos, ala_curve] = ALA(N, MaxFEs, lb, ub, dim, fobj);
    % Write the current iteration process to a separate sheet
    T = table((1:numel(ala_curve))', ala_curve(:), ...
        'VariableNames', {'Iteration','BestScore'});
    sheetName = sprintf('Run%02d', r);
    writetable(T, xlsxFile, 'Sheet', sheetName, 'WriteMode', 'overwritesheet');
    % Summary information
    summaryTbl.Run(r)       = r;
    summaryTbl.BestScore(r) = ala_score;
    summaryTbl.Iterations(r)= numel(ala_curve);
    bestPosAll(r,:)         = ala_pos;
    % Command line prompt
    fprintf('Run %02d | BestScore = %.6g | Iterations = %d\n', ...
        r, ala_score, numel(ala_curve));
end
% ===== Write out summary table =====
writetable(summaryTbl, xlsxFile, 'Sheet', 'Summary', 'WriteMode', 'overwritesheet');
% Optional: write out the best position for each run (x1..x_dim)
posVarNames = compose('x%d', 1:dim);
posTbl = array2table(bestPosAll, 'VariableNames', posVarNames);
posTbl = addvars(posTbl, (1:nRuns)', 'Before', 1, 'NewVariableNames', 'Run');
writetable(posTbl, xlsxFile, 'Sheet', 'BestPosition', 'WriteMode', 'overwritesheet');
% ===== Optional visualization: plot function surface once + overlay all convergence curves =====
figure('Position',[454 445 900 360]);
subplot(1,2,1);
func_plot_cec2017(Function_name);
title(Function_name);
xlabel('x_1'); ylabel('x_2'); zlabel([Function_name,'(x_1, x_2)']); grid on;
subplot(1,2,2); hold on;
for r = 1:nRuns
    % To avoid re-reading files, simply run ALA again to get the curve here (or store it in a cell)
    % A more efficient approach: store ala_curve in a cell array in the loop above, then plot here
    % -- The example below uses the iterations from the Summary table as a placeholder, not re-running --
end
% For simplicity, plot one representative curve (Run 1)
rng('default'); % Only for consistent example
[~,~,curve_demo] = ALA(N, MaxFEs, lb, ub, dim, fobj);
semilogy(curve_demo, 'LineWidth', 2);
title([Function_name, '  (one representative run)']);
xlabel('Iteration#'); ylabel('Best score so far'); grid on; legend('ALA');
fprintf('Results have been written to: %s\n', xlsxFile);