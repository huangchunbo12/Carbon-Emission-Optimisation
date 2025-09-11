% Batch export search space PNG images for several CEC2017 functions
close all; clc;
funcList = {'F1','F5','F12','F16','F22','F29'};
outDir   = 'SearchSpace_PNG';
if ~exist(outDir, 'dir'), mkdir(outDir); end
for k = 1:numel(funcList)
    Function_name = funcList{k};
    % Create a hidden figure and set default font to Times New Roman
    fig = figure('Position',[454 445 400 360], 'Color','w', 'Visible','off');
    set(fig, 'DefaultAxesFontName','Times New Roman');
    set(fig, 'DefaultTextFontName','Times New Roman');
    % Plot the search space
    ax = axes('Parent',fig);
    func_plot_cec2017(Function_name);
    % Title and axis labels (keeping your original style)
    title('SearchSpace', 'FontName','Times New Roman');
    xlabel('x_1', 'FontName','Times New Roman');
    ylabel('x_2', 'FontName','Times New Roman');
    zlabel([Function_name,'( x_1 , x_2 )'], 'FontName','Times New Roman');
    grid on; axis tight; set(ax,'FontName','Times New Roman');
    % Export PNG (300 dpi)
    pngPath = fullfile(outDir, sprintf('SearchSpace_%s.png', Function_name));
    exportgraphics(ax, pngPath, 'Resolution', 300);
    close(fig);
    fprintf('Saved: %s\n', pngPath);
end