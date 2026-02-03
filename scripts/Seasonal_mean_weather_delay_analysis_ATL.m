%% ================================================
% Seasonal Weather–Delay Analysis at ATL
% MATLAB script version – ready to run
% Author: Nur Rahmania Ramadhani, Halmar Halide
%% ================================================
clear; clc; clf;
rng(1);

%% 1. DATA ACQUISITION
file_name = 'data/Data Variabel Meteorologi.xlsx';
T = readtable(file_name, 'PreserveVariableNames', true);

% Rename column if needed
if any(strcmp(T.Properties.VariableNames, 'Number of Delays'))
    T.Properties.VariableNames{'Number of Delays'} = 'NumberOfDelays';
end

%% 2. VARIABLES
delay_var = {'NumberOfDelays'};
weather_vars = {'Temperature', 'Pressure', 'Precipitation', ...
                'Relative Humidity (RH)', 'Vector Wind Shear'};
all_vars = [delay_var, weather_vars];
units = {'Counts', '°C', 'kPa', 'mm', '%', 's^{-1}'};

%% 3. SEASONAL CLASSIFICATION
season_labels = {'DJF', 'MAM', 'JJA', 'SON'};
T.Season = repmat("", height(T),1);

for i = 1:height(T)
    m = T.Month(i);
    if m==12 || m==1 || m==2
        T.Season(i) = "DJF";
    elseif m>=3 && m<=5
        T.Season(i) = "MAM";
    elseif m>=6 && m<=8
        T.Season(i) = "JJA";
    else
        T.Season(i) = "SON";
    end
end
T.Season = categorical(T.Season, season_labels);

%% 4. STATISTICAL COMPUTATION
season_stats = struct();
for v = 1:length(all_vars)
    var_name = all_vars{v};
    means = zeros(1,4);
    stds = zeros(1,4);
    for s = 1:4
        d = T{T.Season == season_labels{s}, var_name};
        d = d(~isnan(d));
        means(s) = mean(d);
        stds(s) = std(d);
    end
    safe_name = matlab.lang.makeValidName(var_name);
    season_stats.(safe_name).mean = means;
    season_stats.(safe_name).std = stds;
end

%% 5. VISUALIZATION COLORS
colors = [
    0.30 0.55 0.85; % DJF Blue
    0.35 0.70 0.40; % MAM Green
    0.95 0.60 0.25; % JJA Orange
    0.85 0.33 0.33  % SON Red
];

%% 6. GRAPHICAL REPRESENTATION
output_dir = 'outputs';
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

figure('Name','Seasonal Analysis at ATL','NumberTitle','off','Color','w');

for v = 1:length(all_vars)
    subplot(2,3,v);
    var_name = all_vars{v};
    safe_name = matlab.lang.makeValidName(var_name);
    stats = season_stats.(safe_name);

    % Bar Plot
    b = bar(stats.mean, 'FaceColor','flat', 'BarWidth',0.65);
    for s = 1:4
        b.CData(s,:) = colors(s,:);
    end
    
    % Error bars
    hold on;
    errorbar(1:4, stats.mean, stats.std, 'k.', 'LineWidth',1.2);
    hold off;

    set(gca, 'XTick', 1:4, 'XTickLabel', season_labels, ...
        'FontSize',10, 'FontName','Times New Roman','LineWidth',1);
    
    xlabel('Season','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    if strcmp(var_name,'NumberOfDelays')
        ylabel('Number of Delays','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    else
        ylabel(sprintf('%s (%s)',var_name,units{v}),'FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    end

    box on; grid on;
    ylim([0, max(stats.mean + stats.std)*1.3]);
end

sgtitle('Seasonal Mean and Variability of Flight Delays and Meteorological Conditions at ATL', ...
    'FontSize',14,'FontWeight','bold','FontName','Times New Roman');

saveas(gcf, fullfile(output_dir,'Seasonal_Analysis_ATL.png'));

%% 7. STATISTICAL ANALYSIS (ONE-WAY ANOVA)
anova_table = cell(length(all_vars)+1,5);
anova_table(1,:) = {'Variable','F-Stat','p-value','Seasonal Effect','Significant'};
for v = 1:length(all_vars)
    var_name = all_vars{v};
    y = T{:,var_name};
    g = T.Season;

    valid_idx = ~isnan(y);
    [p,tbl,~] = anova1(y(valid_idx), g(valid_idx),'off');
    f_stat = tbl{2,5};
    
    if p < 0.001
        p_str = '<0.001';
    else
        p_str = sprintf('%.4f',p);
    end
    
    if p<0.05
        effect = 'Significant';
        sig_flag = 'Yes';
    else
        effect = 'Not Significant';
        sig_flag = 'No';
    end
    
    anova_table(v+1,:) = {var_name,f_stat,p_str,effect,sig_flag};
end

% Display ANOVA table
disp('--- One-Way ANOVA Statistical Summary ---')
disp(anova_table)

% Save ANOVA table
anova_filename = fullfile(output_dir,'ANOVA_Summary.csv');
cell2csv(anova_filename,anova_table); % gunakan fungsi cell2csv di bawah
disp(['ANOVA summary saved to ',anova_filename]);

%% 8. Helper function: cell2csv
function cell2csv(filename, cellArray)
    fid = fopen(filename,'w');
    for i=1:size(cellArray,1)
        fprintf(fid,'%s',cellArray{i,1});
        for j=2:size(cellArray,2)
            if isnumeric(cellArray{i,j})
                fprintf(fid,',%g',cellArray{i,j});
            else
                fprintf(fid,',%s',cellArray{i,j});
            end
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
end
