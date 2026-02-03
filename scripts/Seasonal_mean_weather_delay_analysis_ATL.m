%% ===============================================
% Seasonal Weather-Delay Analysis at ATL
% MATLAB Script: seasonal_weather_delay_analysis.m
% Author: Nur Rahmania Ramadhani
% ===============================================

clear; clc; clf;
rng(1); 

%% 1. DATA ACQUISITION
file_name = 'Data Variabel Meteorologi.xlsx';
T = readtable(file_name, 'PreserveVariableNames', true);

% Rename column
if any(strcmp(T.Properties.VariableNames, 'Number of Delays'))
    T.Properties.VariableNames{'Number of Delays'} = 'NumberOfDelays';
end

%% 2. CONFIGURATION
delay_var = {'NumberOfDelays'};
weather_vars = {'Temperature','Pressure','Precipitation','Relative Humidity (RH)','Vector Wind Shear'};
all_vars = [delay_var, weather_vars];
units = {'Counts','°C','kPa','mm','%','s^{-1}'};

%% 3. SEASONAL CLASSIFICATION
season_labels = {'DJF','MAM','JJA','SON'};

if ismember('Month', T.Properties.VariableNames)
    m_val = T.Month;
else
    m_val = T.month;
end

% Mapping months to seasons
T.Season = categorical(arrayfun(@(m) ...
    seasonal_label(m), m_val));

%% 4. STATISTICAL COMPUTATION
season_stats = struct();
for v = 1:length(all_vars)
    var_name = all_vars{v};
    means = zeros(1,4);
    stds  = zeros(1,4);
    for s = 1:4
        d = T{T.Season == season_labels{s}, var_name};
        d = d(~isnan(d));
        means(s) = mean(d);
        stds(s)  = std(d);
    end
    safe_name = matlab.lang.makeValidName(var_name);
    season_stats.(safe_name).mean = means;
    season_stats.(safe_name).std  = stds;
end

%% 5. COLORS
colors = [
    0.30 0.55 0.85; % DJF
    0.35 0.70 0.40; % MAM
    0.95 0.60 0.25; % JJA
    0.85 0.33 0.33  % SON
];

%% 6. PLOTTING
fig = figure('Name','Seasonal Analysis at ATL','Color','w');

for v = 1:length(all_vars)
    subplot(2,3,v);
    var_name = all_vars{v};
    safe_name = matlab.lang.makeValidName(var_name);
    stats = season_stats.(safe_name);

    % Bar plot
    b = bar(stats.mean,'FaceColor','flat','BarWidth',0.65);
    for s = 1:4
        b.CData(s,:) = colors(s,:);
    end
    
    hold on
    % Error bars
    errorbar(1:4, stats.mean, stats.std, 'k.', 'LineWidth',1.2);

    % Labels above bars
    for s = 1:4
        text(s, stats.mean(s)+stats.std(s)*0.05, sprintf('%.1f',stats.mean(s)), ...
            'HorizontalAlignment','center','FontSize',9,'FontName','Times New Roman');
    end
    hold off

    % Axes settings
    set(gca,'XTick',1:4,'XTickLabel',season_labels, 'FontSize',10,'FontName','Times New Roman','LineWidth',1);
    xlabel('Season','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    if strcmp(var_name,'NumberOfDelays')
        ylabel('Number of Delays','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    else
        ylabel(sprintf('%s (%s)',var_name,units{v}),'FontSize',11,'FontWeight','bold','FontName','Times New Roman');
    end
    box on; grid on;
    ylim([0,max(stats.mean+stats.std)*1.3])
end

sgtitle('Seasonal Mean and Variability of Flight Delays and Meteorological Conditions at ATL', ...
        'FontSize',14,'FontWeight','bold','FontName','Times New Roman');

% Save figure
if ~exist('scripts','dir'); mkdir('scripts'); end
saveas(fig,'scripts/Seasonal_Analysis_ATL.png');

%% 7. ANOVA ANALYSIS
fprintf('\n--- One-Way ANOVA Summary ---\n');
fprintf('%-22s | %-8s | %-10s | %-15s\n','Variable','F-Stat','p-value','Seasonal Effect');
fprintf('%s\n', repmat('-',1,65));

var_names = all_vars;
f_stats = zeros(size(var_names));
p_values = zeros(size(var_names));
effect   = strings(size(var_names));

anova_table = table('Size',[length(var_names),4], ...
    'VariableTypes',{'string','double','string','string'}, ...
    'VariableNames',{'Variable','F_Stat','p_value','Significance'});

for v = 1:length(all_vars)
    var_name = all_vars{v};
    y = T{:,var_name};
    g = T.Season;
    valid_idx = ~isnan(y);
    [p,tbl,~] = anova1(y(valid_idx),g(valid_idx),'off');
    f_stat = tbl{2,5};
    if p<0.001; p_str='<0.001'; else; p_str=sprintf('%.4f',p); end
    if p<0.05; sig='Significant'; else; sig='Not Significant'; end

    fprintf('%-22s | %8.2f | %-10s | %-15s\n',var_name,f_stat,p_str,sig);

    % Save to table
    anova_table.Variable(v) = string(var_name);
    anova_table.F_Stat(v) = f_stat;
    anova_table.p_value(v) = p_str;
    anova_table.Significance(v) = sig;
end

% Save ANOVA summary CSV
writetable(anova_table,'scripts/ANOVA_Summary.csv');

%% ============================
% Helper function: month → season
function label = seasonal_label(m)
    if ismember(m,[12,1,2]); label='DJF';
    elseif ismember(m,3:5); label='MAM';
    elseif ismember(m,6:8); label='JJA';
    else; label='SON'; end
end
