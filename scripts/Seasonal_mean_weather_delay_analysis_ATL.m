clear; clc; clf;
rng(1); 

%% 1. DATA ACQUISITION
% Loading meteorological and operational dataset
file_name = 'Data Variabel Meteorologi.xlsx';
T = readtable(file_name, 'PreserveVariableNames', true);

% Column synchronization for flight delays
if any(strcmp(T.Properties.VariableNames, 'Number of Delays'))
    T.Properties.VariableNames{'Number of Delays'} = 'NumberOfDelays';
end

%% 2. EXPERIMENTAL CONFIGURATION
% Sequence: Flight Delays followed by Meteorological parameters
delay_var = {'NumberOfDelays'};
weather_vars = {'Temperature', 'Pressure', 'Precipitation', ...
                'Relative Humidity (RH)', 'Vector Wind Shear'};
all_vars = [delay_var, weather_vars];

% Measurement units (Pressure in kPa, Shear in s^-1)
units = {'Counts', '°C', 'kPa', 'mm', '%', 's^{-1}'};

%% 3. SEASONAL CLASSIFICATION (DJF, MAM, JJA, SON)
season_labels = {'DJF', 'MAM', 'JJA', 'SON'};
season_bins = [0 3 6 9 12];

% Grouping months into meteorological seasons (Dec-Jan-Feb = DJF)
if ismember('Month', T.Properties.VariableNames)
    m_val = T.Month;
else
    m_val = T.month;
end

season_indices = mod(m_val, 12); 
T.Season = discretize(season_indices, season_bins, 'Categorical', season_labels);

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
% Mapping: DJF (Blue), MAM (Green), JJA (Orange), SON (Red)
colors = [
    0.30 0.55 0.85; % Winter Blue (DJF)
    0.35 0.70 0.40; % Spring Green (MAM)
    0.95 0.60 0.25; % Summer Orange (JJA)
    0.85 0.33 0.33  % Autumn Red (SON)
];

%% 6. GRAPHICAL REPRESENTATION
figure('Name','Seasonal Analysis at ATL','NumberTitle','off','Color','w');

for v = 1:length(all_vars)
    subplot(2,3,v);
    var_name = all_vars{v};
    safe_name = matlab.lang.makeValidName(var_name);
    stats = season_stats.(safe_name);
    
    % Bar Plot Generation
    b = bar(stats.mean, 'FaceColor','flat', 'BarWidth',0.65);
    for s = 1:4
        b.CData(s,:) = colors(s,:);
    end
    
    % Error Bar Integration (Standard Deviation)
    hold on;
    errorbar(1:4, stats.mean, stats.std, 'k.', 'LineWidth', 1.2);
    hold off;

    % Axis and Typography Settings (Times New Roman)
    set(gca, 'XTick', 1:4, 'XTickLabel', season_labels, ...
             'FontSize', 10, 'FontName', 'Times New Roman', 'LineWidth', 1);
    
    xlabel('Season', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

    % Y-Axis Labeling (Variables identified via Y-label)
    if strcmp(var_name, 'NumberOfDelays')
        ylabel('Number of Delays', 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    else
        ylabel(sprintf('%s (%s)', var_name, units{v}), 'FontSize', 11, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    end

    box on; grid on;
    ylim([0, max(stats.mean + stats.std)*1.3]);
end

% Global Figure Title
sgtitle('Seasonal Mean and Variability of Flight Delays and Meteorological Conditions at ATL', ...
        'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

%% 7. STATISTICAL ANALYSIS (ONE-WAY ANOVA SUMMARY)
fprintf('\n--- One-Way ANOVA Statistical Summary ---\n');
fprintf('%-22s | %-8s | %-10s | %-15s\n', 'Variable', 'F-Stat', 'p-value', 'Seasonal Effect');
fprintf('%s\n', repmat('-', 1, 65));

for v = 1:length(all_vars)
    var_name = all_vars{v};
    y = T{:, var_name};
    g = T.Season;
    
    % Data filtering for non-finite values
    valid_idx = ~isnan(y);
    [p, tbl, ~] = anova1(y(valid_idx), g(valid_idx), 'off');
    f_stat = tbl{2,5};
    
    % Format p-value for academic reporting
    if p < 0.001
        p_str = '< 0.001';
    else
        p_str = sprintf('%.4f', p);
    end
    
    % Determine Significance status
    if p < 0.05
        effect = 'Significant';
    else
        effect = 'Not Significant';
    end
    
    fprintf('%-22s | %8.2f | %-10s | %-15s\n', ...
            var_name, f_stat, p_str, effect);
end