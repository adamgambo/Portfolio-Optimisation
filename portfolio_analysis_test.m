%% ============================================================================ test
% PORTFOLIO ANALYSIS AND OPTIMIZATION WITH BITCOIN
% ============================================================================
% Research Question: What is the optimal allocation of Bitcoin in a retail 
% investor's portfolio consisting of stocks, bonds, and Bitcoin using a 
% risk-based approach?
%
% Inputs: 
%   - bonds_data.csv (Date, Open, High, Low, Close, Volume)
%   - sp500_data.csv (Date, Open, High, Low, Close, Volume)
%   - bitcoin_data.csv (Date, Open, High, Low, Close, Volume)
%   - TB3MS.csv (observation_date, TB3MS)
%
% Outputs: 
%   - Plots (PDFs)
%   - Summary statistics
%   - Portfolio metrics
%   - CVaR optimisation results
%
% Author: Adam Gamblin (F116538)
% Institution: Loughborough University
% Date: April 27, 2025
%
% ============================================================================

%% ============================================================================
% INITIALIZATION AND CONFIGURATION
% ============================================================================
% Clear workspace and close figures for a fresh start
clc; clear; close all;

% Set random seed for reproducibility of Monte Carlo simulations
rng(42);

%% ===== CONFIGURATION =====
% Simulation parameters for Monte Carlo analysis
numSim = 100000; % Number of Monte Carlo simulations
nPorts = 20; % Number of frontier points

% Asset and portfolio configuration
% Define the three assets in our portfolio
assetNames = {'Bitcoin', 'S&P 500', 'Bonds'};
% Define three risk profiles for portfolio optimization
portfolioNames = {'Low-Risk', 'Moderate-Risk', 'High-Risk'};
% ---- Baseline portfolio (60/40 stocks‑bonds, 0 % BTC) ----
baselineName   = '60/40 Baseline';
baselineWeights = [0; 0.60; 0.40];   % [BTC, S&P 500, Bonds]
% Maximum Bitcoin allocation caps for each risk profile
maxBTC = [0.01, 0.05, 0.20]; % Bitcoin allocation caps
% Group constraints for stocks and bonds (NaN means no constraint)
grpLimit = [0.40, 0.60, NaN]; % Group constraints for stocks and bonds

% Visualization settings for consistent plot formatting
plotSettings = struct();
plotSettings.figWidth = 800;    % Figure width in pixels
plotSettings.figHeight = 600;   % Figure height in pixels
plotSettings.fontSize = 12;     % Base font size for plots
plotSettings.lineWidth = 1.5;   % Line width for plots
% Color scheme for consistent visualization
plotSettings.colors = {[0, 0.4470, 0.7410], [0.4660, 0.6740, 0.1880], [0.8500, 0.3250, 0.0980]};
plotSettings.exportFormat = 'vector'; % 'vector' or 'image'
plotSettings.exportDPI = 300; % Only used if exportFormat is 'image'

% Performance metrics configuration
metrics = struct();
metrics.varConfidence = 0.95; % VaR confidence level (95%)
metrics.cvarConfidence = 0.95; % CVaR confidence level (95%)
metrics.maxDrawdownThreshold = 0.20; % Maximum acceptable drawdown (20%)

% Advanced estimation parameters for portfolio optimization
estimationParams = struct();
estimationParams.lambda = 0.94; % Decay factor for exponential weighting
% Shrinkage target options: 'constant', 'diagonal', or 'single-index'
estimationParams.shrinkageTarget = 'constant';
estimationParams.shrinkageIntensity = 0.3; % Shrinkage intensity (0-1)
estimationParams.garchWindow = 60; % Window size for GARCH estimation
estimationParams.mcHorizon = 12; % Monte Carlo simulation horizon (months)

%% ===== DATA LOADING AND PREPROCESSING =====
% Load and process all required data
try
    % Load price data for each asset
    bondsTT = readPriceCSV('Data/bonds_data.csv', 'Close');
    sp500TT = readPriceCSV('Data/sp500_data.csv', 'Close');
    bitcoinTT = readPriceCSV('Data/bitcoin_data.csv', 'Close');

    % Load risk-free rate data (3-month Treasury Bill)
    optsRfr = detectImportOptions('Data/TB3MS.csv');
    optsRfr = setvaropts(optsRfr, 'observation_date', 'InputFormat', 'dd/MM/yyyy');
    rfrTbl = readtable('Data/TB3MS.csv', optsRfr);
    rfrTT = timetable(datetime(rfrTbl.observation_date), rfrTbl.TB3MS/100, 'VariableNames', {'RFR'});

    % Synchronize all data on monthly calendar
    allTT = synchronize(bitcoinTT, sp500TT, bondsTT, rfrTT, 'monthly', 'previous');
    if height(allTT) < 2
        error('Insufficient data points after synchronization.');
    end

    % Extract aligned data for analysis
    dates = allTT.Time;
    prices = allTT{:, {'Bitcoin', 'SP500', 'Bonds'}};
    alignedRfr = allTT.RFR;
    pricesBitcoin = prices(:,1);
    pricesSp500 = prices(:,2);
    pricesBonds = prices(:,3);

    % Plot asset price trends
    plotAssetPriceTrends(dates, pricesBitcoin, pricesSp500, pricesBonds, 'Outputs/Figures/asset_price_trends.pdf');

    % Compute monthly log returns for each asset
    returnsMat = diff(log(prices));
    datesReturns = dates(2:end);
    % Remove any rows with missing data
    valid = all(~isnan(returnsMat), 2);
    returnsMat = returnsMat(valid, :);
    datesReturns = datesReturns(valid);
    alignedRfr = alignedRfr(valid);
    returnsBitcoin = returnsMat(:,1);
    returnsSp500 = returnsMat(:,2);
    returnsBonds = returnsMat(:,3);
catch e
    error('Error during data loading: %s', e.message);
end

%% ===== DESCRIPTIVE STATISTICS =====
% Calculate and display summary statistics for each asset's returns
fprintf('Summary Statistics for Monthly Log Returns:\n');
for i = 1:3
    ret = returnsMat(:, i);
    % Display mean, standard deviation, skewness, and kurtosis for each asset
    fprintf('%s: Mean = %.4f, Std = %.4f, Skewness = %.4f, Kurtosis = %.4f\n', ...
        assetNames{i}, mean(ret), std(ret), skewness(ret), kurtosis(ret));
end

% Correlation analysis - CRUCIAL for understanding diversification benefits
% Calculate correlation matrix between all assets
corrMatrix = corr(returnsMat);
% Visualize correlations using a heatmap
plotHeatmap(corrMatrix, assetNames, 'Outputs/Figures/correlation_heatmap.pdf');

%% ===== BITCOIN CHARACTERISTICS ANALYSIS =====
% Bitcoin beta calculation - Important for understanding Bitcoin's relationship with the market
% Use robust regression to handle outliers
mdl = fitlm(returnsSp500, returnsBitcoin, 'RobustOpts', 'on');
betaBitcoin = mdl.Coefficients.Estimate(2);
fprintf('Bitcoin Beta: %.4f\n', betaBitcoin);

% GARCH(1,1) model for Bitcoin volatility - Critical for risk assessment
try
    % Configure GARCH model with t-distribution for fat tails
    Mdl = garch(1,1); Mdl.Distribution = 't';
    % Set optimization options for GARCH estimation
    opts = optimoptions('fmincon', 'Display', 'off', 'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-6);
    % Estimate GARCH parameters
    [EstMdl, EstParamCov, logL] = estimate(Mdl, returnsBitcoin, 'Display', 'off', 'Options', opts);
    % Calculate model selection criteria
    numParams = sum(any(EstParamCov~=0,2));
    [AIC, BIC] = aicbic(logL, numParams, length(returnsBitcoin));
    fprintf('GARCH(1,1): AIC = %.4f, BIC = %.4f\n', AIC, BIC);
    % Calculate conditional volatility
    v = infer(EstMdl, returnsBitcoin);
    condVol = sqrt(v) * sqrt(12) * 100; % Annualized and converted to percentage
catch e
    warning('portfolio_analysis:GARCHFailed', '%s', e.message);
    % Fallback to simple volatility estimation if GARCH fails
    condVol = movstd(returnsBitcoin, 6, 'omitnan') * sqrt(12) * 100;
end

% Calculate rolling realized volatility (6-month window)
window = 6;
realizedVol = movstd(returnsBitcoin, window, 'omitnan') * sqrt(12) * 100;

% Validate volatility data before plotting
if isempty(condVol) || any(isnan(condVol)) || isempty(realizedVol) || any(isnan(realizedVol))
    error('Invalid GARCH or realized volatility data: condVol or realizedVol contains NaNs or is empty.');
end
if length(condVol) ~= length(realizedVol) || length(condVol) ~= length(datesReturns)
    error('Mismatched lengths: condVol (%d), realizedVol (%d), datesReturns (%d).', ...
        length(condVol), length(realizedVol), length(datesReturns));
end

% Create figure for volatility visualization
figure('Position', [100; 100; 1000; 600]);
ymax = max([condVol; realizedVol]) * 1.1;
if isnan(ymax) || ymax <= 0
    error('Invalid ymax: %.2f. Check condVol and realizedVol for valid values.', ymax);
end
ylim([0, ymax]);

% Define historical crash periods for visualization
crashPeriods = { ...
    [2018,1,1], [2018,12,31], '2018 Drawdown'; ...
    [2020,3,1], [2020,3,31], 'COVID-19 Crash'; ...
    [2022,5,1], [2022,6,30], 'May–Jun 2022'; ...
    [2022,11,1], [2022,11,30], 'Nov 2022'};
labelY = [ymax*0.97, ymax*0.93];

% Plot crash periods as shaded regions
for i = 1:size(crashPeriods,1)
    startD = datetime(crashPeriods{i,1});
    endD = datetime(crashPeriods{i,2});
    label = crashPeriods{i,3};
    % Add shaded region for crash period
    fill([startD endD endD startD], [0 0 ymax ymax], [0.9 0.9 0.9], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.4, 'HandleVisibility', 'off');
    % Add crash period label
    midD = startD + (endD - startD)/2;
    yPos = labelY(mod(i-1,2)+1);
    text(midD, yPos, label, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 10, 'FontWeight', 'bold', 'HandleVisibility', 'off');
end

% Plot GARCH and realized volatility
plot(datesReturns, condVol, 'LineWidth', 1.8, 'Color', [0 0.4470 0.7410], ...
    'DisplayName', 'GARCH(1,1) Conditional Volatility');
hold on;
plot(datesReturns, realizedVol, '--', 'LineWidth', 1.8, 'Color', [0.3 0.3 0.3], ...
    'DisplayName', 'Realized Volatility (6-mo)');

% Configure axes
ax = gca;
ax.XLim = [datesReturns(1), datesReturns(end)];
ax.XTick = datesReturns(1:12:end);
xtickformat('yyyy');
xtickangle(45);
xlabel('Date', 'FontSize', plotSettings.fontSize);
ylabel('Annualized Volatility (%)', 'FontSize', plotSettings.fontSize);
title('Bitcoin Volatility: GARCH(1,1) vs Realized (6-Month Rolling)', 'FontSize', plotSettings.fontSize);
legend('Location', 'northwest', 'Box', 'off', 'FontSize', plotSettings.fontSize);
grid on;
box on;
hold off;

% Ensure plot is rendered before export
drawnow;
set(gcf, 'MenuBar', 'none', 'ToolBar', 'none');

% Export to PDF
if strcmp(plotSettings.exportFormat, 'vector')
    exportgraphics(gcf, 'Outputs/Figures/garch_volatility_labeled.pdf', 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    exportgraphics(gcf, 'Outputs/Figures/garch_volatility_labeled.png', 'Resolution', plotSettings.exportDPI, 'BackgroundColor', 'white');
end
close(gcf);

%% ===== PORTFOLIO OPTIMIZATION =====
% Calculate expected returns using exponential weighting (more weight to recent data)
lambda = estimationParams.lambda;
% Create exponentially weighted vector
weights = (1-lambda) * lambda.^((length(returnsMat)-1):-1:0)';
weights = weights / sum(weights);
% Calculate weighted expected returns
expReturnsMonthly = sum(returnsMat .* weights, 1);

% Calculate covariance matrix with shrinkage
% First, compute the sample covariance
sampleCov = cov(returnsMat);

% Apply shrinkage to reduce estimation error
if strcmp(estimationParams.shrinkageTarget, 'constant')
    % Shrink towards constant correlation matrix
    avgVar = mean(diag(sampleCov));
    avgCov = mean(sampleCov(sampleCov ~= diag(sampleCov)));
    target = avgCov * ones(size(sampleCov)) + (avgVar - avgCov) * eye(size(sampleCov));
elseif strcmp(estimationParams.shrinkageTarget, 'diagonal')
    % Shrink towards diagonal matrix
    target = diag(diag(sampleCov));
elseif strcmp(estimationParams.shrinkageTarget, 'single-index')
    % Shrink towards single-index model
    % This is a simplified version - in practice, you'd use market returns
    marketReturns = mean(returnsMat, 2);
    beta = zeros(3, 1);
    for i = 1:3
        beta(i) = cov(returnsMat(:,i), marketReturns) / var(marketReturns);
    end
    marketVar = var(marketReturns);
    target = beta * beta' * marketVar + diag(diag(sampleCov) - diag(beta * beta' * marketVar));
else
    % Default to constant correlation
    avgVar = mean(diag(sampleCov));
    avgCov = mean(sampleCov(sampleCov ~= diag(sampleCov)));
    target = avgCov * ones(size(sampleCov)) + (avgVar - avgCov) * eye(size(sampleCov));
end

% Apply shrinkage to covariance matrix
shrinkageIntensity = estimationParams.shrinkageIntensity;
covMatrixMonthly = shrinkageIntensity * target + (1 - shrinkageIntensity) * sampleCov;

% Annualize returns and covariance
expReturns = 12 * expReturnsMonthly;
covMatrix = 12 * covMatrixMonthly;

% Use the average annual risk-free rate
rf = mean(alignedRfr);
% Convert annual risk-free rate to a monthly equivalent
rfMonthly = (1 + rf)^(1/12) - 1;
fprintf('Average RF (annual): %.4f%%\n', rf*100);
fprintf('Min RF (annual): %.4f%%\n', min(alignedRfr)*100);
fprintf('Max RF (annual): %.4f%%\n', max(alignedRfr)*100);
fprintf('Std Dev of RF series: %.4f%%\n', std(alignedRfr)*100);

% Initialize arrays for portfolio weights and metrics
weightsProfiles = zeros(3, length(portfolioNames));
risksProfiles = zeros(1, length(portfolioNames));
retsProfiles = zeros(1, length(portfolioNames));
sharpeProfiles = zeros(1, length(portfolioNames));

% Optimize portfolios for different risk profiles
fprintf('\nMaximum-Sharpe Portfolios:\n');
% Store frontier points for plotting
p = Portfolio('AssetMean', expReturns, 'AssetCovar', covMatrix);
p = setDefaultConstraints(p);
W = estimateFrontier(p, nPorts);
[frontierRisks, frontierRets] = estimatePortMoments(p, W);

% Calculate unconstrained optimal allocations
pUnconstrained = Portfolio('AssetMean', expReturns, 'AssetCovar', covMatrix);
pUnconstrained = setDefaultConstraints(pUnconstrained);
% Find maximum Sharpe ratio portfolio
[weightsMaxSharpe, ~, ~] = estimateMaxSharpeRatio(pUnconstrained);
% Find minimum variance portfolio
[weightsMinVar, ~, ~] = estimateFrontierLimits(pUnconstrained);
% Find maximum return portfolio
[weightsMaxRet, ~, ~] = estimateFrontierLimits(pUnconstrained, 'max');

fprintf('\nUnconstrained Optimal Allocations:\n');
fprintf('Minimum Variance Portfolio:\n');
fprintf('  Bitcoin: %.1f%%\n', weightsMinVar(1)*100);
fprintf('  S&P 500: %.1f%%\n', weightsMinVar(2)*100);
fprintf('  Bonds: %.1f%%\n\n', weightsMinVar(3)*100);

fprintf('Maximum Sharpe Portfolio:\n');
fprintf('  Bitcoin: %.1f%%\n', weightsMaxSharpe(1)*100);
fprintf('  S&P 500: %.1f%%\n', weightsMaxSharpe(2)*100);
fprintf('  Bonds: %.1f%%\n\n', weightsMaxSharpe(3)*100);

fprintf('Maximum Return Portfolio:\n');
fprintf('  Bitcoin: %.1f%%\n', weightsMaxRet(1)*100);
fprintf('  S&P 500: %.1f%%\n', weightsMaxRet(2)*100);
fprintf('  Bonds: %.1f%%\n\n', weightsMaxRet(3)*100);

% Optimize each risk profile portfolio
for k = 1:length(portfolioNames)
    % Set up portfolio optimization problem
    p = Portfolio('AssetMean', expReturns, 'AssetCovar', covMatrix);
    p = setDefaultConstraints(p);
    % Set Bitcoin allocation cap
    p = setBounds(p, zeros(3,1), [maxBTC(k); 1; 1]);
    % Set group constraints if specified
    if ~isnan(grpLimit(k))
        p = setGroups(p, [1 1 0], [], grpLimit(k));
    end
    % Find efficient frontier points
    W = estimateFrontier(p, nPorts);
    [risks, rets] = estimatePortMoments(p, W);
    % Calculate Sharpe ratios
    srs = (rets - rf) ./ risks;
    % Find maximum Sharpe ratio portfolio
    [~, ix] = max(srs);
    % Store results
    weightsProfiles(:,k) = W(:,ix);
    risksProfiles(k) = risks(ix);
    retsProfiles(k) = rets(ix);
    sharpeProfiles(k) = srs(ix);
    % Display results
    fprintf('%-13s: BTC=%5.1f%%, S&P=%5.1f%%, BND=%5.1f%% | σ=%5.2f%%, μ=%5.2f%%, SR=%4.2f\n', ...
        portfolioNames{k}, W(1,ix)*100, W(2,ix)*100, W(3,ix)*100, risks(ix)*100, rets(ix)*100, srs(ix));
end
weightsLow = weightsProfiles(:,1);
weightsMod = weightsProfiles(:,2);
weightsHigh = weightsProfiles(:,3);

% ===== BASELINE (60/40) METRICS =====
retBaseline    = expReturns * baselineWeights;   % dot‑product gives scalar
riskBaseline   = sqrt(baselineWeights' * covMatrix * baselineWeights);
sharpeBaseline = (retBaseline - rf) / riskBaseline;

weightsBaseline = baselineWeights;   % keep naming symmetric with Low/Mod/High

% Plot efficient frontier - CRUCIAL for understanding risk-return tradeoffs
assetRisk = sqrt(diag(covMatrix));
assetReturn = expReturns;
plotFrontier(frontierRisks, frontierRets, rf, assetRisk, assetReturn, assetNames, 'Outputs/Figures/efficient_frontier_onepanel_clean.pdf');

% Create a focused visualization of optimal Bitcoin allocations
figure('Position', [100; 100; 800; 500]);
barData = [weightsProfiles(1,:)*100; weightsProfiles(2,:)*100; weightsProfiles(3,:)*100]';
b = bar(barData, 'stacked');
title('Optimal Bitcoin Allocation by Risk Profile', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Risk Profile', 'FontSize', 12);
ylabel('Allocation (%)', 'FontSize', 12);
set(gca, 'XTickLabel', portfolioNames);
legend(assetNames, 'Location', 'best', 'Box', 'off', 'FontSize', 12);
grid on;
% Add text labels with exact percentages
for i = 1:length(portfolioNames)
    for j = 1:3
        if barData(i,j) > 5  % Only show labels for allocations > 5%
            text(i, sum(barData(i,1:j))-barData(i,j)/2, sprintf('%.1f%%', barData(i,j)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 10, 'FontWeight', 'bold');
        end
    end
end

% Ensure plot is rendered and UI elements are hidden
drawnow;
set(gcf, 'MenuBar', 'none', 'ToolBar', 'none');

% Export the Bitcoin allocation visualization
if strcmp(plotSettings.exportFormat, 'vector')
    exportgraphics(gcf, 'Outputs/Figures/bitcoin_allocation_by_risk_profile.pdf', 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    exportgraphics(gcf, 'Outputs/Figures/bitcoin_allocation_by_risk_profile.png', 'Resolution', plotSettings.exportDPI, 'BackgroundColor', 'white');
end
close(gcf);

%% ===== RISK ANALYSIS =====
% Monte Carlo simulation with GARCH-based volatility
try
    % Fit t-distribution to Bitcoin returns for better tail risk modeling
    pd = fitdist(returnsBitcoin, 'tLocationScale');
    nu = pd.nu;
    fprintf('Estimated degrees of freedom for t-distribution: %.4f\n', nu);
    if nu <= 2
        warning('Degrees of freedom (nu) <= 2 may lead to infinite variance. Setting nu to 3.');
        nu = 3;
    end
    
    % Use GARCH model for Bitcoin volatility forecasting
    if exist('EstMdl', 'var') && ~isempty(EstMdl)
        % Forecast volatility for the next 12 months
        vForecast = forecast(EstMdl, estimationParams.mcHorizon, 'Y0', returnsBitcoin);
        volForecast = sqrt(vForecast) * sqrt(12); % Annualized
    else
        % Fallback to simple volatility estimation
        volForecast = std(returnsBitcoin) * sqrt(12) * ones(estimationParams.mcHorizon, 1);
    end
    
    % Generate correlated random returns
    % Scale covariance matrix for t-distribution
    Psi = covMatrixMonthly * ((nu - 2) / nu);
    % Generate chi-square random variables for t-distribution
    V = chi2rnd(nu, estimationParams.mcHorizon * numSim, 1) / nu;
    
    % Generate base returns using multivariate normal distribution
    simReturnsRaw = mvnrnd(zeros(1,3), Psi, estimationParams.mcHorizon * numSim);
    
    % Apply time-varying volatility for Bitcoin
    for t = 1:estimationParams.mcHorizon
        startIdx = (t-1)*numSim + 1;
        endIdx = t*numSim;
        % Scale Bitcoin returns by the forecasted volatility ratio
        volRatio = volForecast(t) / (std(returnsBitcoin) * sqrt(12));
        simReturnsRaw(startIdx:endIdx, 1) = simReturnsRaw(startIdx:endIdx, 1) * volRatio;
    end
    
    % Add expected returns to simulated returns
    simReturns = repmat(expReturnsMonthly, estimationParams.mcHorizon * numSim, 1) + simReturnsRaw ./ sqrt(V);
    simReturns = reshape(simReturns, estimationParams.mcHorizon, numSim, 3);
    
    % Calculate portfolio returns for each risk profile
    simPortReturnsLow = squeeze(sum(simReturns .* permute(weightsLow, [3,2,1]), 3));
    portLow = 100000 * cumprod(1 + simPortReturnsLow, 1);
    simPortReturnsMod = squeeze(sum(simReturns .* permute(weightsMod, [3,2,1]), 3));
    portMod = 100000 * cumprod(1 + simPortReturnsMod, 1);
    simPortReturnsHigh = squeeze(sum(simReturns .* permute(weightsHigh, [3,2,1]), 3));
    portHigh = 100000 * cumprod(1 + simPortReturnsHigh, 1);
    simPortReturnsBase = squeeze(sum(simReturns .* permute(weightsBaseline, [3,2,1]), 3));
    portBase = 100000 * cumprod(1 + simPortReturnsBase, 1);
    endValuesBase = portBase(end,:);
    varBasePct  = (100000 - prctile(endValuesBase, 5)) / 100000 * 100;
    cvarBasePct = (100000 - mean(endValuesBase(endValuesBase <= prctile(endValuesBase, 5)))) / 100000 * 100;
    
    % Calculate end values and risk metrics
    endValuesLow = portLow(end,:);
    endValuesMod = portMod(end,:);
    endValuesHigh = portHigh(end,:);
    
    % Calculate Value at Risk (VaR)
    varLowPct = (100000 - prctile(endValuesLow, 5)) / 100000 * 100;
    varModPct = (100000 - prctile(endValuesMod, 5)) / 100000 * 100;
    varHighPct = (100000 - prctile(endValuesHigh, 5)) / 100000 * 100;
    
    % Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    cvarLowPct = (100000 - mean(endValuesLow(endValuesLow <= prctile(endValuesLow, 5)))) / 100000 * 100;
    cvarModPct = (100000 - mean(endValuesMod(endValuesMod <= prctile(endValuesMod, 5)))) / 100000 * 100;
    cvarHighPct = (100000 - mean(endValuesHigh(endValuesHigh <= prctile(endValuesHigh, 5)))) / 100000 * 100;
    
    % Plot Monte Carlo simulation results for each portfolio
    plotMonteCarlo(endValuesLow, 'Low-Risk', 'Outputs/Figures/monte_carlo_low_risk.pdf');
    plotMonteCarlo(endValuesMod, 'Moderate-Risk', 'Outputs/Figures/monte_carlo_moderate_risk.pdf');
    plotMonteCarlo(endValuesHigh, 'High-Risk', 'Outputs/Figures/monte_carlo_high_risk.pdf');
    
    % Display risk metrics
    fprintf('Updated VaR (5%%, 12 Months) with t-Distribution and GARCH:\n');
    fprintf('Low-Risk Portfolio: %.2f%%\n', varLowPct);
    fprintf('Moderate-Risk Portfolio: %.2f%%\n', varModPct);
    fprintf('High-Risk Portfolio: %.2f%%\n', varHighPct);
    fprintf('60/40 Baseline Portfolio: %.2f%%\n', varBasePct);
    
    fprintf('CVaR (Expected Shortfall) at 5%% level:\n');
    fprintf('Low-Risk Portfolio: %.2f%%\n', cvarLowPct);
    fprintf('Moderate-Risk Portfolio: %.2f%%\n', cvarModPct);
    fprintf('High-Risk Portfolio: %.2f%%\n', cvarHighPct);
    fprintf('60/40 Baseline Portfolio: %.2f%%\n', cvarBasePct);
catch e
    warning('portfolio_analysis:MonteCarloFailed', '%s', e.message);
    % Set default values if simulation fails
    varLowPct = NaN; varModPct = NaN; varHighPct = NaN;
    cvarLowPct = NaN; cvarModPct = NaN; cvarHighPct = NaN;
end

% Summary table of portfolio weights and risk metrics - CRUCIAL for answering the research question
summaryTable = table(portfolioNames',                            ...
                     weightsProfiles(1,:)'*100,                  ...
                     weightsProfiles(2,:)'*100,                  ...
                     weightsProfiles(3,:)'*100,                  ...
                     retsProfiles'*100,                          ...
                     risksProfiles'*100,                         ...
                     sharpeProfiles',                            ...
                     [varLowPct; varModPct; varHighPct],         ...
                     [cvarLowPct; cvarModPct; cvarHighPct],      ...
         'VariableNames', {'Portfolio','BTC (%)','S&P 500 (%)','Bonds (%)', ...
                           'Exp Return (%)','Risk (%)','Sharpe Ratio',      ...
                           'VaR 5% (12M, %)','CVaR 5% (12M, %)'} );

% ---- append baseline row ----
baselineRow = {baselineName, baselineWeights(1)*100, baselineWeights(2)*100, ...
               baselineWeights(3)*100, retBaseline*100, riskBaseline*100,   ...
               sharpeBaseline, varBasePct, cvarBasePct};
summaryTable = [summaryTable; baselineRow];
fprintf('Summary of Portfolio Metrics:\n');
disp(summaryTable);

% Create a focused risk-return visualization for the research question
figure('Position', [100; 100; 800; 500]);
scatter(risksProfiles*100, retsProfiles*100, 100, 'filled', 'MarkerFaceColor', [0.3010, 0.7450, 0.9330]);
hold on;
for i = 1:length(portfolioNames)
    text(risksProfiles(i)*100, retsProfiles(i)*100, sprintf('  %s\n  BTC: %.1f%%', portfolioNames{i}, weightsProfiles(1,i)*100), ...
        'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
end
title('Risk-Return Profile with Bitcoin Allocation', 'FontSize', 14, 'FontWeight', 'bold');
subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
xlabel('Portfolio Risk (%)', 'FontSize', 12);
ylabel('Expected Return (%)', 'FontSize', 12);
grid on;
% Export the risk-return visualization
if strcmp(plotSettings.exportFormat, 'vector')
    exportgraphics(gcf, 'Outputs/Figures/risk_return_bitcoin_allocation.pdf', 'ContentType', 'vector');
else
    exportgraphics(gcf, 'Outputs/Figures/risk_return_bitcoin_allocation.png', 'Resolution', plotSettings.exportDPI);
end
close(gcf);

%% ===== HISTORICAL PERFORMANCE ANALYSIS =====
% Calculate historical portfolio returns and values
historicalReturns = returnsMat;
% Calculate returns and cumulative values for each risk profile
portHistReturnsLow = historicalReturns * weightsLow;
portHistValueLow = 100000 * cumprod(1 + portHistReturnsLow);
portHistReturnsMod = historicalReturns * weightsMod;
portHistValueMod = 100000 * cumprod(1 + portHistReturnsMod);
portHistReturnsHigh = historicalReturns * weightsHigh;
portHistValueHigh = 100000 * cumprod(1 + portHistReturnsHigh);

% Plot historical portfolio performance
figure('Position', [100; 100; plotSettings.figWidth; plotSettings.figHeight]);
% Plot each portfolio's value over time
plot(datesReturns, portHistValueLow, 'b-', 'LineWidth', plotSettings.lineWidth, 'DisplayName', 'Low-Risk');
hold on;
plot(datesReturns, portHistValueMod, 'g-', 'LineWidth', plotSettings.lineWidth, 'DisplayName', 'Moderate-Risk');
plot(datesReturns, portHistValueHigh, 'r-', 'LineWidth', plotSettings.lineWidth, 'DisplayName', 'High-Risk');
% Add S&P 500 as benchmark
plot(datesReturns, 100000 * cumprod(1 + returnsSp500), 'k--', 'LineWidth', plotSettings.lineWidth, 'DisplayName', 'S&P 500');
% Configure plot appearance
title('Historical Portfolio Performance (£100,000 Initial Investment)', 'FontSize', plotSettings.fontSize);
subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
xlabel('Date', 'FontSize', plotSettings.fontSize); 
ylabel('Portfolio Value (£)', 'FontSize', plotSettings.fontSize);
legend('Location', 'best', 'Box', 'off', 'FontSize', plotSettings.fontSize); 
grid on; 
hold off;

% Export plot with appropriate settings
if strcmp(plotSettings.exportFormat, 'vector')
    exportgraphics(gcf, 'Outputs/Figures/portfolio_value_over_time.pdf', 'ContentType', 'vector');
else
    exportgraphics(gcf, 'Outputs/Figures/portfolio_value_over_time.png', 'Resolution', plotSettings.exportDPI);
end
close(gcf);

% Portfolio weights pie charts - Important for visualizing optimal allocations
fig = figure('Position', [100; 100; 1200; 400]);
% Create pie charts for each risk profile
for k = 1:3
    subplot(1,3,k);
    weights = weightsProfiles(:,k);
    % Create labels with percentage values
    labels = cell(3,1);
    for i = 1:3
        labels{i} = sprintf('%s (%.1f%%)', assetNames{i}, weights(i)*100);
    end
    % Create and format pie chart
    pie(weights, labels);
    title(sprintf('%s Portfolio Weights', portfolioNames{k}), 'FontSize', plotSettings.fontSize);
    subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
end

% Export pie charts
if strcmp(plotSettings.exportFormat, 'vector')
    exportgraphics(fig, 'Outputs/Figures/portfolio_weights.pdf', 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    exportgraphics(fig, 'Outputs/Figures/portfolio_weights.png', 'Resolution', plotSettings.exportDPI, 'BackgroundColor', 'white');
end
close(fig);

% Drawdown analysis
% Calculate maximum drawdown for each portfolio
cumMaxLow = cummax(portHistValueLow);
drawdownLow = (portHistValueLow - cumMaxLow) ./ cumMaxLow;
cumMaxMod = cummax(portHistValueMod);
drawdownMod = (portHistValueMod - cumMaxMod) ./ cumMaxMod;
cumMaxHigh = cummax(portHistValueHigh);
drawdownHigh = (portHistValueHigh - cumMaxHigh) ./ cumMaxHigh;

% Calculate and display maximum drawdowns
maxDrawdownLow = min(drawdownLow) * 100;
maxDrawdownMod = min(drawdownMod) * 100;
maxDrawdownHigh = min(drawdownHigh) * 100;

fprintf('\nMaximum Drawdowns:\n');
fprintf('Low-Risk Portfolio: %.2f%%\n', maxDrawdownLow);
fprintf('Moderate-Risk Portfolio: %.2f%%\n', maxDrawdownMod);
fprintf('High-Risk Portfolio: %.2f%%\n', maxDrawdownHigh);

%% ===== COMPREHENSIVE PERFORMANCE METRICS =====
% Calculate annualized returns and volatility for each portfolio
annualizedReturns = zeros(3,1);
annualizedVol = zeros(3,1);
sharpeRatios = zeros(3,1);
sortinoRatios = zeros(3,1);
calmarRatios = zeros(3,1);

% Calculate metrics for each risk profile
for k = 1:3
    % Get portfolio returns based on risk profile
    if k == 1
        portReturns = portHistReturnsLow;
    elseif k == 2
        portReturns = portHistReturnsMod;
    else
        portReturns = portHistReturnsHigh;
    end
    
    % Calculate annualized return (assuming monthly data)
    annualizedReturns(k) = (1 + mean(portReturns))^12 - 1;
    
    % Calculate annualized volatility
    annualizedVol(k) = std(portReturns) * sqrt(12);
    
    % Calculate Sharpe ratio using monthly-equivalent risk-free rate
    excessReturns = portReturns - rfMonthly;
    sharpeRatios(k) = sqrt(12) * mean(excessReturns) / std(portReturns);
    
    % Calculate Sortino ratio (downside deviation)
    downsideReturns = portReturns(portReturns < 0);
    downsideDev = std(downsideReturns) * sqrt(12);
    sortinoRatios(k) = sqrt(12) * mean(excessReturns) / downsideDev;
    
    % Calculate Calmar ratio (using maximum drawdown)
    if k == 1
        maxDD = abs(maxDrawdownLow/100);
    elseif k == 2
        maxDD = abs(maxDrawdownMod/100);
    else
        maxDD = abs(maxDrawdownHigh/100);
    end
    calmarRatios(k) = annualizedReturns(k) / maxDD;
end

% Display comprehensive performance metrics
fprintf('\nComprehensive Performance Metrics:\n');
fprintf('Metric\t\t\tLow-Risk\tModerate-Risk\tHigh-Risk\n');
fprintf('Annual Return\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\n', ...
    annualizedReturns(1)*100, annualizedReturns(2)*100, annualizedReturns(3)*100);
fprintf('Annual Volatility\t%.2f%%\t\t%.2f%%\t\t%.2f%%\n', ...
    annualizedVol(1)*100, annualizedVol(2)*100, annualizedVol(3)*100);
fprintf('Sharpe Ratio\t\t%.2f\t\t%.2f\t\t%.2f\n', ...
    sharpeRatios(1), sharpeRatios(2), sharpeRatios(3));
fprintf('Sortino Ratio\t\t%.2f\t\t%.2f\t\t%.2f\n', ...
    sortinoRatios(1), sortinoRatios(2), sortinoRatios(3));
fprintf('Calmar Ratio\t\t%.2f\t\t%.2f\t\t%.2f\n', ...
    calmarRatios(1), calmarRatios(2), calmarRatios(3));

% Create performance metrics table for visualization
metricsTable = table(annualizedReturns*100, annualizedVol*100, sharpeRatios, ...
    sortinoRatios, calmarRatios, 'RowNames', portfolioNames, ...
    'VariableNames', {'AnnualReturn', 'AnnualVol', 'SharpeRatio', ...
    'SortinoRatio', 'CalmarRatio'});

% Display formatted table
disp(metricsTable);

% Save metrics to CSV file
writetable(metricsTable, 'Outputs/portfolio_metrics.csv', 'WriteRowNames', true);

% Create a focused summary for the research question
fprintf('\n===== RESEARCH QUESTION SUMMARY =====\n');
fprintf('Optimal Bitcoin Allocation in a Retail Investor Portfolio:\n\n');
fprintf('Low-Risk Profile: %.1f%% Bitcoin, %.1f%% S&P 500, %.1f%% Bonds\n', ...
    weightsProfiles(1,1)*100, weightsProfiles(2,1)*100, weightsProfiles(3,1)*100);
fprintf('Expected Return: %.2f%%, Risk: %.2f%%, Sharpe Ratio: %.2f\n', ...
    retsProfiles(1)*100, risksProfiles(1)*100, sharpeProfiles(1));
fprintf('Maximum Drawdown: %.2f%%, VaR (5%%): %.2f%%, CVaR (5%%): %.2f%%\n\n', ...
    maxDrawdownLow, varLowPct, cvarLowPct);

fprintf('Moderate-Risk Profile: %.1f%% Bitcoin, %.1f%% S&P 500, %.1f%% Bonds\n', ...
    weightsProfiles(1,2)*100, weightsProfiles(2,2)*100, weightsProfiles(3,2)*100);
fprintf('Expected Return: %.2f%%, Risk: %.2f%%, Sharpe Ratio: %.2f\n', ...
    retsProfiles(2)*100, risksProfiles(2)*100, sharpeProfiles(2));
fprintf('Maximum Drawdown: %.2f%%, VaR (5%%): %.2f%%, CVaR (5%%): %.2f%%\n\n', ...
    maxDrawdownMod, varModPct, cvarModPct);

fprintf('60/40 Baseline Portfolio: %.1f%% Bitcoin, %.1f%% S&P 500, %.1f%% Bonds\n', ...
    baselineWeights(1)*100, baselineWeights(2)*100, baselineWeights(3)*100);
fprintf('Expected Return: %.2f%%, Risk: %.2f%%, Sharpe Ratio: %.2f\n', ...
    retBaseline*100, riskBaseline*100, sharpeBaseline);
fprintf('Maximum Drawdown: N/A, VaR (5%%): %.2f%%, CVaR (5%%): %.2f%%\n\n', ...
    varBasePct, cvarBasePct);

fprintf('High-Risk Profile: %.1f%% Bitcoin, %.1f%% S&P 500, %.1f%% Bonds\n', ...
    weightsProfiles(1,3)*100, weightsProfiles(2,3)*100, weightsProfiles(3,3)*100);
fprintf('Expected Return: %.2f%%, Risk: %.2f%%, Sharpe Ratio: %.2f\n', ...
    retsProfiles(3)*100, risksProfiles(3)*100, sharpeProfiles(3));
fprintf('Maximum Drawdown: %.2f%%, VaR (5%%): %.2f%%, CVaR (5%%): %.2f%%\n', ...
    maxDrawdownHigh, varHighPct, cvarHighPct);

fprintf('\nNote: VaR is calculated using a multivariate t-distribution with GARCH-based volatility forecasting to capture fat-tailed returns and time-varying volatility, improving tail risk estimation.\n');

%% ===== HELPER FUNCTIONS =====

function plotAssetPriceTrends(dates, bitcoinPrices, sp500Prices, bondPrices, fileName)
    % PLOTASSETPRICETRENDS Visualize the price trends of Bitcoin, S&P 500, and bonds.
    %   This function creates a plot showing the historical price trends of the three assets
    %   with proper scaling and labeling.
    %
    %   Inputs:
    %       dates         - Vector of dates
    %       bitcoinPrices - Vector of Bitcoin prices
    %       sp500Prices   - Vector of S&P 500 prices
    %       bondPrices    - Vector of bond prices
    %       fileName      - String filename for exporting the plot
    %
    %   Example:
    %       plotAssetPriceTrends(dates, bitcoinPrices, sp500Prices, bondPrices, 'asset_price_trends.pdf');
    
    % Input validation
    if nargin < 5
        error('plotAssetPriceTrends:NotEnoughInputs', 'Not enough input arguments.');
    end
    
    try
        % Create figure with appropriate size
        figure('Position', [100; 100; 1200; 800]);
        set(gcf, 'Color', 'white');
        
        % Define colors
        colors = {[0, 0.4470, 0.7410];        % S&P 500 (blue)
                 [0.8500, 0.3250, 0.0980];    % Bitcoin (orange)
                 [0.4660, 0.6740, 0.1880]};   % Bonds (green)
        
        % Create single panel plot
        plot(dates, sp500Prices, 'Color', colors{1}, 'LineWidth', 2);
        hold on;
        plot(dates, bitcoinPrices, 'Color', colors{2}, 'LineWidth', 2);
        plot(dates, bondPrices, 'Color', colors{3}, 'LineWidth', 2);
        
        % Add title and labels
        title('Asset Price Trends (2015-2025)', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Date', 'FontSize', 12);
        ylabel('Price (USD)', 'FontSize', 12);
        
        % Add legend
        legend('S&P 500 (^GSPC)', 'Bitcoin (BTC-USD)', 'Bonds (AGG)', ...
            'Location', 'best', 'Box', 'off', 'FontSize', 12);
        
        % Format dates using modern datetime formatting
        ax = gca;
        ax.XAxis.TickLabelFormat = 'yyyy';
        xtickangle(45);
        
        % Add grid
        grid on;
        
        % Add subtitle with data source
        subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
        
        % Add price annotations
        for i = 1:3
            switch i
                case 1
                    prices = sp500Prices;
                    label = 'S&P';
                    vertOffset = 0.5; % 15% above the line
                case 2
                    prices = bitcoinPrices;
                    label = 'BTC';
                    vertOffset = 0.02; % 5% above the line
                case 3
                    prices = bondPrices;
                    label = 'AGG';
                    vertOffset = 0; % On the line
            end
            
            % Calculate vertical offset based on price range
            priceRange = max(prices) - min(prices);
            vertShift = priceRange * vertOffset;
            
            % Start price (on the left with vertical offset)
            text(dates(1), prices(1) + vertShift, sprintf('%s: $%.2f', label, prices(1)), ...
                'Color', colors{i}, 'FontSize', 10, 'FontWeight', 'bold', ...
                'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
            
            % End price (on the right)
            text(dates(end), prices(end), sprintf('%s: $%.2f', label, prices(end)), ...
                'Color', colors{i}, 'FontSize', 10, 'FontWeight', 'bold', ...
                'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        end
        
        % Export figure
        exportgraphics(gcf, fileName, 'ContentType', 'vector', 'BackgroundColor', 'white');
        close(gcf);
    catch e
        error('plotAssetPriceTrends:PlottingError', 'Error during plotting: %s', e.message);
    end
end

function tt = readPriceCSV(fname, varName)
    % READPRICECSV Load CSV file into a timetable with date and price data.
    %   This function reads a CSV file containing financial price data and returns
    %   a timetable with properly formatted dates and price values.
    %
    %   Inputs:
    %       fname   - String, path to CSV file (e.g., 'bonds_data.csv')
    %       varName - String, column name for price data (e.g., 'Close')
    %
    %   Outputs:
    %       tt      - Timetable with Date and price columns
    %
    %   Example:
    %       tt = readPriceCSV('bitcoin_data.csv', 'Close');
    %
    %   See also: timetable, readtable, detectImportOptions
    
    % Input validation
    if nargin < 2
        error('readPriceCSV:NotEnoughInputs', 'Not enough input arguments. Usage: tt = readPriceCSV(fname, varName)');
    end
    
    if ~ischar(fname) && ~isstring(fname)
        error('readPriceCSV:InvalidInput', 'File name must be a string or character array');
    end
    
    if ~ischar(varName) && ~isstring(varName)
        error('readPriceCSV:InvalidInput', 'Variable name must be a string or character array');
    end
    
    try
        % Validate file existence
        if ~isfile(fname)
            error('readPriceCSV:FileNotFound', 'File %s does not exist.', fname);
        end
        
        % Determine asset-specific variable name
        if contains(lower(fname), 'bitcoin')
            assetVarName = 'Bitcoin';
        elseif contains(lower(fname), 'sp500')
            assetVarName = 'SP500';
        elseif contains(lower(fname), 'bonds')
            assetVarName = 'Bonds';
        else
            assetVarName = varName; % Fallback to varName if asset not recognised
        end
        
        % Load CSV with import options
        opts = detectImportOptions(fname);
        
        % Preserve original column headers
        opts.VariableNamingRule = 'preserve';
        
        % Check if Date column exists
        if ~any(strcmp(opts.VariableNames, 'Date'))
            error('readPriceCSV:MissingDateColumn', 'Date column not found in %s', fname);
        end
        
        % Set date format
        opts = setvaropts(opts, 'Date', 'InputFormat', 'dd/MM/yyyy');
        
        % Check if varName exists in the CSV
        if ~any(strcmp(opts.VariableNames, varName))
            error('readPriceCSV:ColumnNotFound', 'Column %s not found in %s. Available columns: %s', ...
                varName, fname, strjoin(opts.VariableNames, ', '));
        end
        
        % Set variable type and thousands separator
        opts = setvartype(opts, varName, 'double');
        opts = setvaropts(opts, varName, 'ThousandsSeparator', ',');
        
        % Read the table
        T = readtable(fname, opts);
        
        % Check if data was loaded successfully
        if height(T) == 0
            error('readPriceCSV:EmptyData', 'No data was loaded from %s', fname);
        end
        
        % Remove NaN rows
        T = T(~isnan(T.(varName)), :);
        
        % Check if any data remains after removing NaNs
        if height(T) == 0
            error('readPriceCSV:NoValidData', 'No valid data (non-NaN) found in %s', fname);
        end
        
        % Create timetable with asset-specific variable name
        tt = timetable(T.Date, T.(varName), 'VariableNames', {assetVarName});
        
        % Sort by date to ensure chronological order
        tt = sortrows(tt, 'Time');
        
    catch e
        % Rethrow the error with a more descriptive message
        error('readPriceCSV:Error', 'Failed to read %s: %s', fname, e.message);
    end
end

function plotHeatmap(correlationMatrix, assetNames, fileName)
    % PLOTHEATMAP Visualise the correlation matrix as a heatmap.
    %   This function creates a heatmap visualization of the correlation matrix
    %   between assets, with proper labeling and color scaling.
    %
    %   Inputs:
    %       correlationMatrix - NxN matrix of correlation coefficients
    %       assetNames       - Cell array of strings containing asset names
    %       fileName         - String filename for exporting the plot (e.g., 'correlation_heatmap.pdf')
    %
    %   Example:
    %       corrMatrix = [1, 0.5, -0.2; 0.5, 1, 0.3; -0.2, 0.3, 1];
    %       assetNames = {'Bitcoin', 'S&P 500', 'Bonds'};
    %       plotHeatmap(corrMatrix, assetNames, 'correlation_heatmap.pdf');
    %
    %   See also: PORTFOLIO_ANALYSIS_TEST, PLOTFRONTIER, PLOTMONTECARLO
    
    % Input validation
    if nargin < 3
        error('plotHeatmap:NotEnoughInputs', 'Not enough input arguments.');
    end
    
    if ~isnumeric(correlationMatrix) || ~ismatrix(correlationMatrix)
        error('plotHeatmap:InvalidInput', 'correlationMatrix must be a numeric matrix');
    end
    
    if ~iscellstr(assetNames) && ~isstring(assetNames)
        error('plotHeatmap:InvalidInput', 'assetNames must be a cell array of strings or a string array');
    end
    
    if ~ischar(fileName) && ~isstring(fileName)
        error('plotHeatmap:InvalidInput', 'fileName must be a string');
    end
    
    % Check matrix properties
    [rows, cols] = size(correlationMatrix);
    if rows ~= cols
        error('plotHeatmap:InvalidMatrix', 'correlationMatrix must be square');
    end
    
    if rows ~= length(assetNames)
        error('plotHeatmap:DimensionMismatch', 'Number of assets must match matrix dimensions');
    end
    
    try
        % Create figure with appropriate size
        figure('Position', [100; 100; 1200; 800]);
        
        % Set background color to white
        set(gcf, 'Color', 'white');
        
        % Create heatmap using imagesc
        imagesc(correlationMatrix);
        
        % Set colormap - vibrant red and blue scheme
        nColors = 64;
        blueColors = linspace(0, 1, nColors/2)';
        redColors = linspace(1, 0, nColors/2)';
        customColormap = [blueColors, zeros(nColors/2, 1), ones(nColors/2, 1); 
                          ones(nColors/2, 1), zeros(nColors/2, 1), redColors];
        colormap(customColormap);
        
        % Set color limits
        clim([-1 1]);
        
        % Add colorbar
        c = colorbar;
        c.Label.String = 'Correlation Coefficient';
        c.Label.FontSize = 12;
        c.Label.FontWeight = 'bold';
        
        % Add text labels for cells
        for i = 1:rows
            for j = 1:cols
                if abs(correlationMatrix(i,j)) > 0.5
                    textColor = 'white';
                else
                    textColor = 'black';
                end
                
                text(j, i, sprintf('%.2f', correlationMatrix(i,j)), ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'Color', textColor, ...
                    'FontSize', 12, ...
                    'FontWeight', 'bold');
            end
        end
        
        % Set axis labels
        set(gca, 'XTick', 1:cols, 'YTick', 1:rows);
        set(gca, 'XTickLabel', assetNames, 'YTickLabel', assetNames);
        set(gca, 'FontSize', 12, 'FontWeight', 'bold');
        xtickangle(45);
        
        % Add title and subtitle
        title('Asset Correlation Heatmap', 'FontSize', 14);
        subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
        
        % Export figure
        exportgraphics(gcf, fileName, 'ContentType', 'vector', 'BackgroundColor', 'white');
        close(gcf);
    catch e
        error('plotHeatmap:PlottingError', 'Error during plotting: %s', e.message);
    end
end

function plotFrontier(risks, returns, rf, assetRisk, assetReturn, assetNames, fileName)
    % PLOTFRONTIER Visualise the efficient frontier with optimal portfolios.
    %   This function creates a clean plot of the efficient frontier, highlighting
    %   the three optimal portfolios and their risk-return characteristics.
    %
    %   Inputs:
    %       risks       - Vector of portfolio risks (standard deviations)
    %       returns     - Vector of portfolio expected returns
    %       rf          - Risk-free rate
    %       assetRisk   - Vector of individual asset risks
    %       assetReturn - Vector of individual asset returns
    %       assetNames  - Cell array of strings containing asset names
    %       fileName    - String filename for exporting the plot
    
    % Input validation
    if nargin < 7
        error('plotFrontier:NotEnoughInputs', 'Not enough input arguments.');
    end
    
    try
        % Create figure with appropriate size
        figure('Position', [100; 100; 1000; 800]);
        set(gcf, 'Color', 'white');
        
        % Define colors
        frontierColor = [0.7; 0.7; 0.7];  % Light gray for frontier
        assetColor = [0.5; 0.5; 0.5];     % Gray for individual assets
        rfColor = [0.3; 0.3; 0.3];        % Dark gray for risk-free rate
        
        % Portfolio colors and markers
        portfolioColors = {[0.4660, 0.6740, 0.1880];  % Green for cautious
                          [0.9290, 0.6940, 0.1250];   % Yellow for balanced
                          [0.8500, 0.3250, 0.0980]};  % Red for bold
        portfolioMarkers = {'o', 's', 'd'};  % Circle, square, diamond
        portfolioNames = {'Cautious (1% BTC)', 'Balanced (5% BTC)', 'Bold (20% BTC)'};
        
        % Portfolio risk and return values
        portfolioRisks = [0.1799, 0.1856, 0.2605];  % Risk values
        portfolioReturns = [0.0394, 0.0739, 0.1646];  % Return values
        
        % Plot efficient frontier (light gray line)
        plot(risks * 100, returns * 100, '-', 'LineWidth', 1.5, 'Color', frontierColor);
        hold on;
        
        % Plot individual assets (gray dots)
        scatter(assetRisk * 100, assetReturn * 100, 100, 'filled', 'MarkerFaceColor', assetColor);
        
        % Add asset labels
        for i = 1:length(assetNames)
            text(assetRisk(i) * 100, assetReturn(i) * 100, ['  ' assetNames{i}], ...
                'FontSize', 10, 'FontWeight', 'normal', 'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle');
        end
        
        % Plot risk-free rate
        scatter(0, rf * 100, 100, 'filled', 'MarkerFaceColor', rfColor);
        text(0, rf * 100, '  Risk-Free Rate', ...
            'FontSize', 10, 'FontWeight', 'normal', 'HorizontalAlignment', 'left');
        
        % Plot the three portfolios with detailed labels and connecting lines
        for i = 1:3
            % Plot portfolio point
            scatter(portfolioRisks(i) * 100, portfolioReturns(i) * 100, 200, ...
                portfolioMarkers{i}, 'filled', 'MarkerFaceColor', portfolioColors{i}, ...
                'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
            
            % Find closest point on frontier
            [~, closestIdx] = min(abs(risks - portfolioRisks(i)));
            frontierRisk = risks(closestIdx) * 100;
            frontierReturn = returns(closestIdx) * 100;
            
            % Plot connecting line to frontier
            plot([portfolioRisks(i) * 100, frontierRisk], ...
                 [portfolioReturns(i) * 100, frontierReturn], ...
                 '--', 'Color', portfolioColors{i}, 'LineWidth', 1);
            
            % Create detailed label
            labelText = sprintf('%s\n%.1f%% Risk\n%.1f%% Return\n%.1f%% from Frontier', ...
                portfolioNames{i}, portfolioRisks(i)*100, portfolioReturns(i)*100, ...
                (portfolioReturns(i) - returns(closestIdx)) * 100);
            
            % Position label to avoid overlap
            if i == 1
                horzAlign = 'right';
                xOffset = -2;
            else
                horzAlign = 'left';
                xOffset = 2;
            end
            
            text(portfolioRisks(i) * 100 + xOffset, portfolioReturns(i) * 100, labelText, ...
                'FontSize', 10, 'FontWeight', 'normal', 'HorizontalAlignment', horzAlign, ...
                'VerticalAlignment', 'middle', 'BackgroundColor', 'white', ...
                'EdgeColor', 'none', 'Margin', 1);
        end
        
        % Add labels and formatting
        xlabel('Portfolio Risk (Annualized Standard Deviation, %)', 'FontSize', 12);
        ylabel('Expected Annual Return (%)', 'FontSize', 12);
        title('Efficient Frontier with Optimal Portfolios', 'FontSize', 14);
        subtitle('Source: Yahoo Finance (2025)', 'FontSize', 10);
        
        % Add explanation text
        text(0.02, 0.98, 'Dashed lines show distance from efficient frontier', ...
            'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'normal', ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
            'BackgroundColor', 'white', 'EdgeColor', 'none', 'Margin', 1);
        
        % Format axes
        ax = gca;
        ax.FontSize = 10;
        grid on;
        box on;
        
        % Ensure plot is rendered and UI elements are hidden
        drawnow;
        set(gcf, 'MenuBar', 'none', 'ToolBar', 'none');
        
        % Export figure
        exportgraphics(gcf, fileName, 'ContentType', 'vector', 'BackgroundColor', 'white');
        close(gcf);
    catch e
        error('plotFrontier:PlottingError', 'Error during plotting: %s', e.message);
    end
end

function plotMonteCarlo(simulatedEndValues, portfolioType, fileName)
    % Create figure with improved size ratio
    fig = figure('Position', [100; 100; 1000; 600]);
    set(fig, 'Color', 'white');
    
    % Calculate statistics
    var95 = quantile(simulatedEndValues, 0.05);
    var99 = quantile(simulatedEndValues, 0.01);
    medianVal = median(simulatedEndValues);
    meanVal = mean(simulatedEndValues);
    
    % Create histogram with improved styling
    histogram(simulatedEndValues, 'Normalization', 'probability', ...
        'FaceColor', [0.3010, 0.7450, 0.9330], ...  % Light blue color
        'EdgeColor', [0.2 0.2 0.2], ...
        'FaceAlpha', 0.8, ...
        'NumBins', 30);  % More bins for smoother distribution
    
    hold on;
    
    % Get axis limits for better text positioning
    yLims = ylim;
    yMax = yLims(2);
    xLims = xlim;
    xRange = xLims(2) - xLims(1);
    
    % Add vertical lines with improved styling
    lineProps = {'LineWidth', 2, 'LineStyle', '--'};
    line([var95 var95], [0 yMax], 'Color', [0.8500, 0.3250, 0.0980], lineProps{:});  % Orange
    line([var99 var99], [0 yMax], 'Color', [0.4940, 0.1840, 0.5560], lineProps{:});  % Purple
    line([medianVal medianVal], [0 yMax], 'Color', [0.4660, 0.6740, 0.1880], lineProps{:});  % Green
    line([meanVal meanVal], [0 yMax], 'Color', [0.9290, 0.6940, 0.1250], lineProps{:});  % Yellow
    
    % Add text annotations with improved formatting
    textBg = {'BackgroundColor', [1 1 1 0.8], ...
              'EdgeColor', [0.8 0.8 0.8], ...
              'Margin', 3};
    textProps = {'FontSize', 10, ...
                 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'left'};
    
    % Calculate vertical positions for text to avoid overlap
    yPositions = [0.95, 0.85, 0.75, 0.65];
    
    % Add annotations with actual values
    text(var95 + xRange*0.01, yMax*yPositions(1), sprintf('95%% VaR: £%.0f', var95), ...
        'Color', [0.8500, 0.3250, 0.0980], textProps{:}, textBg{:});
    text(var99 + xRange*0.01, yMax*yPositions(2), sprintf('99%% VaR: £%.0f', var99), ...
        'Color', [0.4940, 0.1840, 0.5560], textProps{:}, textBg{:});
    text(medianVal + xRange*0.01, yMax*yPositions(3), sprintf('Median: £%.0f', medianVal), ...
        'Color', [0.4660, 0.6740, 0.1880], textProps{:}, textBg{:});
    text(meanVal + xRange*0.01, yMax*yPositions(4), sprintf('Mean: £%.0f', meanVal), ...
        'Color', [0.9290, 0.6940, 0.1250], textProps{:}, textBg{:});
    
    % Improve axis labels and formatting
    xlabel('Portfolio Value (£)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Probability', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Monte Carlo Simulation Results - %s Portfolio', portfolioType), ...
        'FontSize', 14, 'FontWeight', 'bold');
    subtitle(sprintf('Source: Yahoo Finance (2025) | Sample Size: %d', ...
        length(simulatedEndValues)), 'FontSize', 10);
    
    % Format axes
    ax = gca;
    ax.FontSize = 11;
    ax.XAxis.Exponent = 4;  % Force scientific notation with 10^4
    
    % Set y-axis format using standard MATLAB formatting
    ytickformat('%.2f');
    
    % Improve grid appearance
    grid on;
    ax.GridAlpha = 0.15;
    ax.MinorGridAlpha = 0.1;
    ax.Layer = 'top';  % Ensure grid lines are below the plot
    box on;
    
    % Adjust figure margins
    ax.TightInset;
    ax.LooseInset;
    
    % Ensure plot is rendered and UI elements are hidden
    drawnow;
    set(fig, 'MenuBar', 'none', 'ToolBar', 'none');
    
    % Export with high resolution
    exportgraphics(fig, fileName, 'ContentType', 'vector', 'BackgroundColor', 'white', 'Resolution', 300);
    close(fig);
end