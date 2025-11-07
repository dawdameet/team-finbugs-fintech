#!/bin/bash

# Create labels for bugs 11-20
gh label create bug-11 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 11" 2>/dev/null || true
gh label create bug-12 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 12" 2>/dev/null || true
gh label create bug-13 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 13" 2>/dev/null || true
gh label create bug-14 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 14" 2>/dev/null || true
gh label create bug-15 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 15" 2>/dev/null || true
gh label create bug-16 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 16" 2>/dev/null || true
gh label create bug-17 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 17" 2>/dev/null || true
gh label create bug-18 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 18" 2>/dev/null || true
gh label create bug-19 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 19" 2>/dev/null || true
gh label create bug-20 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 20" 2>/dev/null || true

# Bug 11
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 11: Incorrect RSI Calculation Formula" \
  -b "## Bug Description
The RSI (Relative Strength Index) feature is computed as NaN or nonsensical negative values. This corrupts the feature data for all ML models, leading to poor predictions.

## Expected Behavior
RSI should be a value between 0 and 100, calculated from average gains and average losses over a window period.

## Current Behavior
The loss component is calculated incorrectly, producing negative values where positive values are needed, breaking the RSI formula.

## Files Affected
\`stock_price_predictor/predictor.py\`

## Difficulty: HARD **Points: 30**

## Reproduction Steps
1. Run the predictor and examine the features DataFrame
2. Notice RSI values are NaN or negative
3. Examine the create_features method
4. Look at the RSI calculation
5. Understand that RSI needs average gain and average loss
6. Check the 'loss' calculation using .where()
7. Identify that loss values are negative when they should be positive
8. Negate the loss expression

## Solution
Change 'loss = (delta.where(delta < 0, 0)).rolling(window=14).mean()' to 'loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()'

## Hints
Negate the expression: -delta.where(delta < 0, 0)

---
*Bug ID: 11 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-11,domain-fintech,difficulty-hard"

# Bug 12
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 12: Swapped Sharpe Ratio Components" \
  -b "## Bug Description
The Sharpe ratio calculation has its numerator and denominator swapped. Instead of risk-adjusted return (return/volatility), the code calculates volatility-adjusted risk (volatility/return), which is meaningless.

## Expected Behavior
Sharpe ratio should be (mean return - risk-free rate) / standard deviation of returns.

## Current Behavior
The code computes std / mean instead of mean / std, inverting the entire metric.

## Files Affected
\`stock_price_predictor/backtest_strategy.py\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run the backtest
2. Observe that Sharpe ratios are abnormally high for low returns
3. Examine the calculate_sharpe_ratio method
4. Identify the swapped formula

## Solution
Change 'return std / mean' to 'return mean / std'

## Hints
Sharpe = (mean return - risk-free rate) / volatility, not the inverse.

---
*Bug ID: 12 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-12,domain-fintech,difficulty-medium"

# Bug 13
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 13: Max Drawdown Uses Min Instead of Max" \
  -b "## Bug Description
The maximum drawdown metric is calculated using min() instead of max(), resulting in the minimum drawdown being reported instead of maximum.

## Expected Behavior
Maximum drawdown should be the largest peak-to-trough decline in portfolio value.

## Current Behavior
The code uses min() on the drawdown series, which finds the smallest (best) drawdown instead of the largest (worst).

## Files Affected
\`stock_price_predictor/backtest_strategy.py\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run a backtest
2. Observe that max drawdown is suspiciously small
3. Examine the calculate_max_drawdown method
4. Notice the use of min() instead of max()

## Solution
Change 'return drawdown.min()' to 'return drawdown.max()'

## Hints
Maximum drawdown needs max(), not min().

---
*Bug ID: 13 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-13,domain-fintech,difficulty-easy"

# Bug 14
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 14: Prediction Index Off by One" \
  -b "## Bug Description
The predicted prices are aligned with the wrong dates in the backtest. Today's prediction uses tomorrow's actual data, creating lookahead bias.

## Expected Behavior
Prediction for day T should use data up to day T-1 and be compared against day T's actual price.

## Current Behavior
The code uses iloc[i] for prediction but iloc[i-1] for the actual price, misaligning the comparison by one day.

## Files Affected
\`stock_price_predictor/backtest_strategy.py\`

## Difficulty: EXTREME **Points: 40**

## Reproduction Steps
1. Run the backtest
2. Notice unusually good prediction accuracy
3. Examine the backtest loop carefully
4. Trace the indices used for prediction vs actual
5. Identify the off-by-one error

## Solution
Change 'actual = test_data['Close'].iloc[i-1]' to 'actual = test_data['Close'].iloc[i]'

## Hints
The prediction at index i should compare against actual at index i.

---
*Bug ID: 14 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-14,domain-fintech,difficulty-extreme"

# Bug 15
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 15: Portfolio Value Compounds Incorrectly" \
  -b "## Bug Description
The portfolio value calculation multiplies by (1 + daily_return) every day, but uses cumulative returns instead of daily returns, causing exponential growth errors.

## Expected Behavior
Portfolio value should compound using daily returns: value *= (1 + daily_return).

## Current Behavior
The code uses cumulative returns in the compounding formula, causing the portfolio to grow unrealistically.

## Files Affected
\`portfolio/rebalancer.py\`

## Difficulty: DIFFICULT **Points: 30**

## Reproduction Steps
1. Run the portfolio rebalancer
2. Notice portfolio value grows unrealistically
3. Examine the update_portfolio_value method
4. Check what value is being used in the compounding
5. Identify that cumulative return is used instead of daily return

## Solution
Change 'self.portfolio_value *= (1 + cumulative_return)' to 'self.portfolio_value *= (1 + daily_return)'

## Hints
Use daily_return, not cumulative_return.

---
*Bug ID: 15 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-15,domain-fintech,difficulty-difficult"

# Bug 16
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 16: Swapped Risk Percentile Labels" \
  -b "## Bug Description
The 5th and 95th percentile labels are swapped in the risk metrics output. The p5_final key contains the 95th percentile value and vice versa.

## Expected Behavior
p5_final should contain the 5th percentile (worst-case scenario), p95_final should contain the 95th percentile (best-case scenario).

## Current Behavior
The p5 variable is assigned to p95_final and the p95 variable is assigned to p5_final.

## Files Affected
\`portfolio/risk_manager.py\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run risk analysis
2. Notice p5_final is larger than p95_final (should be opposite)
3. Examine the return statement in calculate_var method
4. Identify the swapped assignments

## Solution
Swap the variable assignments in the return dictionary.

## Hints
The p5 variable is being written to p95_final and vice versa.

---
*Bug ID: 16 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-16,domain-fintech,difficulty-easy"

# Bug 17
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 17: Inverted PnL Calculation for Short Trades" \
  -b "## Bug Description
The profit and loss (PnL) for short trades is calculated incorrectly, resulting in inverted gains/losses for short positions. Profitable short trades appear as losses and vice versa.

## Expected Behavior
Short trade PnL should be (entry - exit) / entry, since you profit when price falls after shorting.

## Current Behavior
The code uses the long trade formula (exit - entry) / entry for short trades, inverting the signs.

## Files Affected
\`cpp_services/pair_trading/backtest.cpp\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run the pair trading backtest
2. Observe that short trades show incorrect PnL signs
3. Examine the trade exit logic
4. Find the PnL calculation for short positions
5. Identify that it uses the long formula
6. Invert the formula for short trades

## Solution
Change 'double tradePnl = (exitPrice - entryPrice) / entryPrice;' to 'double tradePnl = (entryPrice - exitPrice) / entryPrice;'

## Hints
Look for where position == -1 (short).

---
*Bug ID: 17 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-17,domain-fintech,difficulty-easy"

# Bug 18
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 18: Flipped Mean-Reversion Entry Logic" \
  -b "## Bug Description
The trading strategy implements the opposite of mean-reversion. It shorts when the spread is low (should long) and longs when the spread is high (should short). The strategy loses money systematically.

## Expected Behavior
Mean-reversion: go LONG when spread < lower bound (spread will revert up), go SHORT when spread > upper bound (spread will revert down).

## Current Behavior
The position assignments are backwards: SHORT when spread < lower bound, LONG when spread > upper bound.

## Files Affected
\`cpp_services/pair_trading/backtest.cpp\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run the backtest and observe poor/negative returns
2. Understand mean-reversion strategy logic
3. Examine the entry signal conditions
4. Check what position is assigned when spread < lowerBound
5. Check what position is assigned when spread > upperBound
6. Identify that these are backwards
7. Swap the position assignments

## Solution
Swap the position values: -1 becomes 1 and 1 becomes -1 in the entry conditions.

## Hints
Swap the 1 and -1 assignments.

---
*Bug ID: 18 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-18,domain-fintech,difficulty-medium"

# Bug 19
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 19: Off-by-One Window Size Error" \
  -b "## Bug Description
The moving average and standard deviation are calculated on a window of N-1 items instead of N items. This skews all trading signals because the statistics are based on incomplete windows.

## Expected Behavior
The window should contain exactly LOOKBACK_WINDOW items for statistical calculations.

## Current Behavior
Items are removed from the window too early, when size >= N instead of size > N, causing the window to max out at N-1 items.

## Files Affected
\`cpp_services/pair_trading/backtest.cpp\`

## Difficulty: DIFFICULT **Points: 30**

## Reproduction Steps
1. Observe that trading signals seem slightly off
2. Examine the window management code
3. Look at the condition for removing old items
4. Trace through: when does the first item get removed?
5. Realize it's removed right after adding the Nth item
6. Understand this keeps window size at N-1
7. Change >= to >

## Solution
Change 'if (spreadWindow.size() >= LOOKBACK_WINDOW)' to 'if (spreadWindow.size() > LOOKBACK_WINDOW)'

## Hints
This is a classic off-by-one error.

---
*Bug ID: 19 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-19,domain-fintech,difficulty-difficult"

# Bug 20
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 20: Lookahead Bias in Signal Calculation" \
  -b "## Bug Description
The backtest is invalid because trading decisions for day 't' use data from day 't' itself. This is lookahead bias - using future information that wouldn't be available at decision time.

## Expected Behavior
Signals for day 't' should be calculated using only data up to day 't-1'. Then the decision is made, and day 't' data is added after.

## Current Behavior
Day 't' spread is added to the window BEFORE calculating signals, meaning the signal uses information from the current day.

## Files Affected
\`cpp_services/pair_trading/backtest.cpp\`

## Difficulty: EXTREME **Points: 40**

## Reproduction Steps
1. Understand the temporal sequence of a backtest
2. Examine the main simulation loop carefully
3. Identify the order of operations: add to window, calculate signals, make decisions
4. Realize that when signals are calculated, today's spread is already in the window
5. Understand this means the decision uses future information
6. Restructure the loop: calculate signals first, make decisions, THEN add today's data

## Solution
Move spreadWindow.push_back(spread) to after all trading logic.

## Hints
Move spreadWindow.push_back(spread) to after all trading logic.

---
*Bug ID: 20 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-20,domain-fintech,difficulty-extreme"

echo "All issues created successfully!"
