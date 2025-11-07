#!/bin/bash

# Create labels for bugs 1-10
gh label create bug-1 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 1" 2>/dev/null || true
gh label create bug-2 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 2" 2>/dev/null || true
gh label create bug-3 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 3" 2>/dev/null || true
gh label create bug-4 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 4" 2>/dev/null || true
gh label create bug-5 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 5" 2>/dev/null || true
gh label create bug-6 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 6" 2>/dev/null || true
gh label create bug-7 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 7" 2>/dev/null || true
gh label create bug-8 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 8" 2>/dev/null || true
gh label create bug-9 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 9" 2>/dev/null || true
gh label create bug-10 -R dawdameet/team-finbugs-fintech --color d73a4a --description "Bug ID 10" 2>/dev/null || true

# Bug 1
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 1: Incorrect VADER Sentiment Threshold" \
  -b "## Bug Description
The sentiment analysis is heavily skewed towards 'neutral'. Even clearly positive headlines (e.g., 'stock surges') are classified as neutral. This results in an inaccurate VADER distribution and a 'neutral' overall sentiment, even when the news is good.

## Expected Behavior
Positive headlines should be classified as 'positive' with appropriate VADER compound scores. The threshold for positive sentiment should correctly identify bullish news.

## Current Behavior
Headlines with positive sentiment are being classified as neutral. The VADER distribution shows an overwhelming majority of 'neutral' classifications even for clearly positive/negative news.

## Files Affected
\`market_sentiment/sentiment_analyzer.py\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run the sentiment analyzer on sample headlines
2. Observe that positive headlines are classified as neutral
3. Check the VADER compound score threshold in analyze_text_vader method
4. Identify that the positive threshold is set incorrectly
5. Adjust the threshold to match VADER's standard thresholds

## Solution
Change 'if compound >= 0.5:' to 'if compound >= 0.05:'

## Hints
Look at in the analyze_text_vader method

---
*Bug ID: 1 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-1,domain-fintech,difficulty-easy"

# Bug 2
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 2: Reversed Stopword Filter Logic" \
  -b "## Bug Description
The word cloud and 'Top 10 Keywords' list show only common English stop words like 'and', 'the', 'for', 'is', etc., instead of relevant financial terms. The keyword extraction is completely broken.

## Expected Behavior
Keywords should be meaningful financial and domain-specific terms like 'stock', 'earnings', 'revenue', 'market', etc. Stop words should be filtered out.

## Current Behavior
The keyword list and word cloud contain only stop words (and, the, for, is, of, to). All meaningful words are being removed.

## Files Affected
\`market_sentiment/sentiment_analyzer.py\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run the sentiment analyzer and examine the word cloud
2. Notice that only stop words appear in the word cloud
3. Check the Top 10 Keywords output
4. Examine the extract_keywords method
5. Look at the list comprehension that filters words
6. Identify that the filter logic is reversed

## Solution
Change 'all_words.extend([w for w in words if w in stop_words and len(w) > 3])' to 'all_words.extend([w for w in words if w not in stop_words and len(w) > 3])'

## Hints
The 'in' operator needs to become 'not in'

---
*Bug ID: 2 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-2,domain-fintech,difficulty-medium"

# Bug 3
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 3: Overly Aggressive Regex Pattern" \
  -b "## Bug Description
The word cloud shows malformed, jumbled 'words' that are actually multiple words concatenated together (e.g., 'teslastocksurgesonrecord...'). The Top 10 Keywords list is empty or shows these same jumbled strings.

## Expected Behavior
Words should be properly separated and individual. The regex should clean special characters while preserving word boundaries.

## Current Behavior
All spaces are being removed from text, causing all words in a headline to be mashed together into one giant string.

## Files Affected
\`market_sentiment/sentiment_analyzer.py\`

## Difficulty: HARD **Points: 30**

## Reproduction Steps
1. Run the analyzer and observe jumbled words in the word cloud
2. Examine the clean_text method
3. Look at the regex pattern being used
4. Identify that the pattern removes ALL non-letter characters
5. Realize that spaces are non-letter characters and are being removed
6. Modify the regex to preserve whitespace

## Solution
Change 'text = re.sub(r'[^A-Za-z]', '', text)' to 'text = re.sub(r'[^A-Za-z\\\\s]', '', text)'

## Hints
The pattern should be r'[^A-Za-z\\\\s]' to keep letters AND spaces

---
*Bug ID: 3 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-3,domain-fintech,difficulty-hard"

# Bug 4
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 4: Headlines List Overwrite" \
  -b "## Bug Description
The Yahoo News analysis shows every single headline as identical - just N copies of the last headline fetched. The entire news DataFrame and subsequent analysis is worthless because it's analyzing one headline 50 times.

## Expected Behavior
Each row in the news DataFrame should contain a different headline from the Yahoo Finance API.

## Current Behavior
All headlines in the DataFrame are identical, showing only the last headline fetched from the API repeated multiple times.

## Files Affected
\`market_sentiment/sentiment_analyzer.py\`

## Difficulty: EXTREME **Points: 40**

## Reproduction Steps
1. Run the analyzer and examine the news DataFrame output
2. Notice all headlines are identical
3. Examine the fetch_yahoo_news method carefully
4. Find where headlines list is built in the loop
5. Look for any code after the loop that modifies the headlines list
6. Identify the line that overwrites the entire list
7. Remove the problematic line

## Solution
Change 'headlines = [title] * len(headlines)' to '# Remove this line entirely'

## Hints
This line should be completely removed, not modified

---
*Bug ID: 4 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-4,domain-fintech,difficulty-extreme"

# Bug 5
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 5: Incorrect VaR Percentile Calculation" \
  -b "## Bug Description
The Value at Risk (VaR) calculation uses the wrong percentile. For 95% confidence VaR, it should use the 5th percentile (bottom 5%), but the code uses the 95th percentile (top 5%), giving completely opposite risk estimates.

## Expected Behavior
VaR at 95% confidence should show the 5th percentile loss (the threshold where 5% of outcomes are worse).

## Current Behavior
The code calculates the 95th percentile instead of the 5th percentile, showing best-case scenarios instead of worst-case risk.

## Files Affected
\`portfolio/risk_manager.py\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run the risk manager
2. Notice VaR values are positive/optimistic instead of showing risk
3. Examine the calculate_var method
4. Check the percentile being calculated
5. Identify that 95th percentile is used instead of 5th

## Solution
Change 'np.percentile(returns, 95)' to 'np.percentile(returns, 5)'

## Hints
VaR at X% confidence uses the (100-X)th percentile.

---
*Bug ID: 5 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-5,domain-fintech,difficulty-medium"

# Bug 6
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 6: Correlation Matrix Uses Wrong Axis" \
  -b "## Bug Description
The correlation matrix is calculated along the wrong axis, producing a time-series correlation instead of asset correlation. The matrix dimensions are wrong and the correlations are meaningless.

## Expected Behavior
Correlation should be calculated across time for each pair of assets (axis=0).

## Current Behavior
The code uses axis=1, calculating correlation across assets for each time point, which is nonsensical.

## Files Affected
\`portfolio/risk_manager.py\`

## Difficulty: HARD **Points: 30**

## Reproduction Steps
1. Run portfolio analysis
2. Notice correlation matrix has wrong dimensions
3. Examine the calculate_correlation_matrix method
4. Check the axis parameter in the correlation calculation
5. Identify that axis=1 is used instead of axis=0

## Solution
Change '.corr(axis=1)' to '.corr(axis=0)' or just '.corr()' (default is axis=0)

## Hints
Correlation should be calculated down the time axis (axis=0), not across assets.

---
*Bug ID: 6 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-6,domain-fintech,difficulty-hard"

# Bug 7
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 7: Sharpe Ratio Missing Annualization Factor" \
  -b "## Bug Description
The Sharpe ratio is calculated using daily returns but reported as if it were annualized. The values are too low by a factor of sqrt(252).

## Expected Behavior
Sharpe ratio should be annualized: multiply by sqrt(252) for daily data or sqrt(12) for monthly data.

## Current Behavior
The code calculates Sharpe from daily returns without annualizing, making all Sharpe ratios appear artificially low.

## Files Affected
\`portfolio/optimizer.py\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run portfolio optimization
2. Notice Sharpe ratios are suspiciously low (< 1 for good portfolios)
3. Examine the calculate_sharpe method
4. Check if annualization is applied
5. Identify missing sqrt(252) factor

## Solution
Change 'return mean / std' to 'return (mean / std) * np.sqrt(252)'

## Hints
Multiply by np.sqrt(252) for daily returns.

---
*Bug ID: 7 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-7,domain-fintech,difficulty-medium"

# Bug 8
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 8: Portfolio Weights Don't Sum to One" \
  -b "## Bug Description
The optimized portfolio weights don't sum to 1.0 (100%). They sum to a random value, making the portfolio allocation invalid.

## Expected Behavior
Portfolio weights should be normalized to sum to exactly 1.0 before returning.

## Current Behavior
The weights are returned without normalization, causing sum != 1.0.

## Files Affected
\`portfolio/optimizer.py\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run portfolio optimization
2. Sum the returned weights
3. Notice they don't equal 1.0
4. Examine the optimize_portfolio method
5. Check if weights are normalized before return
6. Add normalization step

## Solution
Add 'weights = weights / weights.sum()' before returning weights

## Hints
Normalize weights by dividing by their sum.

---
*Bug ID: 8 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-8,domain-fintech,difficulty-easy"

# Bug 9
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 9: Covariance Matrix Uses Variance Instead" \
  -b "## Bug Description
The covariance matrix calculation uses .var() instead of .cov(), producing a 1D variance array instead of a 2D covariance matrix. This crashes the optimization.

## Expected Behavior
Use .cov() to calculate the covariance matrix for portfolio optimization.

## Current Behavior
The code calls .var(), which returns variances only, not covariances.

## Files Affected
\`portfolio/optimizer.py\`

## Difficulty: EASY **Points: 10**

## Reproduction Steps
1. Run portfolio optimization
2. Observe crash or dimension error
3. Examine the calculate_covariance method
4. Notice .var() is called instead of .cov()

## Solution
Change 'returns.var()' to 'returns.cov()'

## Hints
Use .cov() not .var()

---
*Bug ID: 9 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-9,domain-fintech,difficulty-easy"

# Bug 10
gh issue create -R dawdameet/team-finbugs-fintech \
  -t "Bug 10: Rebalancing Threshold Inverted" \
  -b "## Bug Description
The rebalancing logic is inverted. The portfolio rebalances when drift is BELOW the threshold instead of ABOVE it, causing excessive rebalancing.

## Expected Behavior
Rebalance when abs(current_weight - target_weight) > threshold.

## Current Behavior
The code uses < instead of >, rebalancing constantly even when weights are close to target.

## Files Affected
\`portfolio/rebalancer.py\`

## Difficulty: MEDIUM **Points: 20**

## Reproduction Steps
1. Run the rebalancer
2. Notice excessive rebalancing even with small drifts
3. Examine the should_rebalance method
4. Check the comparison operator
5. Identify that < is used instead of >

## Solution
Change 'if drift < threshold:' to 'if drift > threshold:'

## Hints
Rebalance when drift EXCEEDS threshold, not when it's below.

---
*Bug ID: 10 | Domain: fintech | Status: UNRESOLVED*" \
  --label "bug-10,domain-fintech,difficulty-medium"

echo "All bugs 1-10 issues created successfully!"
