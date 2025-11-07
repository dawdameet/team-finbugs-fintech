/*
 * backtest.cpp
 *
 * C++ implementation of the pairs trading backtest.
 * This file contains all the C++ logic (using std::vector, std::deque, etc.)
 * and exposes a single C-compatible function defined in backtest.h.
 */

#include "backtest.h" // Include the C API header

#include <iostream>
#include <vector>
#include <string>
#include <cmath>        // For std::sqrt, std::pow
#include <numeric>      // For std::accumulate
#include <fstream>      // For std::ifstream (file input)
#include <sstream>      // For std::stringstream (parsing lines)
#include <deque>        // For the moving window
#include <iomanip>      // For std::setprecision

/**
 * @struct StockData
 * @brief Holds the simulated prices for our two correlated stocks.
 * (Internal C++ struct, not exposed in the API)
 */
struct StockData {
    double priceA;
    double priceB;
};

/**
 * @brief Calculates the mean and standard deviation of a deque of doubles.
 * (Internal C++ helper function)
 *
 * @param window The deque containing the data for the moving window.
 * @return std::pair<double, double> A pair containing {mean, std_dev}.
 */
std::pair<double, double> calculateMeanAndStdDev(const std::deque<double>& window) {
    if (window.empty()) return {0.0, 0.0};

    double sum = std::accumulate(window.begin(), window.end(), 0.0);
    double mean = sum / window.size();

    double sq_sum = 0.0;
    for (double val : window) {
        sq_sum += std::pow(val - mean, 2);
    }
    // Note: This is population std_dev (divide by N).
    // For sample std_dev, use window.size() - 1.
    // For trading, consistency is more important than the specific formula.
    double std_dev = std::sqrt(sq_sum / window.size());

    return {mean, std_dev};
}

/**
 * @brief Loads stock data from a CSV file.
 * (Internal C++ helper function)
 *
 * @param filename The name of the CSV file to read.
 * @return std::vector<StockData> A vector of the parsed stock data.
 */
std::vector<StockData> loadDataFromCSV(const std::string& filename) {
    std::vector<StockData> all_data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return all_data; // Return empty vector
    }

    std::string line;
    // Skip the header row
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        StockData data_point;

        try {
            // Get priceA
            std::getline(ss, field, ',');
            data_point.priceA = std::stod(field);
            // Get priceB
            std::getline(ss, field, ',');
            data_point.priceB = std::stod(field);

            all_data.push_back(data_point);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Skipping bad line: " << line << "\n";
        }
    }

    file.close();
    return all_data;
}


/**
 * @brief Main C-API function to run the backtest.
 * This function was converted from the original main().
 */
extern "C" double run_backtest(int lookback_window,
                             double std_dev_threshold,
                             const char* data_filename)
{
    // --- 1. Set Parameters ---
    // Parameters are now passed in
    const int LOOKBACK_WINDOW = lookback_window;
    const double STD_DEV_THRESHOLD = std_dev_threshold;
    // Convert C-string to C++ std::string for internal use
    const std::string DATA_FILENAME = data_filename;

    // --- 2. Load Data ---
    std::cout << "Loading data from " << DATA_FILENAME << "...\n";
    std::vector<StockData> all_data = loadDataFromCSV(DATA_FILENAME);

    if (all_data.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 0.0; // Return 0.0 PnL on error
    }
    std::cout << "Data loading complete. " << all_data.size() << " rows loaded.\n\n";

    std::deque<double> spreadWindow; // Holds the most recent 'LOOKBACK_WINDOW' spreads

    // Position state: 0 = Flat, 1 = Long (Bought A, Sold B), -1 = Short (Sold A, Bought B)
    int position = 0;
    double pnl = 0.0; // Profit and Loss
    double entryPrice = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Running Backtest (Window=" << LOOKBACK_WINDOW
              << ", Threshold=" << STD_DEV_THRESHOLD << ") ---\n";

    // --- 3. Simulation Loop ---
    for (int i = 0; i < all_data.size(); ++i) {
        double priceA = all_data[i].priceA;
        double priceB = all_data[i].priceB;
        double spread = priceA / priceB;

        spreadWindow.push_back(spread);
        
        if (spreadWindow.size() >= LOOKBACK_WINDOW) {
            spreadWindow.pop_front();
        }

        // We can only trade once our moving average window is full
        if (i < LOOKBACK_WINDOW) {
            continue; // Wait for window to fill
        }

        // Calculate Bollinger Bands
        auto [mean, std_dev] = calculateMeanAndStdDev(spreadWindow);

        // Avoid trading if std_dev is zero (e.g., all prices in window are same)
        if (std_dev == 0.0) {
            continue;
        }

        double upperBound = mean + (STD_DEV_THRESHOLD * std_dev);
        double lowerBound = mean - (STD_DEV_THRESHOLD * std_dev);

        // --- 4. Trading Logic ---

        if (position == 0) {
            if (spread < lowerBound) {
<<<<<<< HEAD
                position = -1;
                entryPrice = spread;
                std::cout << "Day " << i << ": LONG  Spread at " << spread
                          << " (Mean=" << mean << ", Lower=" << lowerBound << ")\n";
            } else if (spread > upperBound) {
                position = 1;
=======
                // Spread is too cheap. Let's... short it?
                position = 1;
                entryPrice = spread; // We enter at the current spread
                std::cout << "Day " << i << ": LONG  Spread at " << spread
                          << " (Mean=" << mean << ", Lower=" << lowerBound << ")\n";
            } else if (spread > upperBound) {
                // Spread is too expensive. Let's... long it?
                position = -1;
>>>>>>> 01a52af94f048173ad522fc0d7bbbe2d148fdc63
                entryPrice = spread;
                std::cout << "Day " << i << ": SHORT Spread at " << spread
                          << " (Mean=" << mean << ", Upper=" << upperBound << ")\n";
            }
        }
        else if (position == 1 && spread >= mean) {
            double exitPrice = spread;
<<<<<<< HEAD
            double tradePnl = (entryPrice - exitPrice) / entryPrice;
=======
            double tradePnl = (entryPrice - exitPrice) / entryPrice; // PnL as %
>>>>>>> 01a52af94f048173ad522fc0d7bbbe2d148fdc63
            pnl += tradePnl;
            std::cout << "Day " << i << ": EXIT LONG at " << exitPrice
                      << " | Trade PnL: " << (tradePnl * 100.0) << "%\n";
            position = 0;
        } else if (position == -1 && spread <= mean) {
            double exitPrice = spread;
            double tradePnl = (exitPrice - entryPrice) / entryPrice;
            pnl += tradePnl;
            std::cout << "Day " << i << ": EXIT SHORT at " << exitPrice
                      << " | Trade PnL: " << (tradePnl * 100.0) << "%\n";
            position = 0;
        }
    }

    std::cout << "\n--- Backtest Complete --- \n";
    double finalPnlPerc = pnl * 100.0;
    std::cout << "Final Cumulative PnL: " << finalPnlPerc << "%\n";

    // Return the final PnL to the calling program (Python)
    return finalPnlPerc;
}

