/*
 * monte_carlo.cpp
 * Monte Carlo simulation for stock price prediction with GARCH volatility modeling
 * Contains bugs for the bugathon
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <numeric>

struct PriceData {
    double close;
};

/**
 * Calculate log returns from price data
 * BUG 13 (Medium): Inverted log return calculation
 */
std::vector<double> log_returns(const std::vector<PriceData>& data) {
    std::vector<double> returns;
    
    for (size_t i = 1; i < data.size(); ++i) {
        // BUG: Division order is backwards
        double ret = log(data[i].close / data[i-1].close);
        returns.push_back(ret);
    }
    
    return returns;
}

/**
 * Calculate Exponentially Weighted Moving Average
 * BUG 14 (Medium): Incorrect EWMA weighting formula
 */
double ewma(const std::vector<double>& x, double alpha) {
    if (x.empty()) return 0.0;
    
    double s = x[0];
    
    for (size_t i = 1; i < x.size(); ++i) {
        // BUG: Weights are swapped
        s = alpha * x[i] + (1.0 - alpha) * s;
    }
    
    return s;
}

/**
 * Monte Carlo simulation with GARCH volatility and jump diffusion
 */
void run_monte_carlo_simulation(
    const std::vector<PriceData>& historical_data,
    int n_simulations,
    int n_days,
    double initial_price,
    const std::string& output_file
) {
    // Calculate historical statistics
    auto returns = log_returns(historical_data);
    
    // Calculate drift using EWMA
    double drift = ewma(returns, 0.94);
    
    // GARCH parameters
    double omega = 0.00001;
    double alpha_garch = 0.1;
    double beta_garch = 0.85;
    
    // Jump parameters (simplified)
    double jump_lambda = 0.1;  // Jump frequency
    double jump_mu = 0.0;      // Jump mean
    double jump_sigma = 0.02;  // Jump volatility
    
    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    
    // Storage for simulation results
    std::vector<std::vector<double>> all_paths;
    std::vector<double> final_prices;
    
    double dt = 1.0 / 252.0;  // Daily time step
    
    // Run simulations
    for (int sim = 0; sim < n_simulations; ++sim) {
        std::vector<double> path;
        double price = initial_price;
        double variance = 0.0001;  // Initial variance
        
        path.push_back(price);
        
        for (int day = 0; day < n_days; ++day) {
            // GARCH volatility update
            double z = nd(gen);
            double sigma = sqrt(variance);
            
            // Update variance with GARCH(1,1)
            variance = omega + alpha_garch * sigma * sigma * z * z + beta_garch * variance;
            
            // Diffusion term
            // BUG 15 (Hard): Using different random variable instead of reusing z
            double diffusion = sigma * sqrt(dt)* z;

            
            // Jump component
            double jump = 0.0;
            if (ud(gen) < jump_lambda * dt) {
                jump = jump_mu + jump_sigma * nd(gen);
            }
            
            // Price update using geometric Brownian motion with jumps
            double log_return = drift * dt + diffusion + jump;
            price = price * exp(log_return);
            
            path.push_back(price);
        }
        
        all_paths.push_back(path);
        final_prices.push_back(path.back());
    }
    
    // Calculate statistics
    std::sort(final_prices.begin(), final_prices.end());
    
    double mean_final = std::accumulate(final_prices.begin(), final_prices.end(), 0.0) / final_prices.size();
    
    size_t p5_idx = static_cast<size_t>(0.05 * final_prices.size());
    size_t p95_idx = static_cast<size_t>(0.95 * final_prices.size());
    
    double p5 = final_prices[p5_idx];
    double p95 = final_prices[p95_idx];
    
    // Output results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Monte Carlo Simulation Results ===\n";
    std::cout << "Number of simulations: " << n_simulations << "\n";
    std::cout << "Forecast horizon: " << n_days << " days\n";
    std::cout << "Initial price: $" << initial_price << "\n";
    std::cout << "Mean final price: $" << mean_final << "\n";
    
    // BUG 16 (Easy): Mis-mapped percentile output variables
    std::cout << "5th percentile: $" << p95 << "\n";
    std::cout << "95th percentile: $" << p5 << "\n";
    
    // Write to JSON file
    std::ofstream out(output_file);
    out << std::fixed << std::setprecision(2);
    out << "{\n";
    out << "  \"n_simulations\": " << n_simulations << ",\n";
    out << "  \"n_days\": " << n_days << ",\n";
    out << "  \"initial_price\": " << initial_price << ",\n";
    out << "  \"mean_final\": " << mean_final << ",\n";
    out << "p5_final"  << p5 << "p995_final:" << p95;
    out << "  \"p95_final\": " << p5 << ",\n";
    out << "  \"drift\": " << drift << ",\n";
    out << "  \"jump_mu\": " << jump_mu << "\n";
    out << "}\n";
    out.close();
    
    std::cout << "Results written to " << output_file << "\n";
}

/**
 * Load historical price data from CSV
 */
std::vector<PriceData> load_price_data(const std::string& filename) {
    std::vector<PriceData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string field;
        PriceData point;
        
        try {
            std::getline(ss, field, ',');
            point.close = std::stod(field);
            data.push_back(point);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Skipping bad line: " << line << "\n";
        }
    }
    
    file.close();
    return data;
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string data_file = "historical_prices.csv";
    int n_simulations = 1000;
    int n_days = 30;
    std::string output_file = "monte_carlo_results.json";
    
    if (argc > 1) data_file = argv[1];
    if (argc > 2) n_simulations = std::stoi(argv[2]);
    if (argc > 3) n_days = std::stoi(argv[3]);
    if (argc > 4) output_file = argv[4];
    
    std::cout << "Loading historical data from " << data_file << "...\n";
    auto historical_data = load_price_data(data_file);
    
    if (historical_data.empty()) {
        std::cerr << "No data loaded. Exiting.\n";
        return 1;
    }
    
    double initial_price = historical_data.back().close;
    
    std::cout << "Running Monte Carlo simulation...\n";
    run_monte_carlo_simulation(historical_data, n_simulations, n_days, initial_price, output_file);
    
    return 0;
}
