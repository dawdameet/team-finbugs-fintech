/*
 * monte_carlo.h
 * Header file for Monte Carlo simulation module
 */

#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run Monte Carlo simulation for stock price prediction
 * 
 * @param data_file Path to historical price CSV file
 * @param n_simulations Number of simulation paths to generate
 * @param n_days Number of days to forecast
 * @param output_file Path to output JSON file
 * @return 0 on success, non-zero on error
 */
int run_monte_carlo(const char* data_file, int n_simulations, int n_days, const char* output_file);

#ifdef __cplusplus
}
#endif

#endif /* MONTE_CARLO_H */
