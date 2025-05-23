#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <omp.h>

class PortfolioOptimizer {
private:
    std::vector<double> returns;
    std::vector<std::vector<double>> covariance_matrix;
    int num_assets;
    int population_size;
    int num_generations;

public:
    PortfolioOptimizer(const std::vector<double>& returns, 
                       const std::vector<std::vector<double>>& covariance_matrix,
                       int population_size = 1000, 
                       int num_generations = 100)
        : returns(returns), covariance_matrix(covariance_matrix),
          num_assets(returns.size()), population_size(population_size),
          num_generations(num_generations) {}

    std::vector<double> optimize() {
        std::vector<std::vector<double>> population(population_size, std::vector<double>(num_assets));
        std::vector<double> fitness(population_size);

        // Initialize population
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        #pragma omp parallel for
        for (int i = 0; i < population_size; ++i) {
            for (int j = 0; j < num_assets; ++j) {
                population[i][j] = dis(gen);
            }
            normalize(population[i]);
        }

        // Main optimization loop
        for (int generation = 0; generation < num_generations; ++generation) {
            // Evaluate fitness
            #pragma omp parallel for
            for (int i = 0; i < population_size; ++i) {
                fitness[i] = calculate_sharpe_ratio(population[i]);
            }

            // Selection and crossover
            std::vector<std::vector<double>> new_population(population_size, std::vector<double>(num_assets));
            #pragma omp parallel for
            for (int i = 0; i < population_size; i += 2) {
                int parent1 = tournament_selection(fitness);
                int parent2 = tournament_selection(fitness);
                crossover(population[parent1], population[parent2], new_population[i], new_population[i+1]);
            }

            // Mutation
            #pragma omp parallel for
            for (int i = 0; i < population_size; ++i) {
                mutate(new_population[i]);
                normalize(new_population[i]);
            }

            population = new_population;
        }

        // Find best solution
        int best_index = std::max_element(fitness.begin(), fitness.end()) - fitness.begin();
        return population[best_index];
    }

private:
    void normalize(std::vector<double>& weights) {
        double sum = 0.0;
        for (double w : weights) sum += w;
        for (double& w : weights) w /= sum;
    }

    double calculate_sharpe_ratio(const std::vector<double>& weights) {
        double portfolio_return = 0.0;
        double portfolio_risk = 0.0;

        for (int i = 0; i < num_assets; ++i) {
            portfolio_return += weights[i] * returns[i];
            for (int j = 0; j < num_assets; ++j) {
                portfolio_risk += weights[i] * weights[j] * covariance_matrix[i][j];
            }
        }

        portfolio_risk = std::sqrt(portfolio_risk);
        return portfolio_return / portfolio_risk;
    }

    int tournament_selection(const std::vector<double>& fitness) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, population_size - 1);

        int idx1 = dis(gen);
        int idx2 = dis(gen);
        return (fitness[idx1] > fitness[idx2]) ? idx1 : idx2;
    }

    void crossover(const std::vector<double>& parent1, const std::vector<double>& parent2,
                   std::vector<double>& child1, std::vector<double>& child2) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, num_assets - 1);
        int crossover_point = dis(gen);

        for (int i = 0; i < num_assets; ++i) {
            if (i < crossover_point) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            } else {
                child1[i] = parent2[i];
                child2[i] = parent1[i];
            }
        }
    }

    void mutate(std::vector<double>& weights) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::uniform_int_distribution<> idx_dis(0, num_assets - 1);

        if (dis(gen) < 0.1) {  // 10% mutation rate
            int idx = idx_dis(gen);
            weights[idx] = dis(gen);
        }
    }
};

int main() {
    std::vector<double> returns = {0.05, 0.06, 0.07, 0.08, 0.09};
    std::vector<std::vector<double>> covariance_matrix = {
        {0.04, 0.02, 0.015, 0.01, 0.005},
        {0.02, 0.05, 0.0175, 0.015, 0.0075},
        {0.015, 0.0175, 0.06, 0.02, 0.01},
        {0.01, 0.015, 0.02, 0.07, 0.0125},
        {0.005, 0.0075, 0.01, 0.0125, 0.08}
    };

    PortfolioOptimizer optimizer(returns, covariance_matrix);
    std::vector<double> optimal_weights = optimizer.optimize();

    std::cout << "Optimal portfolio weights:" << std::endl;
    for (int i = 0; i < optimal_weights.size(); ++i) {
        std::cout << "Asset " << i + 1 << ": " << optimal_weights[i] << std::endl;
    }

    return 0;
}
