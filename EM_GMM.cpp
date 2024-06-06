#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

float Gaussian(float x, float mu, float sigma){
    return std::exp(-0.5 * std::pow((x - mu) / sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
}

void Posterior(const std::vector<double>& X, const std::vector<double>& mus, const std::vector<double>& sigmas, const std::vector<double>& probs, std::vector<std::vector<double>>& posteriors){
    size_t N = X.size();
    size_t K = mus.size();
    for (size_t i = 0; i < N; ++i) {
        double likeSum = 0.0;
        std::vector<double> temp(K);
        for (size_t j = 0; j < K; ++j) {
            temp[j] = Gaussian(X[i], mus[j], sigmas[j])*probs[j];
            likeSum += temp[j];
        }
        for (size_t j = 0; j < K; ++j) {
            posteriors[j][i] = temp[j]/likeSum;
        }
    }
}

void OptimalPi(const std::vector<std::vector<double>>& posteriors, std::vector<double>& probs){
    size_t N = posteriors[0].size();
    double postSum = 0.0, postZero = 0.0, pi1 = 0.0;
    for (size_t i = 0; i < N; ++i) {
        postZero += posteriors[0][i];
        postSum += (posteriors[0][i]+posteriors[1][i]);
    }
    pi1 = postZero/postSum;
    probs[0] = pi1;
    probs[1] = 1-pi1;
}

void OptimalMu(const std::vector<double>& X, const std::vector<std::vector<double>>& posteriors, std::vector<double>& mus){
    size_t N = posteriors[0].size();
    size_t K = mus.size();
    
    for (size_t j = 0; j < K; ++j) {
        double postSum = 0.0, postSumWeighted = 0.0;
        for (size_t i = 0; i < N; ++i) {
            postSum += posteriors[j][i];
            postSumWeighted += (X[i]*posteriors[j][i]);
        }
        mus[j] = postSumWeighted/postSum;
    }
}

void OptimalSigmas(const std::vector<double>& X, const std::vector<std::vector<double>>& posteriors, const std::vector<double>& mus, std::vector<double>& sigmas){
    size_t N = posteriors[0].size();
    size_t K = mus.size();
    
    for (size_t j = 0; j < K; ++j) {
        double postSum = 0.0, postSumWeighted = 0.0;
        for (size_t i = 0; i < N; ++i) {
            postSum += posteriors[j][i];
            postSumWeighted += (std::pow(X[i]-mus[j],2.0)*posteriors[j][i]);
        }
        sigmas[j] = std::sqrt(postSumWeighted/postSum);
    }
}

double IncompleLogLikelihood(const std::vector<double>& X, const std::vector<double>& mus, const std::vector<double>& sigmas, const std::vector<double>& probs){
    size_t N = X.size();
    size_t K = mus.size();
    double logLikeSum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double likeSum = 0.0;
        for (size_t j = 0; j < K; ++j) {
            likeSum += Gaussian(X[i], mus[j], sigmas[j])*probs[j];
        }
        logLikeSum += std::log(likeSum);
    }
    return logLikeSum;
}

void ExpectationMaximisation(const std::vector<double>& X, std::vector<double>& mus, std::vector<double>& sigmas, std::vector<double>& probs){
    double currentLike = 0.0, prevLike = 0.0, epsilon = 0.000001, delta = 1.0;
    int iter = 0;
    std::vector<std::vector<double>> posteriors(mus.size(), std::vector<double>(X.size()));
    while(delta>epsilon && iter < 10000){
        Posterior(X, mus, sigmas, probs, posteriors);
        OptimalPi(posteriors, probs);
        OptimalMu(X, posteriors, mus);
        OptimalSigmas(X, posteriors, mus, sigmas);
        prevLike = currentLike;
        currentLike = IncompleLogLikelihood(X, mus, sigmas, probs);
        if (iter > 0){
            delta = currentLike - prevLike;
        }
        iter++;
    }
    std::cout << "EM Algorithm completed in " << iter << " iterations!\n";
}

int main()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    
    // True unkown params
    double mu1 = 0.0, mu2 = 4.0, sigma1 = 1.0, sigma2 = 2.0, pi1 = 0.2;

    // Size of total dataset
    int N = 1000;
    int split_index = static_cast<int>(N * 0.2);

    // First Gaussian
    std::vector<double> Z1(split_index);
    std::normal_distribution d1{mu1, sigma1};
    for (int i = 0; i < split_index; ++i) {
        Z1[i] = d1(gen);
    }

    // Second Gaussian
    std::vector<double> Z2(N-split_index);
    std::normal_distribution d2{mu2, sigma2};
    for (int i = 0; i < N-split_index; ++i) {
        Z2[i] = d2(gen);
    }

    // GMM combininig Z1 and Z2
    std::vector<double> X;
    X.reserve( Z1.size() + Z2.size() );
    X.insert( X.end(), Z1.begin(), Z1.end() );
    X.insert( X.end(), Z2.begin(), Z2.end() );

    // EM ALGORITHM:

    // Initial guess for params of GMM
    std::vector<double> mus = {0.0, 0.5};
    std::vector<double> sigmas = {1.0, 1.5};
    std::vector<double> probs = {0.5, 0.5};

    // Get prediction
    ExpectationMaximisation(X, mus, sigmas, probs);

    // Output predicted params
    std::cout << "mu1 = " << mus[0] << ", mu2 = " << mus[1] << ".\n";
    std::cout << "sigmas1 = " << sigmas[0] << ", sigmas2 = " << sigmas[1] << ".\n";
    std::cout << "probs1 = " << probs[0] << ", probs2 = " << probs[1] << ".\n";

    // Write data to JSON file
    json j;
    j["X"] = X;
    j["mus"] = mus;
    j["sigmas"] = sigmas;
    j["probs"] = probs;

    std::ofstream file("data.json");
    file << j.dump(4);
    file.close();

    std::cout << "Data saved to data.json" << std::endl;
    return 0;
}