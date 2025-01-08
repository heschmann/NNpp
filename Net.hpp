#ifndef NET_HPP
#define NET_HPP

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <ctime>
#include <stdexcept>
#include <algorithm>
// #include <execution>

#include "Layer.hpp"
#include "TrainingData.hpp"

class Net
{
public:
    Net(const std::vector<std::pair<unsigned, LayerType>> &topology, OptimizerOpts &options);
    Net(const std::string fileName, OptimizerOpts &options);
    void feedForward(const Eigen::MatrixXd &inputVals);
    void getResults(Eigen::MatrixXd &resultVals) const;
    void backProp(const Eigen::MatrixXd &targetVals);
    // double getRecentAverageError(void) const;
    // double getError(void) const;
    Eigen::MatrixXd calcOutputGradients(const Eigen::MatrixXd &targetVal, const Eigen::MatrixXd &outputVal) const;
    // void updateEtas(double eta);
    double getPostError(const Eigen::MatrixXd &inputVal, const Eigen::MatrixXd &targetVals);
    void trainStep(TrainingData &trainData);
    void train(TrainingData &trainData, TrainingData &testData, unsigned totalIters);
    double eval(TrainingData &testData);
    std::function<double(double)> setLossFunction(LossType type) const;
    std::function<double(double)> setLossFunctionDerivative(LossType type) const;
    LossType getLossType(void) { return m_loss; }
    void save(std::string fileName) const;
    std::vector<std::pair<unsigned, LayerType>> getTopology(void) const;
    // Eigen::MatrixXd in;
    // Eigen::MatrixXd out;
private:
    OptimizerOpts &m_options;
    const Optimizer m_optimizer;
    const bool m_parallel;
    const LossType m_loss;
    std::vector<std::pair<unsigned, LayerType>> m_topology;
    std::vector<Layer> m_layers;
    // double m_error;
    // double m_recentAverageError = 0.0;
    // const double m_recentAverageSmoothingFactor = 10.0;
    const std::function<double(double)> m_lossFunction;
    const std::function<double(double)> m_lossFunctionDerivative;
    std::vector<std::pair<unsigned, LayerType>> loadTopology(std::ifstream &myFile);
    bool readParameter(std::string label, Eigen::MatrixXd &parameter, std::ifstream &myFile);
};
#endif