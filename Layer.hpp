#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include "eigen3/Eigen/Dense"

enum LossType
{
    MSE,
    MAE,
    HUBER,
    SSE,
    LOGISTIC,
    LOGISTIC_CLIPPED
};

enum LayerType
{
    INPUT,
    LINEAR,
    RELU,
    LEAKY_RELU,
    TANH,
    SIGMOID,
    SOFTMAX
};

enum RegularizationType
{
    L1,
    L2,
    NONE
};

enum Optimizer
{
    SGD,
    MOMENTUM,
    ADAM,
    AMSGRAD
};

struct OptimizerOpts
{
    // Parallel Settings
    bool parallel = false;
    int parallelBatches = 12;
    // Losses
    LossType loss = MSE;
    double Huber_delta = 0.1;
    double SSE_factor = 1.0;
    double LOGISTIC_CLIPPED_delta = 0.000001;
    // Regularization
    RegularizationType regularization = NONE;
    double lambda = 0.005;
    // Optimizer
    Optimizer optimizer = MOMENTUM;
    double alphaInit = 0.4;
    double alpha = alphaInit;
    double LRD = 1.0; // exponential learning rate decay
    // MOMENTUM
    double MOMENTUM_moment = 0.1;
    // ADAM and AMSGRAD
    double ADAM_beta1 = 0.9;
    double ADAM_beta2 = 0.999;
    double ADAM_eps = pow(10.0, -8);
    int ADAM_t = 0;
    // LOGGING
    bool LOG_history = true;
    bool LOG_bestModel = false;
    bool LOG_checkpoints = false;
    std::string LOG_modelFolder = "tmp/models/";
    std::string LOG_historyFolder = "tmp/history/";
};

class Layer
{
public:
    Layer(std::pair<unsigned, LayerType> info, std::pair<unsigned, LayerType> previous, OptimizerOpts &options);
    void setOutput(Eigen::MatrixXd output);
    // void setEta(double eta);
    Eigen::MatrixXd getOutput() const;
    void feedForward(const Layer &previousLayer);
    Eigen::MatrixXd calcHiddenGradients(const Eigen::MatrixXd &dA, const Layer &previousLayer);
    void updateInputWeights(void);
    std::function<double(double)> setTransferFunction(LayerType type) const;
    std::function<double(double)> setTransferFunctionDerivative(LayerType type) const;
    unsigned getNumNeurons() const;
    LayerType getLayerType() const;
    void setParameters(Eigen::MatrixXd weights, Eigen::VectorXd bias);
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getParameters(void) const;

private:
    const LayerType m_type;
    const RegularizationType m_regularization;
    const Optimizer m_optimizer;
    const unsigned m_states;
    OptimizerOpts &m_options;
    Eigen::MatrixXd m_weights;         // in x out
    Eigen::VectorXd m_bias;            // out
    Eigen::MatrixXd m_deltaWeights;    // in x out
    Eigen::VectorXd m_deltaBias;       // out
    Eigen::MatrixXd m_outputVal;       // out
    Eigen::MatrixXd m_cache;           // out
    Eigen::MatrixXd m_gradientWeight;  // in x out
    Eigen::VectorXd m_gradientBias;    // out
    Eigen::MatrixXd m_ADAM_Mweight;    // in x out
    Eigen::VectorXd m_ADAM_Mbias;      // out
    Eigen::MatrixXd m_ADAM_Vweight;    // in x out
    Eigen::VectorXd m_ADAM_Vbias;      // out
    Eigen::MatrixXd m_AMSGRAD_VweightMax; // out
    Eigen::VectorXd m_AMSGRAD_VbiasMax; // out
    const std::function<double(double)> m_transferFunction;
    const std::function<double(double)> m_transferFunctionDerivative;
    // static double transferFunction(double &x);
    // static double transferFunctionDerivative(double &x);
};
#endif