#include "Layer.hpp"

Layer::Layer(std::pair<unsigned, LayerType> info, std::pair<unsigned, LayerType> previous, OptimizerOpts &options) : m_type{info.second}, m_options{options}, m_regularization{options.regularization}, m_optimizer{options.optimizer}, m_states{info.first}, m_transferFunction{setTransferFunction(info.second)}, m_transferFunctionDerivative{setTransferFunctionDerivative(info.second)}
{
    if (info.first)
    {
        double factor = 0.1;
        if (info.second == RELU || info.second == LEAKY_RELU)
        {
            factor = 1.0 * sqrt(2.0 / info.first); // He
        }
        else
        {
            // factor = 1.0 * sqrt(1.0 / info.first);                  // Xavier
            factor = 1.0 * sqrt(6.0 / (info.first + previous.first)); // Normalized Xavier
        }
        m_weights = factor * Eigen::MatrixXd::Random(info.first, previous.first);
        m_bias = 0.01 * Eigen::VectorXd::Random(info.first);
        // MOMENTUM
        m_deltaBias.setZero(info.first);
        m_deltaWeights.setZero(info.first, previous.first);
        // ADAM and AMSGRAD
        m_ADAM_Mweight.setZero(info.first, previous.first);
        m_ADAM_Mbias.setZero(info.first);
        m_ADAM_Vweight.setZero(info.first, previous.first);
        m_ADAM_Vbias.setZero(info.first);
        m_AMSGRAD_VweightMax.setZero(info.first, previous.first);
        m_AMSGRAD_VbiasMax.setZero(info.first);
    }
}

std::function<double(double)> Layer::setTransferFunction(LayerType type) const
{
    switch (type)
    {
    case RELU:
        return [](double x)
        { return (x >= 0.0) ? x : 0; };
        break;
    case LEAKY_RELU:
        return [](double x)
        { return (x >= 0.0) ? x : -0.1 * x; };
        break;
    case TANH:
        return [](double x)
        { return tanhf(x); };
        break;
    case SIGMOID:
        return [](double x)
        { return (1.0 / (1.0 + exp(-x))); };
        break;
    case SOFTMAX:
        return [](double x) // not used
        { return x; };
        break;
    case LINEAR:
        return [](double x)
        { return x; };
    case INPUT:
        return [](double x) // not used
        { return 1; };
    default:
        throw std::invalid_argument("The specified activation function does not exist");
        break;
    }
}

std::function<double(double)> Layer::setTransferFunctionDerivative(LayerType type) const
{
    switch (type)
    {
    case RELU:
        return [](double x)
        { return (x >= 0.0) ? 1 : 0; };
        break;
    case LEAKY_RELU:
        return [](double x)
        { return (x >= 0.0) ? 1 : -0.1; };
        break;
    case TANH:
        return [](double t)
        { return 1.0 - t * t; }; // t corresponds to tanhf(x) here
        break;
    case SIGMOID:
        return [](double s)
        { return s * (1.0 - s); }; // s corresponds to sigmoid(x) here
        break;
    case SOFTMAX:
        return [](double s)
        { return s * (1.0 - s); }; // s corresponds to softmax(x) here, only helper
        break;
    case LINEAR:
        return [](double x) // not used
        { return 1; };
    case INPUT:
        return [](double x)
        { return 1; };
    default:
        throw std::invalid_argument("The specified activation function does not exist");
        break;
    }
}

Eigen::MatrixXd Layer::getOutput() const
{
    return m_outputVal;
}

void Layer::setOutput(Eigen::MatrixXd output)
{
    m_outputVal = output;
}

// void Layer::setEta(double eta)
// {
//     m_alpha = eta;
// }

void Layer::feedForward(const Layer &prevLayer)
{
    Eigen::MatrixXd input = prevLayer.getOutput();
    if (m_type == SIGMOID || m_type == TANH) // derivative defined in terms of output
    {
        m_outputVal = ((m_weights * input).colwise() + m_bias).unaryExpr(m_transferFunction);
        m_cache = m_outputVal;
    }
    else if (m_type == SOFTMAX)
    {
        m_outputVal = ((m_weights * input).colwise() + m_bias);
        m_outputVal = m_outputVal.array().exp().rowwise() / m_outputVal.array().exp().colwise().sum(); // apply softmax
        m_cache = m_outputVal;
    }
    else
    {
        m_cache = (m_weights * input).colwise() + m_bias;
        m_outputVal = m_cache.unaryExpr(m_transferFunction);
    }
    // std::cout << m_cache << std::endl;
    // std::cout << m_outputVal << std::endl;
}

void Layer::updateInputWeights()
{
    if (m_optimizer == SGD)
    {
        Eigen::MatrixXd newDeltaWeight = m_options.alpha * m_gradientWeight;
        m_weights -= newDeltaWeight;

        Eigen::VectorXd newDeltaBias = m_options.alpha * m_gradientBias;
        m_bias -= newDeltaBias;
    }
    else if (m_optimizer == MOMENTUM)
    {
        Eigen::MatrixXd newDeltaWeight = m_options.alpha * (1.0 - m_options.MOMENTUM_moment) * m_gradientWeight;
        newDeltaWeight += m_options.alpha * m_options.MOMENTUM_moment * m_deltaWeights;
        m_deltaWeights = newDeltaWeight;
        m_weights -= newDeltaWeight;

        Eigen::VectorXd newDeltaBias = m_options.alpha * (1.0 - m_options.MOMENTUM_moment) * m_gradientBias;
        newDeltaBias += m_options.alpha * m_options.MOMENTUM_moment * m_deltaBias;
        m_deltaBias = newDeltaBias;
        m_bias -= newDeltaBias;
    }
    else // ADAM and AMSGRAD
    {
        m_ADAM_Mweight = m_options.ADAM_beta1 * m_ADAM_Mweight + (1.0 - m_options.ADAM_beta1) * m_gradientWeight;
        m_ADAM_Vweight = m_options.ADAM_beta2 * m_ADAM_Vweight.array() + (1.0 - m_options.ADAM_beta2) * m_gradientWeight.array().square();
        Eigen::MatrixXd m_ADAM_MweightHat = m_ADAM_Mweight / (1.0 - pow(m_options.ADAM_beta1, m_options.ADAM_t));
        Eigen::MatrixXd m_ADAM_VweightHat = m_ADAM_Vweight / (1.0 - pow(m_options.ADAM_beta2, m_options.ADAM_t));

        m_ADAM_Mbias = m_options.ADAM_beta1 * m_ADAM_Mbias + (1.0 - m_options.ADAM_beta1) * m_gradientBias;
        m_ADAM_Vbias = m_options.ADAM_beta2 * m_ADAM_Vbias.array() + (1.0 - m_options.ADAM_beta2) * m_gradientBias.array().square();
        Eigen::VectorXd m_ADAM_MbiasHat = m_ADAM_Mbias / (1.0 - pow(m_options.ADAM_beta1, m_options.ADAM_t));
        Eigen::VectorXd m_ADAM_VbiasHat = m_ADAM_Vbias / (1.0 - pow(m_options.ADAM_beta2, m_options.ADAM_t));

        if (m_optimizer == AMSGRAD)
        {
            m_AMSGRAD_VweightMax = m_AMSGRAD_VweightMax.cwiseMax(m_ADAM_VweightHat);
            m_AMSGRAD_VbiasMax = m_AMSGRAD_VbiasMax.cwiseMax(m_ADAM_VbiasHat);
            m_weights.array() -= m_options.alpha * m_ADAM_MweightHat.array() / (m_AMSGRAD_VweightMax.array().sqrt() + m_options.ADAM_eps);
            m_bias.array() -= m_options.alpha * m_ADAM_MbiasHat.array() / (m_AMSGRAD_VbiasMax.array().sqrt() + m_options.ADAM_eps);
        }
        else // ADAM
        {
            m_weights.array() -= m_options.alpha * m_ADAM_MweightHat.array() / (m_ADAM_VweightHat.array().sqrt() + m_options.ADAM_eps);
            m_bias.array() -= m_options.alpha * m_ADAM_MbiasHat.array() / (m_ADAM_VbiasHat.array().sqrt() + m_options.ADAM_eps);
        }
    }
}

Eigen::MatrixXd Layer::calcHiddenGradients(const Eigen::MatrixXd &dA, const Layer &previousLayer)
{
    Eigen::MatrixXd dZ(dA.rows(), dA.cols());
    // error gradient loss
    if (m_type == SOFTMAX)
    {
        Eigen::VectorXd s(dZ.rows());
        Eigen::MatrixXd bg(dZ.rows(), dZ.rows());
        for (int i = 0; i < dZ.cols(); i++) // non-vectorized version
        {
            s = m_cache.col(i);
            bg = -s * s.transpose();
            bg.diagonal() = s.unaryExpr(m_transferFunctionDerivative);
            dZ.col(i) = bg * dA.col(i);
        }
    }
    else
    {
        dZ = dA.array() * (m_cache.unaryExpr(m_transferFunctionDerivative)).array();
    }
    m_gradientWeight = 1.0 / dA.cols() * dZ * previousLayer.getOutput().transpose();
    m_gradientBias = 1.0 / dA.cols() * dZ.rowwise().sum();
    // regularization term
    switch (m_regularization)
    {
    case L1:
        // L1 regularization
        m_gradientWeight += m_options.lambda / dA.cols() * m_weights.unaryExpr([](double x)
                                                                               { return -1.0 * std::signbit(x); });
        // m_gradientBias += m_lambda / dA.cols() * m_bias.unaryExpr([](double x) { return -1.0 * std::signbit(x); });
        break;
    case L2:
        // L2 regularization
        m_gradientWeight += m_options.lambda / dA.cols() * m_weights;
        // m_gradientBias += m_lambda / dA.cols() * m_bias;
        break;
    default:
        // NONE
        break;
    }
    return m_weights.transpose() * dZ;
}

unsigned Layer::getNumNeurons() const
{
    return m_states;
};

LayerType Layer::getLayerType() const
{
    return m_type;
};

void Layer::setParameters(Eigen::MatrixXd weights, Eigen::VectorXd bias)
{
    m_weights = weights;
    m_bias = bias;
};

std::pair<Eigen::MatrixXd, Eigen::VectorXd> Layer::getParameters(void) const
{
    return {m_weights, m_bias};
};