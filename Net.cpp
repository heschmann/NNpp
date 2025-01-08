#include "Net.hpp"

Net::Net(const std::vector<std::pair<unsigned, LayerType>> &topology, OptimizerOpts &options) : m_options{options}, m_optimizer{options.optimizer}, m_parallel{m_options.parallel}, m_loss{options.loss}, m_lossFunction{setLossFunction(options.loss)}, m_lossFunctionDerivative{setLossFunctionDerivative(options.loss)}
{
    m_topology = topology;
    unsigned numLayers = topology.size();
    m_layers.push_back(Layer(topology[0], {0, INPUT}, options)); // input layer
    for (unsigned layerNum = 1; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer(topology[layerNum], topology[layerNum - 1], options));
    }
}

Net::Net(std::string fileName, OptimizerOpts &options) : m_options{options}, m_optimizer{options.optimizer}, m_parallel{m_options.parallel}, m_loss{options.loss}, m_lossFunction{setLossFunction(options.loss)}, m_lossFunctionDerivative{setLossFunctionDerivative(options.loss)}
{
    std::ifstream myFile;
    myFile.open(fileName.c_str());
    std::vector<std::pair<unsigned, LayerType>> topology = loadTopology(myFile);
    m_topology = topology;
    unsigned numLayers = topology.size();
    m_layers.push_back(Layer(topology[0], {0, INPUT}, options)); // input layer
    for (unsigned layerNum = 1; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer(topology[layerNum], topology[layerNum - 1], options));
        Eigen::MatrixXd W(1, topology[layerNum].first * topology[layerNum - 1].first);
        Eigen::MatrixXd b(1, topology[layerNum].first);
        bool flagW = readParameter("W:", W, myFile);
        if (!flagW)
        {
            std::cout << "Some parameters could not be loaded correctly!" << std::endl;
            abort();
        }
        bool flagB = readParameter("b:", b, myFile);
        if (!flagB)
        {
            std::cout << "Some parameters could not be loaded correctly!" << std::endl;
            abort();
        }
        m_layers[layerNum].setParameters(W.reshaped(topology[layerNum].first, topology[layerNum - 1].first), b.transpose());
    }
    myFile.close();
    std::cout << "Model was loaded from file." << std::endl;
}

void Net::feedForward(const Eigen::MatrixXd &inputVal)
{

    // Assign (latch) the input values into the input neurons
    m_layers[0].setOutput(inputVal);

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        m_layers[layerNum].feedForward(prevLayer);
    }
}

void Net::getResults(Eigen::MatrixXd &resultVals) const
{
    resultVals = m_layers.back().getOutput();
};

void Net::backProp(const Eigen::MatrixXd &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    Eigen::MatrixXd outputVals = outputLayer.getOutput();

    // m_error = getPostError(m_layers.front().getOutput() ,targetVals); // average L2 norm

    // Implement a recent average measurement
    /* m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);*/

    // Calculate output layer gradients (move calcOutputGradients to Net as it depends on the loss not the layer)
    Eigen::MatrixXd dA = calcOutputGradients(targetVals, outputVals);

    // Calculate hidden layer gradients
    // (pass previous Layer to calcHiddenGradients instead and pass dA as an argument, starting with calcOutputGradients)
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &previousLayer = m_layers[layerNum - 1];
        dA = hiddenLayer.calcHiddenGradients(dA, previousLayer);
    }

    // For all layers from outputs to first hidden layer, update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &layer = m_layers[layerNum];
        // Layer &prevLayer = m_layers[layerNum - 1];
        layer.updateInputWeights(); // prevLayer
    }
}

/* double Net::getRecentAverageError() const
{
    return m_recentAverageError;
} */

/* double Net::getError() const
{
    return m_error;
}
 */
double Net::getPostError(const Eigen::MatrixXd &inputVal, const Eigen::MatrixXd &targetVals)
{
    feedForward(inputVal);
    Layer &outputLayer = m_layers.back();
    Eigen::MatrixXd outputVals = outputLayer.getOutput();
    // use mean by default
    if (m_loss == LOGISTIC)
    {
        Eigen::MatrixXd left = outputVals.array().log();
        Eigen::MatrixXd right = (1.0 - outputVals.array()).log();
        Eigen::MatrixXd loss = (targetVals.array() > 0.5).select(left, right); // avoids some numerical issues
        return -1.0 * loss.array().sum() / targetVals.rows();
    }
    else if (m_loss == LOGISTIC_CLIPPED)
    {
        Eigen::MatrixXd left = outputVals.unaryExpr(m_lossFunction).array().log();
        Eigen::MatrixXd right = (1.0 - outputVals.array().unaryExpr(m_lossFunctionDerivative)).log();
        Eigen::MatrixXd loss = (targetVals.array() > 0.5).select(left, right); // avoids some numerical issues
        return -1.0 * loss.array().sum() / targetVals.rows();
    }
    else
    {
        return (targetVals - outputVals).unaryExpr(m_lossFunction).array().sum() / targetVals.rows();
    }
}

Eigen::MatrixXd Net::calcOutputGradients(const Eigen::MatrixXd &targetVal, const Eigen::MatrixXd &outputVal) const
{
    if (m_loss == LOGISTIC)
    {
        Eigen::MatrixXd left = -1.0 / (outputVal).array();
        Eigen::MatrixXd right = 1.0 / (1.0 - outputVal.array());
        Eigen::MatrixXd dA = (targetVal.array() > 0.5).select(left, right); // avoids some numerical issues
        return dA;
    }
    else if (m_loss == LOGISTIC_CLIPPED)
    {
        Eigen::MatrixXd left = -1.0 / outputVal.unaryExpr(m_lossFunction).array();
        Eigen::MatrixXd right = 1.0 / (1.0 - outputVal.unaryExpr(m_lossFunctionDerivative).array());
        Eigen::MatrixXd dA = (targetVal.array() > 0.5).select(left, right); // avoids some numerical issues
        return dA;
    }

    else
    {
        return (outputVal - targetVal).unaryExpr(m_lossFunctionDerivative);
    }
}

/* void Net::updateEtas(double eta)
{
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &layer = m_layers[layerNum];
        layer.setEta(eta);
    }
} */

void Net::trainStep(TrainingData &trainData)
{
    // updateEtas(eta);
    Eigen::MatrixXd inputVals(m_layers.front().getNumNeurons(), trainData.getBatchSize());
    Eigen::MatrixXd targetVals(m_layers.back().getNumNeurons(), trainData.getBatchSize());
    trainData.ReloadData(inputVals, targetVals);
    unsigned numBatches = trainData.getNumBatches();
    if (m_parallel)
    {
        const int numPar = m_options.parallelBatches;
        std::vector<Eigen::MatrixXd> inputs;
        std::vector<Eigen::MatrixXd> targets;
        int numParAct = 0;
        while (trainData.next(inputVals, targetVals))
        {
            inputs.push_back(inputVals);
            targets.push_back(targetVals);
            numParAct++;
            if (numParAct == numPar)
            {
                numParAct = 0;
                std::vector<Net> nets(numPar, *this);
                /* for (size_t iPar = 0; iPar < numPar; iPar++)
                {
                    nets[iPar].in = inputs[iPar];
                    nets[iPar].out = targets[iPar];
                } */
                for (size_t iPar = 0; iPar < numPar; iPar++)
                {
                    nets[iPar].feedForward(inputs[iPar]);
                    nets[iPar].backProp(targets[iPar]);
                }
                /*std::for_each(std::execution::par_unseq, nets.begin(), nets.end(), [](Net &net)
                              {
                    net.feedForward(net.in);
                    net.backProp(net.out); });*/
                /* #pragma omp parallel
                for (size_t iPar = 0; iPar < numPar; iPar++)
                {
                    nets[iPar].feedForward(nets[iPar].in);
                    nets[iPar].backProp(nets[iPar].out);
                } */
                for (size_t iLayer = 1; iLayer < m_layers.size(); iLayer++)
                {
                    auto parameters = nets[0].m_layers[iLayer].getParameters();
                    Eigen::MatrixXd weights = parameters.first;
                    Eigen::VectorXd bias = parameters.second;
                    for (size_t iNet = 1; iNet < numPar; iNet++)
                    {
                        auto parameters = nets[iNet].m_layers[iLayer].getParameters();
                        weights += parameters.first;
                        bias += parameters.second;
                    }
                    // TO DO FIX momentum term
                    m_layers[iLayer].setParameters(weights / numPar, bias / numPar);
                }
                inputs.clear();
                targets.clear();
                nets.clear();
            }
            m_options.ADAM_t++;
        }
    }
    else
    {
        unsigned processedBatches = 0;
        while (trainData.next(inputVals, targetVals))
        {
            feedForward(inputVals);
            backProp(targetVals);
            processedBatches++;
            if (std::fmod(processedBatches, (numBatches / 25.0)) <= 25.0 / numBatches)
            {
                std::cout << "=";
            }
            m_options.ADAM_t++;
        }
    }
    std::cout << std::endl;
};

void Net::train(TrainingData &trainData, TrainingData &testData, unsigned totalIters)
{
    std::ofstream history;
    time_t startTime, endTime;
    struct tm *timeInfo;
    char buffer[80];
    time(&startTime);
    timeInfo = localtime(&startTime);
    strftime(buffer, sizeof(buffer), "%d_%m_%Y_at_%H_%M", timeInfo);
    std::string str(buffer);
    std::string runName = "BS" + std::to_string(trainData.getBatchSize()) + "_" + str + ".txt";

    if (m_options.LOG_history)
    {
        history.open(m_options.LOG_historyFolder + runName);
    }

    double errorCurrentIter = 0.0;
    double errorBestIter = INFINITY;
    m_options.alpha = m_options.alphaInit;
    m_options.ADAM_t = 1;
    for (size_t iter = 1; iter <= totalIters; iter++)
    {
        trainStep(trainData);
        errorCurrentIter = eval(testData); // evaluate actual performance
        m_options.alpha *= m_options.LRD;  // eta becomes fraction of init value within total iters

        if (errorCurrentIter < errorBestIter)
        {
            errorBestIter = errorCurrentIter;
            if (m_options.LOG_bestModel)
                save(m_options.LOG_modelFolder + "model_best_" + runName);
        }
        if (m_options.LOG_history)
            history << errorCurrentIter << "\n";

        if (m_options.LOG_checkpoints)
            save(m_options.LOG_modelFolder + "model_CP" + std::to_string(iter) + "_" + runName);

        std::cout << "Actual error of iteration " << std::right << std::setw(4) << iter << ": " << std::left << std::setw(10) << errorCurrentIter << " (best: " << std::setw(9) << errorBestIter << ")" << std::endl;
    }
    if (m_options.LOG_history)
    {
        history.close();
    }

    time(&endTime);
    std::cout << std::endl
              << "Training finished after " << difftime(endTime, startTime) << "s" << std::endl;
};

double Net::eval(TrainingData &testData)
{
    Eigen::MatrixXd inputVals(m_layers.front().getNumNeurons(), testData.getBatchSize());
    Eigen::MatrixXd targetVals(m_layers.back().getNumNeurons(), testData.getBatchSize());
    testData.ReloadData(inputVals, targetVals);
    double sumErrors = 0.0;
    unsigned processedPoints = 0;
    while (testData.next(inputVals, targetVals))
    {
        // testData.next(inputVals, targetVals);
        sumErrors += getPostError(inputVals, targetVals);
        processedPoints += inputVals.cols();
    }
    return sumErrors / processedPoints;
}

std::function<double(double)> Net::setLossFunction(LossType type) const
{
    const double delta = m_options.Huber_delta;
    const double factor = m_options.SSE_factor;
    const double clipping = m_options.LOGISTIC_CLIPPED_delta;
    switch (type)
    {
    case MSE:
        return [](double x)
        { return x * x; }; // 1/n * e^2
        break;
    case MAE:
        return [](double x)
        { return (x > 0.0) ? x : -x; }; // 1/n * |e|
        break;
    case HUBER:
        return [delta](double x)
        { return (abs(x) < delta) ? 0.5 * x * x : delta * (abs(x) - 0.5 * delta); };
        break;
    case SSE:
        return [factor](double x)
        { return factor * x * x; };
        break;
    case LOGISTIC:
        // this derivative is not defined via an unary expression, see Net::calcOutputGradients()
        return [](double x)
        { return 0; };
        break;
    case LOGISTIC_CLIPPED:
        // clipping function
        return [clipping](double x)
        { return std::max(x, clipping); };
        break;
    default:
        throw std::invalid_argument("The specified argument for the loss function does not exist");
        break;
    }
}

std::function<double(double)> Net::setLossFunctionDerivative(LossType type) const
{
    const double delta = m_options.Huber_delta;
    const double factor = m_options.SSE_factor;
    const double clipping = m_options.LOGISTIC_CLIPPED_delta;
    switch (type)
    {
    case MSE:
        return [](double x)
        { return 2.0 * x; };
        break;
    case MAE:
        return [](double x)
        { return (x > 0.0) ? 1.0 : -1.0; };
        break;
    case HUBER:
        return [delta](double x)
        { return (abs(x) < delta) ? x : -delta * std::signbit(x); };
        break;
    case SSE:
        return [factor](double x)
        { return 2.0 * factor * x; };
        break;
    case LOGISTIC:
        // this derivative is not defined via an unary expression, see Net::calcOutputGradients()
        return [](double x)
        { return 0; };
        break;
    case LOGISTIC_CLIPPED:
        // clipping function
        return [clipping](double x)
        { return std::min(x, 1.0 - clipping); };
        break;
    default:
        throw std::invalid_argument("The specified argument for the loss function does not exist");
        break;
    }
}

void Net::save(std::string fileName) const
{
    std::ofstream myFile;
    Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");

    if (fileName.find_last_of('.') >= fileName.size())
    {
        time_t startTime, endTime;
        struct tm *timeInfo;
        char buffer[80];
        time(&startTime);
        timeInfo = localtime(&startTime);
        strftime(buffer, sizeof(buffer), "%d_%m_%Y_at_%H_%M", timeInfo);
        std::string str(buffer);
        fileName = fileName + "model_" + buffer + ".txt";
    }

    myFile.open(fileName);
    myFile << "topology:";
    for (int iLayer = 0; iLayer < m_layers.size(); iLayer++)
    {
        myFile << " " << m_layers[iLayer].getNumNeurons() << " " << m_layers[iLayer].getLayerType();
    }
    myFile << "\n";
    for (int iLayer = 1; iLayer < m_layers.size(); iLayer++)
    {
        Layer layer = m_layers[iLayer];
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> param = layer.getParameters();
        myFile << "W: " << param.first.reshaped(1, param.first.size()).format(HeavyFmt) << "\n";
        myFile << "b: " << param.second.transpose().format(HeavyFmt) << "\n";
    }
    myFile.close();
    std::cout << "Model saved under: " << fileName << std::endl;
};

std::vector<std::pair<unsigned, LayerType>> Net::loadTopology(std::ifstream &myFile)
{
    std::string line;
    std::string label;

    std::getline(myFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (label.compare("topology:") != 0)
    {
        abort();
    }
    std::vector<std::pair<unsigned, LayerType>> topology;
    while (!ss.eof())
    {
        unsigned dim;
        int type;
        ss >> dim;
        ss >> type;
        topology.push_back({dim, (LayerType)type});
    }

    return topology;
}

bool Net::readParameter(std::string label, Eigen::MatrixXd &parameter, std::ifstream &myFile)
{
    std::string line;
    std::getline(myFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (label.compare(label) != 0)
    {
        return false; // wrong label
    }
    double oneValue;
    int i = 0;
    while (ss >> oneValue)
    {
        if (i >= parameter.size()) // too many parameters
        {
            return false;
        }
        parameter(0, i) = oneValue;
        i++;
    }
    return (i == parameter.size()); // not enough parameters?
};

std::vector<std::pair<unsigned, LayerType>> Net::getTopology(void) const
{
    return m_topology;
};