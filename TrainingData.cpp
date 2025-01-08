#include "TrainingData.hpp"

/* void TrainingData::getTopology(std::vector<unsigned> &topology)
{
    std::string line;
    std::string label;

    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    while (!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
} */

TrainingData::TrainingData(const std::string filename, const unsigned batchSize) : m_filename{filename}, m_batchSize{batchSize}, m_numBatches{calcNumBatches()}
{
    // m_filename = filename;
    // m_trainingDataFile.open(filename.c_str());
}

void TrainingData::ReloadData(Eigen::MatrixXd &inputVals, Eigen::MatrixXd &targetOutputVals)
{
    // restore old batch size
    inputVals.resize(inputVals.rows(), m_batchSize);
    targetOutputVals.resize(targetOutputVals.rows(), m_batchSize);
    // reset string stream
    m_trainingDataFile.clear();
    m_trainingDataFile.seekg(0);
    // reset pre load
    m_preLoadPos = 0;
    // m_trainingDataFile.open(m_filename.c_str());
}

bool TrainingData::getNextInputs(Eigen::VectorXd &inputVals)
{
    // inputVals.clear();

    std::string line;
    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double oneValue;
        int i = 0;
        while (ss >> oneValue)
        {
            inputVals(i) = oneValue;
            i++;
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool TrainingData::getTargetOutputs(Eigen::VectorXd &targetOutputVals)
{
    // targetOutputVals.clear();

    std::string line;
    std::getline(m_trainingDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double oneValue;
        int i = 0;
        while (ss >> oneValue)
        {
            targetOutputVals(i) = oneValue;
            i++;
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool TrainingData::next(Eigen::VectorXd &inputVals, Eigen::VectorXd &targetOutputVals)
{
    /*if (m_preLoad)
    {
        if (m_preLoadPos < m_preLoadInputs.cols())
        {
            inputVals = m_preLoadInputs.col(m_preLoadPos);
            targetOutputVals = m_preLoadOutputs.col(m_preLoadPos);
            m_preLoadPos++;
            return true;
        }
        return false;
    }
    else
    {*/
    return (getNextInputs(inputVals) && getTargetOutputs(targetOutputVals));
    //}
};

/*void TrainingData::setBatchSize(unsigned size)
{
    m_batchSize = size;
    calcNumBatches();
}*/

bool TrainingData::next(Eigen::MatrixXd &inputVals, Eigen::MatrixXd &targetOutputVals)
{
    if (m_preLoad)
    {
        if (m_preLoadPos >= m_preLoadInputs.cols())
            return false;
        int i = (m_preLoadPos + m_batchSize < m_preLoadInputs.cols()) ? m_batchSize : m_preLoadInputs.cols() - m_preLoadPos;
        inputVals = m_preLoadInputs.block(0, m_preLoadPos, inputVals.rows(), i).eval();
        targetOutputVals = m_preLoadOutputs.block(0, m_preLoadPos, targetOutputVals.rows(), i).eval();
        m_preLoadPos += i;
        return true;
    }
    else
    {
        Eigen::VectorXd input(inputVals.rows());
        Eigen::VectorXd target(targetOutputVals.rows());
        unsigned i = 0;
        while (i < m_batchSize && next(input, target))
        {
            // next(input, target);
            inputVals.col(i) = input;
            targetOutputVals.col(i) = target;
            i++;
        }
        // handle incomplete non-empty batch
        if (i != m_batchSize && i != 0)
        {
            // eval is needed here to avoid needless copy and invalidation of the reference
            inputVals = inputVals.block(0, 0, inputVals.rows(), i).eval();
            targetOutputVals = targetOutputVals.block(0, 0, targetOutputVals.rows(), i).eval();
        }
        return (i != 0);
    }
}

unsigned TrainingData::getBatchSize() const
{
    return m_batchSize;
};

unsigned TrainingData::getNumBatches() const
{
    return m_numBatches;
};

unsigned TrainingData::calcNumBatches()
{
    m_trainingDataFile.open(m_filename.c_str());
    m_trainingDataFile.clear();
    m_trainingDataFile.seekg(0);
    m_numPoints = std::count(std::istreambuf_iterator<char>(m_trainingDataFile),
                                    std::istreambuf_iterator<char>(), '\n') /
                         2;
    unsigned numBatches = m_numPoints / m_batchSize;
    if (m_numPoints % m_batchSize > 0)
    {
        numBatches++; // incomplete batch
    }
    return numBatches;
}

void TrainingData::usePreload(bool preLoad, unsigned inDim, unsigned outDim)
{
    m_preLoad = preLoad;
    m_trainingDataFile.clear();
    m_trainingDataFile.seekg(0);
    if (m_preLoad)
    {
        Eigen::VectorXd input(inDim);
        Eigen::VectorXd target(outDim);
        m_preLoadInputs.resize(input.rows(), m_numPoints);
        m_preLoadOutputs.resize(target.rows(), m_numPoints);
        int i = 0;
        while (next(input, target))//(i < m_preLoadInputs.cols())
        {
            // getNextInputs(input);
            // getTargetOutputs(target);
            // next(input, target);
            m_preLoadInputs.col(i) = input;
            m_preLoadOutputs.col(i) = target;
            i++;
        }
        std::cout << i << " data points were loaded into memory." << std::endl;
    }
    else
    {
        m_preLoadInputs.resize(0, 0);
        m_preLoadOutputs.resize(0, 0);
    }
};