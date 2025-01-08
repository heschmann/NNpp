#ifndef TRAINING_DATA_HPP
#define TRAINING_DATA_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include "eigen3/Eigen/Dense"
#include <iostream>

class TrainingData
{
public:
    TrainingData(const std::string filename, const unsigned batchSize);
    void ReloadData(Eigen::MatrixXd &inputVals, Eigen::MatrixXd &targetOutputVals);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    // void getTopology(std::vector<unsigned> &topology);
    bool getNextInputs(Eigen::VectorXd &inputVals);
    bool getTargetOutputs(Eigen::VectorXd &targetOutputVals);
    bool next(Eigen::VectorXd &inputVals, Eigen::VectorXd &targetOutputVals);
    bool next(Eigen::MatrixXd &inputVals, Eigen::MatrixXd &targetOutputVals);
    // void setBatchSize(unsigned size);
    unsigned getBatchSize(void) const;
    unsigned getNumBatches(void) const;
    void usePreload(bool preLoad, unsigned inDim, unsigned outDim);
private:
    unsigned calcNumBatches();
    std::ifstream m_trainingDataFile;
    const std::string m_filename;
    const unsigned m_batchSize;
    unsigned m_numPoints;
    const unsigned m_numBatches;
    bool m_preLoad = false;
    unsigned m_preLoadPos = 0;
    Eigen::MatrixXd m_preLoadInputs;
    Eigen::MatrixXd m_preLoadOutputs;
};
#endif