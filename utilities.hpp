#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#define REGRESSION 0
#define CLASSIFICATION 1
#define MULTILABEL 2
#define MNIST 3

#include <iostream>
#include <sstream>
#include <fstream>
#include <iostream>
#include "Net.hpp"

Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");

void showVectorVals(std::string label, Eigen::MatrixXd &v)
{
    std::cout << label;
    std::cout << v;
    std::cout << std::endl;
}

void createDataSet(std::string fileName, unsigned n)
{
    std::ofstream myFile;
    myFile.open(fileName);
    // myFile << "topology: 2 100 50 2\n";
    Eigen::VectorXd X(2);
    Eigen::VectorXd Y(2);
    for (size_t i = 0; i < n; i++)
    {
        X.setRandom(2);
        Y(0) = cos(3 * X(0) * X(1)) * sin(4 * X(0) * X(0)); // * 0.5 + 0.5
        Y(1) = sin(3 * X(0) * X(1));                        // * 0.5 + 0.5
        myFile << "in: ";
        myFile << X.transpose().format(HeavyFmt) << "\n";
        myFile << "out: ";
        myFile << Y.transpose().format(HeavyFmt) << "\n";
        ;
    }

    myFile.close();
}

void createDataSetClassification(std::string fileName, unsigned n)
{
    std::ofstream myFile;
    myFile.open(fileName);
    // myFile << "topology: 2 100 50 2\n";
    Eigen::VectorXd X(2);
    Eigen::VectorXd Y(1);
    for (size_t i = 0; i < n; i++)
    {
        X.setRandom(2);
        Y(0) = (abs(X(0)) + abs(X(1)) > 0.5) ? 1.0 : 0.0;
        myFile << "in: ";
        myFile << X.transpose() << "\n";
        myFile << "out: ";
        myFile << Y(0) << "\n";
        ;
    }
    myFile.close();
}

void createDataSetClassificationMC(std::string fileName, unsigned n, int dim)
{
    std::ofstream myFile;
    myFile.open(fileName);
    // myFile << "topology: 2 100 50 2\n";
    Eigen::VectorXd X(dim);
    Eigen::VectorXd Y(int(pow(2, dim)));
    for (size_t i = 0; i < n; i++)
    {
        X.setRandom(dim);
        X *= 0.5;
        X.array() += 0.5;
        X = X.unaryExpr([](double x)
                        { return round(x); });
        myFile << "in: ";
        myFile << X.transpose() << "\n";
        Y.setZero(int(pow(2, dim)));
        int idx = 0;
        for (size_t i = 0; i < dim; i++)
        {
            idx += X(i) * pow(2, i);
        }
        Y(int(idx)) = 1;
        myFile << "out: ";
        myFile << Y.transpose() << "\n";
        ;
    }
    myFile.close();
}

Eigen::MatrixXi getLabels(const Eigen::MatrixXd &input)
{
    Eigen::VectorXi indicies(input.cols());
    for (size_t i = 0; i < input.cols(); i++)
    {
        int idx;
        input.col(i).maxCoeff(&idx);
        indicies(i) = idx;
    }
    return indicies.transpose();
}

void printPredictions(Net &myNet, TrainingData &testData)
{
    const std::vector<std::pair<unsigned, LayerType>> topology = myNet.getTopology();
    Eigen::MatrixXd inputVals(topology.front().first, testData.getBatchSize());
    Eigen::MatrixXd targetVals(topology.back().first, testData.getBatchSize());
    Eigen::MatrixXd resultVals(topology.back().first, testData.getBatchSize());
    testData.ReloadData(inputVals, targetVals);
    int wrongLabels = 0;
    int processedPoints = 0;
    while (testData.next(inputVals, targetVals))
    {
        myNet.feedForward(inputVals);
        myNet.getResults(resultVals);
        if (myNet.getLossType() == LOGISTIC || myNet.getLossType() == LOGISTIC_CLIPPED)
        {
            if (topology.back().second == SOFTMAX)
            { // get labels
                Eigen::MatrixXi targetLabels = getLabels(targetVals);
                Eigen::MatrixXi predictedLabels = getLabels(resultVals);
                std::cout << "Target Labels:\n"
                          << targetLabels << std::endl;
                std::cout << "Predicted Labels:\n"
                          << predictedLabels << std::endl;
                wrongLabels += (targetLabels.array() != predictedLabels.array()).cast<int>().sum();
            }
            else
            { // get binary predictions
                resultVals = resultVals.unaryExpr([](double x)
                                                  { return round(x); });
                Eigen::MatrixXd err = targetVals - resultVals;
                showVectorVals("Errors:\n", err);
                wrongLabels += (targetVals.array() != resultVals.array()).cast<int>().sum();
            }
        }
        else
        { // print errors
            Eigen::MatrixXd err = targetVals - resultVals;
            showVectorVals("Errors:\n", err);
        }
        processedPoints += targetVals.cols();
        if (myNet.getLossType() == LOGISTIC || myNet.getLossType() == LOGISTIC_CLIPPED)
        {
            std::cout << "Wrong Labels: " << wrongLabels << " / " << processedPoints << std::endl;
        }
        std::cout << std::endl;
    }
}

void printDigitMNSIT(Eigen::MatrixXd &input, int n, int m)
{
    std::cout << input.reshaped(n, m) << std::endl;
}
#endif