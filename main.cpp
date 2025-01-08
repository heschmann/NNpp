#include <vector>
#include <iostream>

#include "Net.hpp"
#include "utilities.hpp"

#define DEMOCASE REGRESSION

int main()
{
    OptimizerOpts opts;
    const int totalIters = 50;

#if DEMOCASE == REGRESSION
    opts.loss = MSE;
    const unsigned batchSize = 8;
    opts.optimizer = ADAM;
    opts.LRD = pow(0.01, 1.0 / totalIters);
    opts.alphaInit = 0.001;
    createDataSet("tmp/train.txt", 2000);
    createDataSet("tmp/test.txt", 400);
    TrainingData trainData("tmp/train.txt", batchSize);
    TrainingData testData("tmp/test.txt", 400);
    // std::vector<std::pair<unsigned, unsigned>> topology{{2, INPUT}, {50, RELU}, {50, RELU}, {2, LINEAR}};
    std::vector<std::pair<unsigned, LayerType>> topology{{2, INPUT}, {300, RELU}, {300, RELU}, {100, RELU}, {2, LINEAR}};
#endif

#if DEMOCASE == CLASSIFICATION
    opts.loss = LOGISTIC;
    const unsigned batchSize = 8;
    opts.optimizer = AMSGRAD;
    opts.LRD = 1.0;
    opts.alphaInit = 0.05;
    createDataSetClassification("tmp/train_class.txt", 1000);
    createDataSetClassification("tmp/test_class.txt", 200);
    TrainingData trainData("tmp/train_class.txt", batchSize);
    TrainingData testData("tmp/test_class.txt", 200);
    std::vector<std::pair<unsigned, LayerType>> topology{{2, INPUT}, {40, RELU}, {1, SIGMOID}};
#endif

#if DEMOCASE == MULTILABEL
    opts.loss = LOGISTIC;
    const unsigned batchSize = 8;
    opts.alphaInit = 0.2;
    opts.optimizer = MOMENTUM;
    opts.alphaInit = 0.2;
    opts.LRD = pow(0.01, 1.0 / totalIters);
    int dim = 4;
    createDataSetClassificationMC("tmp/train_classMC.txt", 1000, dim);
    createDataSetClassificationMC("tmp/test_classMC.txt", 200, dim);
    TrainingData trainData("tmp/train_classMC.txt", batchSize);
    TrainingData testData("tmp/test_classMC.txt", 200);
    std::vector<std::pair<unsigned, LayerType>> topology{{dim, INPUT}, {40, RELU}, {unsigned(pow(2, dim)), SOFTMAX}};
#endif

#if DEMOCASE == MNIST
    // make sure you generate the training data using the Jupiter notebook first
    const unsigned batchSize = 100;
    opts.optimizer = ADAM;
    opts.LRD = 1.0;
    opts.alphaInit = 0.0000001;
    opts.loss = LOGISTIC;
    opts.regularization = NONE;
    opts.LOG_bestModel = true;
    TrainingData trainData("tmp/mnist/mnist_train.txt", batchSize); // 60000
    TrainingData testData("tmp/mnist/mnist_test.txt", 1000);        // 10000
    trainData.usePreload(true, 28 * 28, 10);
    testData.usePreload(true, 28 * 28, 10);
    std::vector<std::pair<unsigned, LayerType>> topology{{28 * 28, INPUT}, {400, SIGMOID}, {250, SIGMOID}, {100, SIGMOID}, {10, SOFTMAX}};
#endif

    // Net myNet("tmp/models/test.txt", opts); // load from File
    Net myNet(topology, opts); // create new neural network

    myNet.train(trainData, testData, totalIters);
    
    // myNet.save("tmp/models/test.txt"); //  save current model to file
    printPredictions(myNet, testData); // print some outputs / errors
}
