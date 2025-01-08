# NN++

A simple fully connected feed forward neural network implementation in C++. This toolbox can be used for classification and regression and includes a number of different activation functions and optimizers. This project serves as lesson for me to get more comfortable with modern C++ features :D

# How to use?
Simply clone or download this repository. The vector-matrix computations are based on the [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) header-only library. The `main.cpp` files demonstrates the toolboxes basic use.

To build this executable, make sure your local Eigen3 (I´m using version 3.3.8) library is accessible by your compiler or place it in the current working directory.

# Creation, Loading, and Saving of a Neural Network

To create a new `Net` object pass a network topology of type `std::vector<std::pair<unsigned, LayerType>>` to `myNet`. The second input is an `OptimizerOpts` object specifying the the settings for the training procedure. The first entry of the pair therby represents the size of the layer of type `LayerType` (see below).
Alternatively, a file path to a neural network can be provided as a first input instead.
To save a network to a file use the train method and pass a file location, e.g., `myNet.save("tmp/models/test.txt");`, or use the provided logging capabilities (following below under Logging).

## Layer Types
When creating the topology for the network you can choose one of the following options when selecting a layer type.

| Name | Description |
| ------------- | ------------- |
| `INPUT` | input layer |
| `LINEAR` | linear layer |
| `RELU` | ReLU layer |
| `LEAKY_RELU` | leaky ReLU layer |
| `TANH` | tanh layer |
| `SIGMOID` | sigmoid layer |
| `SOFTMAX` | softmax layer (not vectorized) |

# Input and Output Data
To create simple input output data sets for regression and classification I provided some utilities, namely,
`createDataSet` for regression, `createDataSetClassification` for classification, and `createDataSetClassificationMC` for classification with multiple labels. Moreover, I included a Jupiter notebook to create a training data file for the MNIST data set for handwritten digit classification. I could archive an accuracy of about 98% on the validation data with a comparatively small fully connected neural network trained on a laptop computer.

To create a data set simply pass the file path to the txt file and the batch size you want to use.
The TrainingData object also conveniently implements two methods, namely, `next` which returns the next (possibly incomplete) batch of data and `usePreload` which preloads the training data into memory upon creation rather than fetching it from the txt file directly. This may take some time upfront but usually gives a slight reduction in training time for more epochs.

In case you want to use your own data set, the txt file must have the following format
```
in: 1 0
out: 0.5 0.2
in: 0 1
out: -0.3 0.7
...
```

# Training
Upon creation of the of an `OptimizerOpts` object the following options for the training of the network can be set. You can do a single training step using the Net's `trainStep` method or multiple epochs using the `train` method. The network can be evaluated using the `eval` method.

## Loss Types
You can choose one of the following options when selecting a loss type.
| Name | Description |
| ------------- | ------------- |
| `MSE` | mean squared error |
| `MAE` | mean absolute error |
| `HUBER` | Huber loss, options see below |
| `SSE` | sum of squared errors, options see below |
| `LOGISTIC` | logistic loss |
| `LOGISTIC_CLIPPED` | clipped logistic loss, options see below |

| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `loss` | `LossType` | `MSE` | select on of the above loss functions |
| `Huber_delta` | `double` | 0.1 | sets the parameter delta for the huber loss function |
| `SSE_factor` | `double` | 1.0 | multiplicative factor for the SSE loss function |
| `LOGISTIC_CLIPPED_delta` | `double` | 0.000001 | clipping threshold for the LOGISTIC_CLIPPED loss, values are clipped to a range between LOGISTIC_CLIPPED_delta and 1.0-LOGISTIC_CLIPPED_delta |

## Regularization Types
You can choose one of the following options when selecting a regularization type.
| Name | Description |
| ------------- | ------------- |
| `L1` | L1 or absolute regularization of the weights |
| `L2` | L2 or squared regularization of the weights |
| `NONE` | no regularization of the weights |

| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `regularization` | `RegularizationType` | `NONE` | type of regularization |
| `lambda` | `double` | 0.005 | multiplicative factor of the regularization term |

## Optimizers
You can choose one of the following options when selecting an optimizer.

| Name | Description |
| ------------- | ------------- |
| `SGD` | stochastic gradient decent, the mini batch size is determined by the training data object |
| `MOMENTUM` | adds a portion of the previous gradient to the SGD term |
| `ADAM` | ADAM implementation the the gradient decent algorithm |
| `AMSGRAD` | the AMSGRAD implementation of the ADAM algorithm |

| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `optimizer` | `Optimizer` | `MOMENTUM` | selects one of the optimizers above |
| `alphaInit` | `double` | 0.4 | initial learning rate |
| `alpha` | `double` | `alphaInit` | current learning rate |
| `LRD` | `double` | 1.0 | exponential learning rate decay. After each training pass the current alpha is multiplied with LRD |

In addition you have the following optimizer specific settings.

### Momentum
| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `regularization` | `RegularizationType` | `NONE` | type of regularization |
| `MOMENTUM_moment` | `double` | 0.1 | portion of the previous gradient that is used vs. the current gradient |

### ADAM and AMSGRAD
For further information on this algorithm, see [ADAM](https://arxiv.org/abs/1412.6980) and [AMSGRAD](https://arxiv.org/abs/1904.09237).
| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `regularization` | `RegularizationType` | `NONE` | type of regularization |
| `ADAM_beta1` | `double` | 0.9 | parameter beta1 if the ADAM algorithm |
| `ADAM_beta2` | `double` | 0.999 | parameter beta2 if the ADAM algorithm | 
| `ADAM_eps` | `double` | `pow(10.0, -8)` | parameter eps if the ADAM algorithm |
| `ADAM_t` | `int` | 0 | time step of the ADAM algorithm, this parameter is incremented automatically and does not need to me manually modified |

## Logging
This toolbox provides several logging capabilities.
These are used when the train method of the Net object is called.
You can choose one of the following options.
| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `LOG_history` | `bool` | `true` | creates a txt file with the performance of the model on the test data in each epoch |
| `LOG_bestModel` | `bool` | `false` | saves the current best model of all epochs to a file |
| `LOG_checkpoints` | `bool` | `false` | saves a model of the net after each epoch | 
| `LOG_modelFolder` | `std::string` | `"tmp/models/"` | file path for the models to be logged |
| `LOG_historyFolder` | `std::string` | `"tmp/history/"` | file path for the history files |

## Parallel Settings (not implemented)
In parallel mode for each training pass `parallelBatches` copies of the current net are created. Each of these nets is trained on a different batch of training data. Afterwards the gradients resulting from the back propagation are averaged and applied to the original net. This option is not yet implemented, however.

| Name | Type | Default | Description |
| ------------- | ------------- | ------------- | ------------- |
| `parallel` | `bool` | `false` | toggles between singe thread and parallel execution (not yet implemented) |
| `parallelBatches` | `int` | 12 | number of copies of the net to work on in parallel |
