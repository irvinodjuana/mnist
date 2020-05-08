# MNIST
MNIST implementations from scratch

## Overview
The MNIST problem is a famous image classification problem based on the MNIST dataset - a collection of tens of thousands of handwritten digits from 0-9. In this repo, I have implemented five ML models from scratch using Python and NumPy. 

Please note that some of the utility functions and specialized optimization routines are taken from The University of British Columbia's CPSC 340 course assignments. 

## Performance
| -m | Model Name | Accuracy |
| --- | --- | ---|
| 1.1 | KNN | 96.8% |
| 1.2 | Logistic Regression | 96.3% |
| 1.3 | SVM | 96.7% |
| 1.4 | Multi-layer Perceptron | 98.0% |
| 1.5 | CNN | 95.0% |

The "-m" flag above is used for running the Python script. The accuracies reported above are the best classification accuracies I achieved for each model on the MNIST test set. Records for the best accuracies historically achieved for various models can be found here: http://yann.lecun.com/exdb/mnist/

## Running the Code

To start, this code requires Python 3.6+ and pip to be installed on your machine. 
Run the code below in your terminal to clone the repository and setup your environment.

```
git clone https://github.com/irvinodjuana/mnist.git
cd mnist/
pip install -r requirements.txt
```

To train and evaluate one of the models listed above, navigate to the code directory and run main.py with one of the -m arguments listed above. For example, to run the KNN model, use:

```
cd code/
python3 main.py -q 1.1
```

Some of the models require a significant amount of memory, and may not be runnable on machines with less memory. Each model's hyperparameters can be adjusted in the main.py script as well.



