# MNIST Digit Classification from Scratch

A complete implementation of a neural network for handwritten digit recognition using the MNIST dataset, built entirely from scratch using only NumPy, Pandas, and Matplotlib.

## 🎯 Project Overview

This project demonstrates the fundamental concepts of neural networks by implementing a 2-layer neural network from scratch to classify handwritten digits (0-9) from the famous MNIST dataset. The implementation includes forward propagation, backpropagation, and gradient descent optimization without using any deep learning frameworks.

## 🧠 Model Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit class)
- **Loss Function**: Cross-entropy loss
- **Optimization**: Gradient descent

## 📊 Dataset

The project uses the MNIST dataset containing:
- **Training Set**: 41,000 samples (after splitting)
- **Development Set**: 1,000 samples
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)

## 🚀 Features

- **From-scratch implementation**: No deep learning frameworks used
- **Data preprocessing**: Normalization and train/dev split
- **Visualization**: Display predictions with actual digit images
- **Training monitoring**: Real-time accuracy tracking during training
- **Modular design**: Clean, readable functions for each component

## 📋 Requirements

```python
numpy
pandas
matplotlib
```

## 🔧 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Neural-network-from-scratch-digit-classification-MNIST

```

2. Install required packages:
```bash
pip install numpy pandas matplotlib
```

3. Download the MNIST dataset from Kaggle:
   - Go to [Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)
   - Download `train.csv` and place it in the `/kaggle/input/digit-recognizer/` directory

## 🏃‍♂️ Usage

### Basic Training

```python
# Load and preprocess data
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
np.random.shuffle(data)

# Split data
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:] / 255.

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:] / 255.

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)
```

### Making Predictions

```python
# Test on a specific sample
test_prediction(index=8, W1, b1, W2, b2)
```

### Custom Prediction Function

```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```

## 📈 Model Performance

The model achieves approximately **83%** accuracy on the training set after 500 iterations with a learning rate of 0.1.

Training progress example:
```
Iteration: 0   - Accuracy: 8.39%
Iteration: 50  - Accuracy: 42.23%
Iteration: 100 - Accuracy: 60.72%
Iteration: 150 - Accuracy: 68.73%
Iteration: 200 - Accuracy: 73.63%
Iteration: 250 - Accuracy: 76.71%
Iteration: 300 - Accuracy: 79.03%
Iteration: 350 - Accuracy: 80.74%
Iteration: 400 - Accuracy: 81.98%
Iteration: 450 - Accuracy: 83.00%
```

## 🔍 Key Functions

### Core Neural Network Functions

- **`init_params()`**: Initialize weights and biases randomly
- **`forward_prop()`**: Forward propagation through the network
- **`backward_prop()`**: Compute gradients using backpropagation
- **`update_params()`**: Update parameters using gradient descent
- **`gradient_descent()`**: Main training loop

### Activation Functions

- **`ReLU()`**: Rectified Linear Unit activation
- **`softmax()`**: Softmax activation for output layer
- **`ReLU_deriv()`**: Derivative of ReLU for backpropagation

### Utility Functions

- **`one_hot()`**: Convert labels to one-hot encoding
- **`get_predictions()`**: Get predicted classes from probabilities
- **`get_accuracy()`**: Calculate model accuracy
- **`test_prediction()`**: Visualize predictions with images

## 🎨 Visualization

The project includes visualization capabilities to:
- Display handwritten digit images
- Show model predictions vs. actual labels
- Monitor training progress

## 🔬 Technical Details

### Forward Propagation
```
Z1 = W1 · X + b1
A1 = ReLU(Z1)
Z2 = W2 · A1 + b2
A2 = Softmax(Z2)
```

### Backpropagation
```
dZ2 = A2 - Y_one_hot
dW2 = (1/m) · dZ2 · A1^T
db2 = (1/m) · sum(dZ2)
dZ1 = W2^T · dZ2 * ReLU'(Z1)
dW1 = (1/m) · dZ1 · X^T
db1 = (1/m) · sum(dZ1)
```

### Parameter Update
```
W1 = W1 - α · dW1
b1 = b1 - α · db1
W2 = W2 - α · dW2
b2 = b2 - α · db2
```

## 🎯 Hyperparameters

- **Learning Rate (α)**: 0.10
- **Iterations**: 500
- **Hidden Layer Size**: 10 neurons
- **Batch Size**: Full batch (all training samples)

## 🚀 Potential Improvements

1. **Architecture Enhancements**:
   - Add more hidden layers
   - Experiment with different layer sizes
   - Try different activation functions

2. **Optimization**:
   - Implement mini-batch gradient descent
   - Add momentum or Adam optimizer
   - Implement learning rate scheduling

3. **Regularization**:
   - Add L1/L2 regularization
   - Implement dropout
   - Add batch normalization

4. **Data Augmentation**:
   - Rotation, scaling, translation
   - Noise injection
   - Elastic deformations



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or suggestions, please feel free to reach out!

---

*This project was created for educational purposes to demonstrate the fundamental concepts of neural networks and deep learning.*
