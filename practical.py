#1. How do you create a simple perceptron for basic binary classification!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Example dataset (X: inputs, y: labels)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1]) 

# Build perceptron model
model = Sequential([
    Dense(1, input_dim=2, activation='sigmoid')  # 1 neuron, sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, verbose=1)

#2.How can you build a neural network with one hidden layer using Keras!
# Build the model
model = Sequential([
    Dense(16, input_shape=(4,), activation='relu'),  # Hidden layer with 16 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()




#3.How do you initialize weights using the Xavier (Glorot) initialization method in Keras!
from tensorflow.keras.initializers import GlorotUniform

# Example layer with Xavier (Glorot) initialization
model.add(Dense(16, input_shape=(4,), activation='relu', kernel_initializer=GlorotUniform()))



#4.How can you apply different activation functions in a neural network in Keras!
# Example layers with different activation functions
model = Sequential([
    Dense(32, activation='relu'),        # ReLU activation
    Dense(16, activation='tanh'),        # Tanh activation
    Dense(1, activation='sigmoid')       # Sigmoid activation for output
])




#5.How do you add dropout to a neural network model to prevent overfitting!
from tensorflow.keras.layers import Dropout

# Add Dropout after a Dense layer
model = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),  # Drop 50% of neurons to prevent overfitting
    Dense(1, activation='sigmoid')
])



#6.How do you manually implement forward propagation in a simple neural network!
X = np.array([1, 2])
weights = np.array([0.5, -1.5])
bias = 0.3

# Activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Forward propagation
output = relu(np.dot(X, weights) + bias)
print(f"Output: {output}")




#7.How do you add batch normalization to a neural network model in Keras!
from tensorflow.keras.layers import BatchNormalization

# Add Batch Normalization after a Dense layer
model = Sequential([
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])




#8.How can you visualize the training process with accuracy and loss curves!
import matplotlib.pyplot as plt

# Fit the model and save training history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

# Plot training & validation accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.title('Training and Validation Accuracy and Loss')
plt.show()




#9.How can you use gradient clipping in Keras to control the gradient size and prevent exploding gradients!
from tensorflow.keras.optimizers import Adam

# Apply gradient clipping with a maximum norm of 1.0
optimizer = Adam(clipnorm=1.0)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])




#10. How can you create a custom loss function in Keras!
from tensorflow.keras.losses import Loss

class CustomLoss(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))  # Example: Mean Squared Error

# Compile the model with the custom loss
model.compile(optimizer='adam', loss=CustomLoss(), metrics=['accuracy'])




#11.How can you visualize the structure of a neural network model in Keras?
from tensorflow.keras.utils import plot_model

# Generate a plot of the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
