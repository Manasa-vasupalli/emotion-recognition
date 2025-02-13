# emotion-recognition
This project implements emotion recognition using Convolutional Neural Networks (CNNs) to classify facial expressions into different emotions such as happiness, sadness, anger, and surprise. It uses the FER-2013 and disgust images from face-emotion-recognition images dataset for training and supports real-time emotion detection via webcam. 
The goal is to develop a model that can accurately recognize emotions from facial images with a user-friendly interface.

# Dataset Link 

Link to dataset :  https://www.kaggle.com/datasets/msambare/fer2013/data
https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition

# Significance of the Project

The convergence of deep learning and quantum computing in this project reflects the forefront of AI research, pushing the limits of what emotion recognition systems can achieve. By addressing challenges like data imbalance and computational latency, this project lays the foundation for scalable and accurate emotion recognition systems, with the potential to transform industries and improve human-machine interactions.

# Final Model Architecture

The CNN architecture is implemented in the create_classical_model function. Key components include:
1.Convolutional Layers:
  a.Each Conv2D layer applies filters to extract features like edges and textures.
  b.The first layer uses 64 filters, increasing to 256 in later layers, enhancing the model's ability to learn complex patterns.
  c.ReLU activation introduces non-linearity to learn intricate relationships.
2.Batch Normalization:
  a.Normalizes the output of each layer to stabilize learning and speed up convergence.
3.Pooling Layers:
  a.MaxPooling2D reduces the spatial dimensions, retaining the most important features while reducing computational complexity.
4.Dropout Layers:
  a.Dropout rates (0.3 to 0.5) randomly deactivate neurons during training, preventing overfitting.
5.Flatten Layer:
  a.Converts the 2D feature maps into a 1D vector for the fully connected layers.
6.Fully Connected Layers:
  a.Dense layer with 256 neurons learns high-level combinations of features.
  b.The final dense layer has num_classes neurons with a softmax activation to output probabilities for each emotion category.


# Quantum Approaches:

1.QUBO for Hyperparameter Tuning with CONBQA:
  a.Optimized number of neurons with the Conbqa Library.
2.QUBO for Feature Selection:
  a.Leap Hybrid Sampler identified critical features, enhancing training efficiency.
3.Pennylane Quantum Circuits:
  a.Quantum gates (RX, RY, and CNOT) applied for entanglement.
  b.Experimented with two qubits for fully quantum layers.





