# emotion-recognition
This project leverages Convolutional Neural Networks (CNNs) for emotion recognition, classifying facial expressions into distinct emotions such as happiness, sadness, anger, surprise, and disgust. It utilizes the FER-2013 dataset, supplemented with additional disgust images from the face-emotion-recognition dataset, to enhance model accuracy. The system supports real-time emotion detection via webcam, enabling dynamic interaction. With a focus on robust feature extraction and classification, the model aims to achieve high precision in recognizing human emotions. Additionally, a user-friendly interface enhances accessibility, making it suitable for applications in customer service, mental health analysis, and human-computer interaction.

# Dataset Link 

Link to dataset :  https://www.kaggle.com/datasets/msambare/fer2013/data
https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition

# Significance of the Project

This project represents a cutting-edge fusion of deep learning and quantum computing, pushing the boundaries of AI-driven emotion recognition. By leveraging the strengths of both technologies, it tackles critical challenges such as data imbalance, computational latency, and model scalability. The integration of quantum-inspired optimization techniques enhances processing efficiency, enabling faster and more accurate emotion classification. This pioneering approach not only advances the field of affective computing but also lays the groundwork for real-world applications in healthcare, human-computer interaction, and customer experience optimization. With its potential to redefine emotion recognition systems, this project marks a significant step toward the future of AI-driven emotional intelligence.

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





