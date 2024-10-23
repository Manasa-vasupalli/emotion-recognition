import numpy as np
from dwave.system import LeapHybridSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from dwave.system import DWaveSampler, EmbeddingComposite
import dwave.inspector

def create_model(learning_rate=0.0001, dropout_rate=0.3, num_layers=3, filters=32):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Conv2D(filters, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        else:
            model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 classes for emotion recognition
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_qubo():
    Q = {
        (0, 0): 1,   # Cost associated with learning rate = 0.0001
        (0, 1): -1,  # Cost associated with learning rate = 0.001
        (1, 0): 1,   # Cost associated with dropout rate = 0.3
        (1, 1): -1,  # Cost associated with dropout rate = 0.5
        (2, 0): 1,   # Cost associated with number of layers = 3
        (2, 1): -1,  # Cost associated with number of layers = 4
        (3, 0): 1,   # Cost associated with filters = 16
        (3, 1): -1,  # Cost associated with filters = 32
        (3, 2): -2,  # Cost associated with filters = 64
    }
    return Q

def quantum_hyperparameter_optimization():
    Q = create_qubo()  # Define the QUBO matrix
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q, num_reads=100)
    # sampler = LeapHybridSampler()  
    # response = sampler.sample_qubo(Q)
    
    best_sample = response.first.sample  
    
    return best_sample  

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'Dataset/train',  
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

test_generator = test_datagen.flow_from_directory(
    'Dataset/test',  
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

num_epochs = 50  
best_params = quantum_hyperparameter_optimization()
        
learning_rate = 0.0001 if best_params[0] == 0 else 0.001
dropout_rate = 0.3 if best_params[1] == 0 else 0.5
num_layers = 3 if best_params[2] == 0 else 4  
filters = 16 if best_params[3] == 0 else (32 if best_params[3] == 1 else 64) 



for epoch in range(num_epochs):
    model = create_model(learning_rate, dropout_rate, num_layers, filters)  
    history = model.fit(train_generator, validation_data=test_generator)

    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"Epoch {epoch + 1}/{num_epochs} - Training accuracy: {train_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}")

model.save('final_model.keras')

loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.2f}")


