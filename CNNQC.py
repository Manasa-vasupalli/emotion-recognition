import numpy as np
import dimod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dwave.system import DWaveSampler, EmbeddingComposite
from tensorflow.keras.utils import to_categorical
from dwave.system import LeapHybridSampler

train_data_dir = 'Dataset/train'  
validation_data_dir = 'Dataset/test'  
img_size = (48, 48)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation='softmax'))
    
    return model


cnn_model = create_cnn_model()
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_model.fit(train_generator, epochs=20, validation_data=validation_generator)  


def extract_features(model, generator):
    features = model.predict(generator)
    return features

X_train_features = extract_features(cnn_model, train_generator)
X_test_features = extract_features(cnn_model, validation_generator)


X_train_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_flat = X_test_features.reshape(X_test_features.shape[0], -1)


y_train = train_generator.classes
y_test = validation_generator.classes

def create_qubo(X_train_flat, lambda_reg=0.1):
    num_features = X_train_flat.shape[1]
    Q = {}

    weights = np.random.rand(num_features)
    for i in range(num_features):
        Q[(i, i)] = -weights[i]  
        for j in range(i + 1, num_features):
            Q[(i, j)] = 0  
    for i in range(num_features):
        Q[(i, i)] += lambda_reg  

    return Q

qubo = create_qubo(X_train_flat, lambda_reg=0.1)

sampler = LeapHybridSampler()  
response = sampler.sample_qubo(qubo)

best_solution = response.first.sample
selected_features = [i for i in best_solution if best_solution[i] == 1]

print(f"Selected features: {selected_features}")

X_train_selected = X_train_flat[:, selected_features]
X_test_selected = X_test_flat[:, selected_features]

X_train_selected = X_train_selected.reshape(X_train_selected.shape[0], 1, 1, len(selected_features))
X_test_selected = X_test_selected.reshape(X_test_selected.shape[0], 1, 1, len(selected_features))

final_model = Sequential()
final_model.add(Flatten(input_shape=(1, 1, len(selected_features))))  
final_model.add(Dense(64, activation='relu'))
final_model.add(Dropout(0.5))
final_model.add(Dense(7, activation='softmax'))  


y_train_onehot = to_categorical(y_train, num_classes=7)
y_test_onehot = to_categorical(y_test, num_classes=7)


final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
final_model.fit(X_train_selected, y_train_onehot, epochs=20, validation_data=(X_test_selected, y_test_onehot))


accuracy = final_model.evaluate(X_test_selected, y_test_onehot)
print(f"Final model accuracy with selected features: {accuracy[1]:.4f}")
