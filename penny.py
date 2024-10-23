import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pennylane as qml
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_height = 48  
img_width = 48   
num_classes = 7  


def circuit(inputs):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def quantum_circuit(input_tensor):
        qml.RX(input_tensor[0], wires=0)
        qml.RY(input_tensor[0], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    return tf.vectorized_map(quantum_circuit, inputs)

class QuantumConvLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(QuantumConvLayer, self).__init__(**kwargs)

    def call(self, inputs):
        quantum_output = circuit(inputs)  
        return quantum_output

def create_quantum_model():
    input_layer = layers.Input(shape=(img_height, img_width, 1))
    x = layers.Flatten()(input_layer)
    quantum_output = QuantumConvLayer()(x)
    x = layers.Dropout(0.5)(quantum_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x) 
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = data_gen.flow_from_directory(
    "Dataset/train",
    target_size=(img_height, img_width),
    batch_size=32,
    color_mode='grayscale',
    class_mode='sparse'
)


validation_data = keras.preprocessing.image_dataset_from_directory(
    "Dataset/test", 
    image_size=(img_height, img_width),
    batch_size=32,
)

quantum_model = create_quantum_model()
quantum_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

quantum_model.fit(train_data, validation_data=validation_data, epochs=30)  

predictions = quantum_model.predict(validation_data)

loss, accuracy = quantum_model.evaluate(validation_data)
print(f"Test accuracy: {accuracy:.2f}")