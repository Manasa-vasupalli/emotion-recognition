import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('my_model.h5')

img_height, img_width = 48, 48  
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Converting the frame to grayscale and resize it to match the model input
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (img_height, img_width))
    
    img_array = img_to_array(resized_frame)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  

    # Predict emotion
    prediction = model.predict(img_array)
    emotion_label = class_labels[np.argmax(prediction)]  

    
    cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Recognition', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
