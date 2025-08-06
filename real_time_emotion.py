import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("model/emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face, (48, 48))  # FER2013 size
    roi = face_resized.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi)[0]
    label = emotion_labels[np.argmax(prediction)]

    # Show label on the video frame
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Facial Emotion Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
