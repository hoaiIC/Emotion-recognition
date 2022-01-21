from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/cnn.195-0.98.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]
cv2.ocl.setUseOpenCL(False)
# load photo
cv2.namedWindow('your_face')
img = cv2.imread('test/a6.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
canvas = np.zeros((250, 300, 3), dtype="uint8")
frameClone = img.copy()
if len(faces) > 0:
    # faces = sorted(faces, reverse=True,
    #                 key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    for (fX, fY, fW, fH) in faces:
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]

        emotion_probability = np.max(preds)

        label = EMOTIONS[preds.argmax()]
        text = "{}: {:.2f}%".format(label, emotion_probability * 100)
        cv2.putText(frameClone, label, (fX, fY),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (25, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 0, 255), 2)

# for c in range(0, 3):
#     frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#     (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#     10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

# cv2.imshow('your_face', cv2.resize(frameClone,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
cv2.imshow('your_face', cv2.resize(frameClone, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
cv2.waitKey()
cv2.destroyAllWindows()
