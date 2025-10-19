import cv2
import numpy as np
import os

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

dataset_path = './face_data/'
face_data = []
labels = []
class_id = 0
names = {}

if not os.path.exists(dataset_path):
    print(" Dataset folder not found. Run face_data_collect.py first.")
    exit()

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        data_item = np.load(os.path.join(dataset_path, fx))
        face_data.append(data_item)
        names[class_id] = fx[:-4]
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

if len(face_data) == 0:
    print(" No .npy files found in face_data/. Please collect data first.")
    exit()

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset.reshape(face_dataset.shape[0], -1), face_labels), axis=1)

print(" Training data loaded with shape:", trainset.shape)
print(" Classes:", names)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        pred_name = names[int(out)]

        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

