import cv2
import numpy as np
import os


dataset_path = './face_data/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

name = input("Enter your name: ").strip().lower()
if not name:
    print("❌ Name cannot be empty!")
    exit()

print(f"\n🎬 Collecting data for '{name}'. Press 'q' to stop early.\n")

cap = cv2.VideoCapture(0)

face_data = []
skip = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Captured {len(face_data)} faces...")

  
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"Capturing {name}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Collecting Face Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(face_data) >= 50: 
        break

cap.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
np.save(os.path.join(dataset_path, name + '.npy'), face_data)
print(f"\n✅ Data saved successfully as {name}.npy with shape {face_data.shape}")
