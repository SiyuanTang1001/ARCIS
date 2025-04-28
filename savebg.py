import numpy as np
import cv2
import random

def getRandomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def centroid(vertexes):
    x_list = [vertex[0][0] for vertex in vertexes]
    y_list = [vertex[0][1] for vertex in vertexes]
    x = int(sum(x_list) / len(vertexes))
    y = int(sum(y_list) / len(vertexes))
    return (x, y)

filename = "DJI_20250408122911_0003stabilizationstabChecked.mp4"
filePath = "./output_stable_3/" + filename
cap = cv2.VideoCapture(filePath)

# Load zones from numpy file
zones = np.load("./DJI_20250408122911_0003stabilizationstabCheckedzones.npy", allow_pickle=True)
randomColor = [getRandomColor() for _ in range(len(zones))]

# Choose a random frame
frameId = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform())
cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)
ret, img = cap.read()

if ret:
    for i, zone in enumerate(zones):
        cv2.polylines(img, [zone], True, randomColor[i], 5)
        centerPoint = centroid(zone)
        cv2.putText(img, str(i), (centerPoint[0], centerPoint[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 3, randomColor[i], 5)

    # Save the annotated frame
    cv2.imwrite("c8.png", img)
    print("Frame saved as: ", filename)

cap.release()
cv2.destroyAllWindows()
