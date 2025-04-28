import pandas as pd
import numpy as np
import cv2

# User input
inputName = './ARCISD2-master/shaoyan_drone_4_26/DJI_20250408122911_0004stabilizationstabChecked.csv'
zones = np.load("./DJI_20250408122911_0004stabilizationstabCheckedoutzones.npy", allow_pickle=True)

# Load data
df = pd.read_csv(inputName)

# Function to determine lane number
def assign_lane_number(row, zones):
    point = (int(row['carCenterX']), int(row['carCenterY']))
    for i, zone in enumerate(zones):
        if cv2.pointPolygonTest(zone, point, False) == 1:
            return i
    return -1

# Apply function across DataFrame
df['laneNumber'] = df.apply(assign_lane_number, zones=zones, axis=1)

# Output filename
outputName = inputName[:-4] + "withLane.csv"

# Save updated DataFrame
df.to_csv(outputName, index=False)
print(f"Data with lane numbers saved to {outputName}")
