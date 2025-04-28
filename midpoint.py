import pandas as pd

# Load the data from CSV
file_path = './c8_1_142_c_out.csv'  # Update this to your CSV file path
data = pd.read_csv(file_path)

# Function to calculate the center of the bounding box
def calculate_center(row):
    x_coords = [row['p1x'], row['p2x'], row['p3x'], row['p4x']]
    y_coords = [row['p1y'], row['p2y'], row['p4y'], row['p4y']]
    center_x = sum(x_coords) / 4
    center_y = sum(y_coords) / 4
    return center_x, center_y

# Apply the function to each row in the dataframe
data['center_x'], data['center_y'] = zip(*data.apply(calculate_center, axis=1))

data.to_csv(file_path, index=False)

print("CSV file has been updated with center points and saved back to:", file_path)
