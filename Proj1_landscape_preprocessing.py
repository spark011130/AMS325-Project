import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm
import re
import pickle

"""
Sorts filenames based on a custom sorting rule, which separates the prefix (letters) and suffix (numbers).

Parameters:
filename (str): The name of the file to be sorted.

Returns:
tuple: A tuple containing the prefix and number (as an integer), used for sorting filenames.
"""
def sort_key(filename):
    match = re.match(r'([A-Za-z]+)(\d+)\.jpg', filename)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (filename, 0)

### Data Processing For Rating Data
'''
Reads the rating data from an Excel file and sorts the filenames lexicographically using a custom sorting function.
Calculates the average rating for each unique filename.
'''
wd = os.getcwd() 
rating_df = pd.read_excel( wd + '/Datasets/All_Ratings.xlsx' ) # Data Frame for ratings
rating_df = rating_df.sort_values(by="Filename")
del rating_df['Rater']
del rating_df['original Rating']
# Changing lexicographical order to custom sorting rules, to make it perfectly sorted.
filenames = sorted(rating_df['Filename'].unique(), key=sort_key)

rating_by_name = dict()
print('averaging ratings for 60 people (5500*60 = 330000 calculation)')
for filename in tqdm(filenames):
    rating_by_name[filename] = rating_df[rating_df['Filename'] == filename]['Rating'].mean()

### Landmark Analyzer
'''
Analyzes the face landmarks of images using the MediaPipe FaceMesh model.
Calculates and stores the landmark coordinates for each image where a face is detected.
'''

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
landmarks_by_name = dict()

print('calculating landmarks for 5500 people')
for filename in tqdm(filenames):
    image = cv2.imread(wd + '/Datasets/images/' + filename)
    # cv2 reads in BGR format, so requires it to be reversed.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract landmark (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=en)
    results = face_mesh.process(rgb_image)
    
    # If face were not detected
    if not results.multi_face_landmarks:
        continue
    h, w, _ = image.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
    landmarks_by_name[filename] = landmarks

# Processes the data by combining ratings and landmarks for each image, then saves the result into a CSV file.

data = []
print("Final data processing")
for filename in tqdm(filenames):
    data.append([filename, rating_by_name[filename], landmarks_by_name[filename]])
    
df = pd.DataFrame(data, columns = ['Filename', 'Rating', 'Landmarks'])
df.to_csv('landmark_AMS325.csv')

print("landmark csv file has successfully generated.")

# Filters the DataFrame for entries that start with a specified section (e.g., 'CF') and saves the ratings for that section.
# Four choices: AM, AF, CM, CF to train.
section = 'AM'
df = df[df['Filename'].str.startswith(section)]
df = df.reset_index(drop=True)    
ratings = df['Rating'].tolist()
with open(f'files/section.pkl', 'wb') as file:
    pickle.dump(section, file)
with open(f'files/{section}_ratings.pkl', 'wb') as file:
    pickle.dump(ratings, file)

print("Ratings has successfully generated.")