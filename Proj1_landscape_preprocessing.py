import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm
import re

def sort_key(filename):
    match = re.match(r'([A-Za-z]+)(\d+)\.jpg', filename)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (filename, 0)

### Data Processing For Rating Data
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
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
landmarks_by_name = dict()

print('calculating landscapes for 5500 people')
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

data = []
print("Final data processing")
for filename in tqdm(filenames):
    data.append([filename, rating_by_name[filename], landmarks_by_name[filename]])
    
df = pd.DataFrame(data, columns = ['Filename', 'Rating', 'Landmarks'])
df.to_csv('landscape_AMS325.csv')

print("landscape csv file has successfully generated.")