import pandas as pd
from ast import literal_eval

df = pd.read_csv('/Users/andypark/Desktop/2024 FALL/AMS 325/Project/landscape_AMS325.csv')
del df['Unnamed: 0']
df['Landmarks'] = df['Landmarks'].apply(literal_eval)

landmarks = df.loc[0]['Landmarks']

print(landmarks)