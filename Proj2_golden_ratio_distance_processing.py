import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import pickle

def average_position(p1, p2):
    return [(a + b) / 2 for a, b in zip(p1, p2)]

class Parameter:
    def __init__(self, position):
        self._position = position
        self._distance = None
        self._ratio = None

    # Getter for position
    @property
    def position(self):
        return self._position

    # Setter for position
    @position.setter
    def position(self, value):
        self._position = value
        # Update distance and ratio when position changes
        self._distance = self.positions_to_distances(self._position)
        self._ratio = self.distances_to_ratios(self._distance)

    # Getter for distance
    @property
    def distance(self):
        return self._distance

    # Setter for distance
    @distance.setter
    def distance(self, value):
        self._distance = value
        self._ratio = self.distances_to_ratios(self._distance)

    # Getter for ratio
    @property
    def ratio(self):
        return self._ratio

    # Setter for ratio
    @ratio.setter
    def ratio(self, value):
        self._ratio = value

    def distance_position(self, p1, p2):
        return ((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)**0.5

    def distances_to_ratios(self, array):
        max_distance = array.max()
        ret = array / max_distance
        return ret

    def positions_to_distances(self, array):
        raise NotImplementedError("Abstract class: the positions_to_distances method must be implemented in a subclass.")

class Parameter_Continuous(Parameter):
    def __init__(self, position):
        super().__init__(position)
        self.distance = self.positions_to_distances(self._position)
        self.ratio = self.distances_to_ratios(self._distance)

    def positions_to_distances(self, array):
        ret = np.zeros(len(array)-1)
        for i in range(len(array) - 1):
            ret[i] = self.distance_position(array[i], array[i + 1])
        return ret

class Parameter_By_2(Parameter):
    def __init__(self, position_pairs):
        super().__init__(position_pairs)
        self.distance = self.positions_to_distances(self._position)
        self.ratio = self.distances_to_ratios(self._distance)

    def positions_to_distances(self, position_pairs):
        ret = np.zeros(len(position_pairs))
        for i, pair in enumerate(position_pairs):
            ret[i] = self.distance_position(pair[0], pair[1])
        return ret

with open("/Users/andypark/Desktop/2024 FALL/AMS 325/Project/section.pkl", 'rb') as file: 
    section = pickle.load(file)

# Load and preprocess data
df = pd.read_csv('/Users/andypark/Desktop/2024 FALL/AMS 325/Project/landscape_AMS325.csv')
if 'Unnamed: 0' in df.columns:
    del df['Unnamed: 0']
df = df[df['Filename'].str.startswith(section)]
df['Landmarks'] = df['Landmarks'].apply(literal_eval)
df = df.reset_index(drop=True)

N = len(df)
print(df.shape)
print(df.head())
parameters = [0 for _ in range(N)]
print(N)
for i in tqdm(range(N)):
    # Retrieve landmarks from a specific row
    landmarks = df.loc[i]['Landmarks']

    # Parameter_Continuous classes
    # four parameters
    vertical = Parameter_Continuous([
        landmarks[10],  # Forehead
        landmarks[9],   # Mid eyebrow
        landmarks[2],   # Under the nose
        landmarks[17],  # Lower lip
        landmarks[152]  # Chin
    ])

    # five parameters
    horizontal = Parameter_Continuous([
        average_position(landmarks[162], landmarks[127]),  # Left face end
        landmarks[33],   # Left eye left
        landmarks[133],  # Left eye right
        landmarks[362],  # Right eye left
        landmarks[263], # Righy eye right
        average_position(landmarks[389], landmarks[356]) # Left face end
    ])

    # Parameter_By_2 classes
    # two parameters
    nose_to_lip = Parameter_By_2([
        [landmarks[279], landmarks[49]], # Nose horizontal length
        [landmarks[291], landmarks[61]] # Lip horizontal length
    ])

    # two parameters
    upper_lower_face = Parameter_By_2([
        [average_position(landmarks[162], landmarks[127]),
        average_position(landmarks[389], landmarks[356])], # [Left face end, Right face end]
        [landmarks[397], 
        landmarks[172]] # [Chin right, Chin left]
    ])

    parameters[i] = np.concatenate((vertical.ratio, horizontal.ratio, nose_to_lip.ratio, upper_lower_face.ratio))

with open(f'{section}_parameters.pkl', 'wb') as file:
    pickle.dump(parameters, file)
    
print("Parameter has successfully generated.")