import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import pickle

"""
Calculates the average position between two points by averaging their coordinates.

Parameters:
p1 (tuple): The first point's coordinates (x, y).
p2 (tuple): The second point's coordinates (x, y).

Returns:
list: The average of the coordinates of p1 and p2.
Example:
>>> average_position((1, 2), (3, 4))
[2.0, 3.0]
"""
def average_position(p1, p2):
    return [(a + b) / 2 for a, b in zip(p1, p2)]

"""
A class to manage parameters related to positions, distances, and ratios. Provides getter and setter methods for position, distance, and ratio.
Also includes methods to compute distances between positions and convert distances to ratios.

Parameters:
position (list): Initial position data to be managed by the Parameter class.

Attributes:
_position (list): The current position.
_distance (list or None): The current distance between positions.
_ratio (list or None): The current ratio calculated from the distances.

Example:
>>> param = Parameter([0, 1, 2])
>>> param.position
[0, 1, 2]
>>> param.distance
None
>>> param.ratio
None
"""
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
        """
        Calculates the Euclidean distance between two points.

        Parameters:
        p1 (tuple): The first point (x, y).
        p2 (tuple): The second point (x, y).

        Returns:
        float: The Euclidean distance between p1 and p2.
        """
        return ((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)**0.5

    def distances_to_ratios(self, array):
        """
        Converts an array of distances to ratios based on the maximum distance.

        Parameters:
        array (np.array): An array of distances.

        Returns:
        np.array: An array of distances scaled to ratios.
        """
        max_distance = array.max()
        ret = array / max_distance
        return ret

    def positions_to_distances(self, array):
        """
        Abstract method to be implemented by subclasses for calculating distances from positions.

        Parameters:
        array (list): List of positions.

        Returns:
        NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract class: the positions_to_distances method must be implemented in a subclass.")

"""
A subclass of Parameter that handles continuous positions. It calculates distances between each consecutive position and converts them to ratios.

Parameters:
position (list): List of positions (x, y) to be managed.

Example:
>>> param_cont = Parameter_Continuous([(1, 2), (3, 4), (5, 6)])
>>> param_cont.distance
[2.82842712 2.82842712]
>>> param_cont.ratio
[0.70710678 0.70710678]
"""
class Parameter_Continuous(Parameter):
    def __init__(self, position):
        super().__init__(position)
        self.distance = self.positions_to_distances(self._position)
        self.ratio = self.distances_to_ratios(self._distance)
    """
    Calculates the distances between each consecutive pair of positions.

    Parameters:
    array (list): A list of positions.

    Returns:
    np.array: An array of distances between consecutive positions.
    """
    def positions_to_distances(self, array):
        ret = np.zeros(len(array)-1)
        for i in range(len(array) - 1):
            ret[i] = self.distance_position(array[i], array[i + 1])
        return ret
"""
A subclass of Parameter that handles pairs of positions. It calculates the distance between each pair of positions and converts them to ratios.

Parameters:
position_pairs (list of lists): A list of position pairs to be managed.

Example:
>>> param_by2 = Parameter_By_2([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
>>> param_by2.distance
[2.82842712 2.82842712]
>>> param_by2.ratio
[0.70710678 0.70710678]
"""
class Parameter_By_2(Parameter):
    def __init__(self, position_pairs):
        super().__init__(position_pairs)
        self.distance = self.positions_to_distances(self._position)
        self.ratio = self.distances_to_ratios(self._distance)

    def positions_to_distances(self, position_pairs):
        """
        Calculates the distance between each pair of positions.

        Parameters:
        position_pairs (list of lists): A list of pairs of positions.

        Returns:
        np.array: An array of distances between the pairs of positions.
        """
        ret = np.zeros(len(position_pairs))
        for i, pair in enumerate(position_pairs):
            ret[i] = self.distance_position(pair[0], pair[1])
        return ret


'''
Loads the data, preprocesses the landmarks, and calculates the parameters for each image. 
Stores the calculated parameters into a pickle file for further use.
'''
with open("files/section.pkl", 'rb') as file: 
    section = pickle.load(file)

# Load and preprocess data
df = pd.read_csv('landmark_AMS325.csv')
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
from tqdm import tqdm
import numpy as np

for i in tqdm(range(N)):
    # Retrieve landmarks
    landmarks = df.iloc[i]['Landmarks']

    # Precompute average positions for efficiency
    left_face_end = average_position(landmarks[162], landmarks[127])
    right_face_end = average_position(landmarks[389], landmarks[356])
    nose_width = average_position(landmarks[49], landmarks[279])
    mouth_width = average_position(landmarks[61], landmarks[291])
    glabella = average_position(landmarks[10], landmarks[9])  # 미간

    # Parameter_Continuous classes
    vertical = Parameter_Continuous([
        landmarks[10],   # Forehead
        glabella,        # Glabella
        landmarks[2],    # Under the nose
        landmarks[13],   # Upper lip 
        landmarks[17],   # Lower lip
        landmarks[152]   # Chin
    ])

    horizontal = Parameter_Continuous([
        left_face_end,  # Left face end
        landmarks[33],  # Left eye left
        landmarks[133], # Left eye right
        nose_width,     # Nose width 
        landmarks[362], # Right eye left
        landmarks[263], # Right eye right
        mouth_width,    # Mouth width 
        right_face_end  # Right face end
    ])

    # Parameter_By_2 classes
    nose_to_mouth_ratio = Parameter_By_2([
        [nose_width, mouth_width]
    ])

    mid_lower_face_ratio = Parameter_By_2([
        [glabella, landmarks[13]],  # Midface height (Glabella to Upper Lip)
        [landmarks[13], landmarks[152]]  # Lower face height (Upper Lip to Chin)
    ])

    # Store concatenated results
    parameters[i] = np.concatenate((
        vertical.ratio, horizontal.ratio, nose_to_mouth_ratio.ratio, mid_lower_face_ratio.ratio
    ))

with open(f'files/{section}_parameters.pkl', 'wb') as file:
    pickle.dump(parameters, file)
    
print("Parameter has successfully generated.")
print(parameters)