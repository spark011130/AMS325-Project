from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
from Proj2_golden_ratio_distance_processing import Parameter_By_2, Parameter_Continuous, average_position

def main(model_path, img_path):
    model = load_model(model_path)
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    img = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    h, w, _ = img.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]
    
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

    parameters = np.concatenate((vertical.ratio, horizontal.ratio, nose_to_lip.ratio, upper_lower_face.ratio))
    parameters_reshaped = parameters.reshape(1, -1)
    y_pred = model.predict(parameters_reshaped).flatten()
    print(y_pred)
    
if __name__ == "__main__":
    main('AF_trained_model.keras', 'cww.png')