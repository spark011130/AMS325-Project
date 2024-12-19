import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime
import pickle

# Data Preparation
with open("/Users/andypark/Desktop/2024 FALL/AMS 325/Project/section.pkl", 'rb') as file: section = pickle.load(file)
parameter_path = f"/Users/andypark/Desktop/2024 FALL/AMS 325/Project/{section}_parameters.pkl"
ratings_path = f"/Users/andypark/Desktop/2024 FALL/AMS 325/Project/{section}_ratings.pkl"

with open(parameter_path, 'rb') as file: parameters = pickle.load(file)
with open(ratings_path, 'rb') as file: ratings = pickle.load(file)
X = np.array(parameters)
y = np.array(ratings)

# Data Split: Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# DNN Model Building 13 - 64 - 32 - 1
model = Sequential([
    Dense(64, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# TensorBoard call back
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)


# Train
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback],
    verbose=1
)

# Evaluation
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# Prediction
y_pred = model.predict(X_test).flatten()

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(f'{section}_training_history.png')
plt.close()

# Actual vs Prediction y Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([1,5], [1,5], 'r--')  # Most Romantic Prediction Line
plt.title('Actual vs Predicted')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.savefig(f'{section}_actual_vs_predicted.png')
plt.close()

# Model Save
model.save(f'{section}_trained_model.keras')
print("model has successfully saved.")

# tensorboard --logdir=logs/fit