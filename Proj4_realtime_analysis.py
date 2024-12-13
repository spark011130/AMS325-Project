import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

model = load_model('trained_model.h5')

for layer in model.layers:
    weights = layer.get_weights()  # weights[0]: 가중치, weights[1]: 편향
    if weights:
        print(f'레이어: {layer.name}, 가중치 형태: {weights[0].shape}')
        
        
# 특정 레이어의 가중치 추출
layer_name = 'dense'  # 예: 'dense', 'dense_1' 등
layer = model.get_layer(name=layer_name)
weights, biases = layer.get_weights()

# 히스토그램으로 가중치 분포 시각화
plt.figure(figsize=(8, 4))
plt.hist(weights.flatten(), bins=30, color='blue', alpha=0.7)
plt.title(f'{layer_name} 레이어 가중치 분포')
plt.xlabel('가중치 값')
plt.ylabel('빈도')
plt.savefig(f'{layer_name}_weights_histogram.png')
plt.close()

# 히트맵으로 가중치 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(weights, cmap='viridis')
plt.title(f'{layer_name} 레이어 가중치 히트맵')
plt.xlabel('입력 노드')
plt.ylabel('출력 노드')
plt.savefig(f'{layer_name}_weights_heatmap.png')
plt.close()