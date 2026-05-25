from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def applyPCA(data, numComponents=30):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], numComponents))
    print(f"Data shape after PCA: {new_data.shape}")
    return new_data, pca

def normalizeData(data):
    shapeor = data.shape
    data = data.reshape(np.prod(data.shape[:2]), np.prod(data.shape[2:]))

    std_scaler = StandardScaler()
    std_data = std_scaler.fit_transform(data)
    data = std_data.reshape(shapeor)
    print("Applied normalization to data.")
    return data