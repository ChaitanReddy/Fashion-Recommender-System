import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
from matplotlib import pyplot as plt

# Load features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Load and preprocess query image
img = image.load_img('sample/jersey'
'.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract features
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Find nearest neighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

# Display results using matplotlib
for i, file_idx in enumerate(indices[0][1:6], start=1):
    temp_img = cv2.imread(filenames[file_idx])
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)  # convert for matplotlib
    plt.subplot(1, 5, i)
    plt.imshow(cv2.resize(temp_img, (224,224)))
    plt.axis('off')

plt.show()
