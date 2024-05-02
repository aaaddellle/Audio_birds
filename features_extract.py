import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import Model
from classifier import model, feature_extraction_model

print("Libraries downloaded")

data_path = "C:\\Users\\Lenovo\\Desktop\\python\\asimplest\\mels_1"
img_size = (64, 64)

def load_data(data_path):
    images = []
    labels = []
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img = Image.open(os.path.join(folder_path, filename)).convert('L')
                img = img.resize(img_size)
                img = np.array(img)
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(folder_name)
    return np.array(images), np.array(labels)

# Load all the data
x_data, y_data = load_data(data_path)
print("Data loading done")


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y_data.reshape(-1, 1))

print("Model training starts")

history = model.fit(x_data, y_encoded, epochs=30, batch_size=40)

print("Model evaluation on the same dataset (optional)")
test_loss, test_acc = model.evaluate(x_data, y_encoded)
print("Test accuracy:", test_acc)

#print("Predictions start")
#predictions = model.predict(x_data)
#predicted_classes = np.argmax(predictions, axis=1)

print("Feature extraction start")
# Feature extraction on all data
extracted_features = feature_extraction_model.predict(x_data)

print("======> Extracted Features:")
print(extracted_features)
print(extracted_features.shape)

import pandas as pd

reshaped_features = extracted_features.reshape(extracted_features.shape[0], -1)
features_df = pd.DataFrame(reshaped_features)

file_path = "C:\\Users\\Lenovo\\Desktop\\python\\AtoI\\features_all.json"
features_df.to_json(file_path, orient='records')
print("extracted features saved to:", file_path)

