import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the encrypted data and known decryption keys
encrypted_data = np.load('encrypted_data.npy')
decryption_keys = np.load('decryption_keys.npy')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encrypted_data, decryption_keys, test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model performance
score = model.score(X_test, y_test)
print('Model score: ', score)

# Use the model to predict a decryption key for new encrypted data
new_encrypted_data = np.load('new_encrypted_data.npy')
predicted_key = model.predict(new_encrypted_data)
print('Predicted decryption key: ', predicted_key)