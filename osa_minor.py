import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from scipy.stats import mode
import matplotlib.pyplot as plt

# Mount Google Drive to access files
drive.mount('/content/drive')

# Path to your CSV file in Google Drive
file_path = '/content/drive/My Drive/subject1.csv'  # Update this path
df = pd.read_csv(file_path)

# Drop rows with missing values in 'pleth' and 'Spo2'
df = df.dropna(subset=['pleth', 'Spo2'])

# Create a synthetic label based on Spo2 levels
df['OSA_label'] = (df['Spo2'] < 95).astype(int)  # Adjust threshold for OSA detection

X = df[['pleth', 'Spo2']]
y = df['OSA_label']

# Remove noise from the dataset
X_clean = df[['pleth', 'Spo2']]

# Split the clean data into training and testing sets
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_clean = scaler.fit_transform(X_train_clean)
X_test_clean = scaler.transform(X_test_clean)

# Initialize Dummy (Baseline) Classifier
baseline_classifier = DummyClassifier(strategy='uniform', random_state=4)

# Train the Baseline classifier on clean data
baseline_classifier.fit(X_train_clean, y_train_clean)

# Make predictions with the Baseline classifier
baseline_predictions_clean = baseline_classifier.predict(X_test_clean)

# Convert clean data for LSTM (Tensor-based Classifier Model)
X_train_clean_lstm = np.expand_dims(X_train_clean, axis=2)
X_test_clean_lstm = np.expand_dims(X_test_clean, axis=2)
y_train_clean_lstm = to_categorical(y_train_clean)
y_test_clean_lstm = to_categorical(y_test_clean)

# Define and train an LSTM model
lstm_model_clean = Sequential()
lstm_model_clean.add(LSTM(64, input_shape=(X_train_clean_lstm.shape[1], X_train_clean_lstm.shape[2]), return_sequences=True))
lstm_model_clean.add(LSTM(32))
lstm_model_clean.add(Dense(2, activation='softmax'))  # Assuming binary classification

lstm_model_clean.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and save the history
history = lstm_model_clean.fit(X_train_clean_lstm, y_train_clean_lstm,
                               epochs=50, batch_size=32, verbose=1,
                               validation_data=(X_test_clean_lstm, y_test_clean_lstm))

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions with LSTM model
lstm_predictions_prob_clean = lstm_model_clean.predict(X_test_clean_lstm)
lstm_predictions_clean = np.argmax(lstm_predictions_prob_clean, axis=1)

# Combine predictions using majority voting (LSTM and Baseline)
predictions_clean = np.vstack([baseline_predictions_clean, lstm_predictions_clean])
combined_predictions_clean, _ = mode(predictions_clean, axis=0)
combined_predictions_clean = combined_predictions_clean.ravel()

# Calculate combined accuracy for clean data
combined_accuracy_clean = accuracy_score(y_test_clean, combined_predictions_clean)
combined_accuracy_percentage_clean = combined_accuracy_clean * 100

print(f'Combined Accuracy for both LSTM and Baseline Classifier: {combined_accuracy_percentage_clean:.2f}%')
print('Classification Report for Combined Predictions: ')
print(classification_report(y_test_clean, combined_predictions_clean))

# Confusion Matrix for the combined predictions
cm = confusion_matrix(y_test_clean, combined_predictions_clean)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Calculate Sleep Apnea Index (SAI)
# Assume total hours of sleep is approximated by the number of samples
total_samples = len(df)
apneas_count = df['OSA_label'].sum()  # Number of apneas (Spo2 < 95)

# Simplified SAI (Number of apneas per sample)
sai = apneas_count / total_samples

print(f'Sleep Apnea Index (SAI) (apneas per sample): {sai:.4f}')
