import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
features = np.random.randn(n_samples, 10)  # 10 independent features
error = 0.5 * np.random.randn(n_samples)   # Synthetic error term
dependent_feature = 2 * features[:, 0] + 1.5 * features[:, 5] - 3 * features[:, 8] + error

# Create a DataFrame
df = pd.DataFrame(data=np.column_stack([features, dependent_feature]), columns=[f'feat_{i}' for i in range(10)] + ['error'])

# Split the data into features (X) and the dependent feature (y)
X = df.drop(columns=['error']).values
y = df['error'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for training
seq_length = 10  # Adjust the sequence length as needed
X_seq = [X_scaled[i:i+seq_length] for i in range(len(X_scaled)-seq_length+1)]
y_seq = [y[i:i+seq_length] for i in range(len(y)-seq_length+1)]

# Convert sequences to numpy arrays
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the sequence-to-sequence model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 10), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))  # Output a sequence of errors
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
y_pred_seq = model.predict(X_test)

# Reshape predictions and true values
y_pred_flat = y_pred_seq.reshape(-1, seq_length)
y_test_flat = y_test.reshape(-1, seq_length)

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
print(f"Root Mean Squared Error (RMSE): {rmse}")
