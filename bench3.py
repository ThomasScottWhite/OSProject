import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import os, psutil
process = psutil.Process()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

for _ in range(10):
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Benchmark the training time
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    end_time = time.time()

    # Display results
    print(end_time - start_time)

    # Note: You may need to adjust the model architecture, hyperparameters, and dataset based on your specific use case.
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
