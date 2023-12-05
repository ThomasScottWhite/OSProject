import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os, psutil
process = psutil.Process()

for _ in range(10):
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    lr_classifier = LogisticRegression(random_state=42)
    lr_classifier.fit(X_train, y_train)
    lr_predictions = lr_classifier.predict(X_test)
    end_time = time.time()

    # Display results
    print(end_time - start_time)
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

