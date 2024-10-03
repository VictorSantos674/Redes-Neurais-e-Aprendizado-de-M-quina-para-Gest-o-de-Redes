# -*- coding: utf-8 -*-
"""ProjetoRedesML.

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wKpCZk_s6qHIJJnTDhPtLyxX1OrfAog7
"""

!pip install xgboost pandas scikit-learn

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random

def generate_synthetic_data(num_packets=1000):
    """
    Generate synthetic Ethernet data with normal traffic and replay attacks.
    :param num_packets: Number of packets to generate.
    :return: A Pandas DataFrame with mock network traffic.
    """
    base_time = 1633012800
    timestamps = np.cumsum(np.random.randint(1, 10, size=num_packets)) + base_time

    src_ips = [f"192.168.0.{random.randint(1, 255)}" for _ in range(num_packets)]
    dst_ips = [f"192.168.1.{random.randint(1, 255)}" for _ in range(num_packets)]

    protocols = np.random.choice(['0x0800', '0x0806', '0x86DD'], size=num_packets)

    lengths = np.random.randint(64, 1518, size=num_packets)

    labels = np.random.choice([0, 1], size=num_packets, p=[0.9, 0.1])

    df = pd.DataFrame({
        'timestamp': timestamps,
        'src_ip': src_ips,
        'dst_ip': dst_ips,
        'protocol': protocols,
        'length': lengths,
        'label': labels
    })

    return df

df = generate_synthetic_data(1000)

df.head()

df['time_delta'] = df['timestamp'].diff().fillna(0)
df['src_dst_pair'] = df['src_ip'] + '-' + df['dst_ip']
df = pd.get_dummies(df, columns=['protocol', 'src_dst_pair'])

X = df.drop(['label', 'timestamp', 'src_ip', 'dst_ip'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))