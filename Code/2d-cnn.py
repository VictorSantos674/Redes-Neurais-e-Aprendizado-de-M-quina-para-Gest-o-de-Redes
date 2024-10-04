import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

columns = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]

indoor_original = pd.DataFrame([
    [0.000000, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [0.129900112, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [0.329914825, "::", "ff02::1:ff00:a", "ICMPv6", 82, "Neighbor Solicitation for fe80::2fc:70ff:fe00:a"],
    [0.418011871, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.063318243, "IntrepidCont_00:00:0a", "LLDP_Multicast", "PTPv2", 72, "Peer_Delay_Req Message"],
    [1.08806199, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.24820258, "::", "ff02::1:ff00:7", "ICMPv6", 82, "Neighbor Solicitation for fe80::2fc:70ff:fe00:7"],
    [1.330881177, "fe80::2fc:70ff:fe00:a", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.330938086, "fe80::2fc:70ff:fe00:a", "ff02::2", "ICMPv6", 74, "Router Solicitation from 00:fc:70:00:00:0a"],
    [1.334335352, "IntrepidCont_00:00:0a", "IEEE1722aWor_01:00:00", "IEEE1722-1", 86, "AVDECC Discovery Protocol"],
    [1.349976373, "fe80::2fc:70ff:fe00:a", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.430086231, "IntrepidCont_00:00:0a", "LLDP_Multicast", "MRP-MSRP", 64, "Multiple Stream Reservation Protocol"]
], columns=columns)

indoor_injection = pd.DataFrame([
    [0.000000, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [0.129900112, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [0.329914825, "::", "ff02::1:ff00:a", "ICMPv6", 82, "Neighbor Solicitation for fe80::2fc:70ff:fe00:a"],
    [0.418011871, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.063318243, "IntrepidCont_00:00:0a", "LLDP_Multicast", "PTPv2", 72, "Peer_Delay_Req Message"],
    [1.08806199, "::", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.24820258, "::", "ff02::1:ff00:7", "ICMPv6", 82, "Neighbor Solicitation for fe80::2fc:70ff:fe00:7"],
    [1.330881177, "fe80::2fc:70ff:fe00:a", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.330938086, "fe80::2fc:70ff:fe00:a", "ff02::2", "ICMPv6", 74, "Router Solicitation from 00:fc:70:00:00:0a"],
    [1.334335352, "IntrepidCont_00:00:0a", "IEEE1722aWor_01:00:00", "IEEE1722-1", 86, "AVDECC Discovery Protocol"],
    [1.349976373, "fe80::2fc:70ff:fe00:a", "ff02::16", "ICMPv6", 94, "Multicast Listener Report Message v2"],
    [1.430086231, "IntrepidCont_00:00:0a", "LLDP_Multicast", "MRP-MSRP", 64, "Multiple Stream Reservation Protocol"],
    [0.00017617, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
[0.000179643, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
[0.000183013, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
[0.000186398, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
[0.000189837, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"]

], columns=columns)

driving_original = pd.DataFrame([
    [0, "IntrepidCont_00:00:02", "LLDP_Multicast", "PTPv2", 64, "Sync Message"],
    [0.000248, "IntrepidCont_00:00:02", "LLDP_Multicast", "PTPv2", 94, "Follow_Up Message"],
    [0.005792, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "Audio Video Transport Protocol  [MP2T fragment of a reassembled packet]"],
    [0.019286, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019323, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019415, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019536, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019665, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019787, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.019912, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.020037, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.020162, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"]

], columns=columns)

driving_injection = pd.DataFrame([
    [0, "IntrepidCont_00:00:02", "LLDP_Multicast", "PTPv2", 64, "Sync Message"],
    [0.000056797, "IntrepidCont_00:00:02", "LLDP_Multicast", "PTPv2", 94, "Follow_Up Message"],
    [0.000095978, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "Audio Video Transport Protocol  [MP2T fragment of a reassembled packet]"],
    [0.000131925, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000137307, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000141105, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000144551, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000148026, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.00015145, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000154836, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000162291, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.000166016, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet]"],
    [0.00017617, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
    [0.000179643, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
    [0.000183013, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
    [0.000186398, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"],
    [0.000189837, "IntrepidCont_00:00:02", "91:ef:00:00:fe:00", "MPEG TS", 438, "[MP2T fragment of a reassembled packet"]
], columns=columns)

X_train = pd.concat([indoor_original, indoor_injection], ignore_index=True) #concatenar dados (tabelas)
X_test = pd.concat([driving_original, driving_injection], ignore_index=True)

y_train = np.array([0] * len(indoor_original) + [1] * len(indoor_injection))  # definindo 0 para original 1 para injetado
y_test = np.array([0] * len(driving_original) + [1] * len(driving_injection))

benign_count_train = len(indoor_original) #contando qtd de pacotes malignos e benignos para melhor entendimento
malicious_count_train = len(indoor_injection)
benign_count_test = len(driving_original)
malicious_count_test = len(driving_injection)

print(f"Treinamento: {benign_count_train} pacotes benignos, {malicious_count_train} pacotes maliciosos")
print(f"Teste: {benign_count_test} pacotes benignos, {malicious_count_test} pacotes maliciosos")

# pre processamento de dados
X_train['Protocol'] = LabelEncoder().fit_transform(X_train['Protocol'])
X_train['Source'] = LabelEncoder().fit_transform(X_train['Source'])
X_train['Destination'] = LabelEncoder().fit_transform(X_train['Destination'])

X_test['Protocol'] = LabelEncoder().fit_transform(X_test['Protocol'])
X_test['Source'] = LabelEncoder().fit_transform(X_test['Source'])
X_test['Destination'] = LabelEncoder().fit_transform(X_test['Destination'])

X_train['Info'] = LabelEncoder().fit_transform(X_train['Info']) #transforma string em inteiro
X_test['Info'] = LabelEncoder().fit_transform(X_test['Info'])

X_train = X_train[["Time", "Source", "Destination", "Protocol", "Length", "Info"]].values #colunas relevantes para analise p teste e treino
X_test = X_test[["Time", "Source", "Destination", "Protocol", "Length", "Info"]].values

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  #2d
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# rede convolucional 2d
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2) #treino

y_pred = model.predict(X_test) #avaliacao
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# metricas
print("Relatório de Classificação:")
print(classification_report(y_test_classes, y_pred_classes))



