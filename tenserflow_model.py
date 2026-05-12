import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import tensorflow as tf

df_train = pd.read_csv("Ag_Guvenligi_Veriseti.csv", sep=';')
df_test = pd.read_csv("KDDTest+.txt", sep=',', names=df_train.columns)

df_train['attack'] = df_train['attack'].astype(str).str.replace('.', '', regex=False)
df_test['attack'] = df_test['attack'].astype(str).str.replace('.', '', regex=False)

attack_mapping = {
    'normal': 'Normal',
    'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS', 'land': 'DoS', 'back': 'DoS',
    'apache2': 'DoS', 'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'multihop': 'R2L', 'imap': 'R2L', 'warezmaster': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'httptunnel': 'R2L', 'named': 'R2L', 'sendmail': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R', 'mte': 'U2R'
}

df_train['attack'] = df_train['attack'].map(attack_mapping)
df_test['attack'] = df_test['attack'].map(attack_mapping)

top_services = df_train['service'].value_counts().nlargest(15).index
df_train['service'] = df_train['service'].apply(lambda x: x if x in top_services else 'other_service')
df_test['service'] = df_test['service'].apply(lambda x: x if x in top_services else 'other_service')

top_flags = df_train['flag'].value_counts().nlargest(4).index
df_train['flag'] = df_train['flag'].apply(lambda x: x if x in top_flags else 'other_flag')
df_test['flag'] = df_test['flag'].apply(lambda x: x if x in top_flags else 'other_flag')

df_train['is_test'] = 0
df_test['is_test'] = 1
df_combined = pd.concat([df_train, df_test], ignore_index=True)

le = LabelEncoder()
for sutun in ['protocol_type', 'service', 'flag', 'attack']:
    df_combined[sutun] = le.fit_transform(df_combined[sutun])

df_train = df_combined[df_combined['is_test'] == 0].drop('is_test', axis=1)
df_test = df_combined[df_combined['is_test'] == 1].drop('is_test', axis=1)

X_train = df_train.drop(['attack', 'level'], axis=1)
y_train = df_train['attack']
X_test = df_test.drop(['attack', 'level'], axis=1)
y_test = df_test['attack']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

X_train_smote, y_train_smote = shuffle(X_train_smote, y_train_smote, random_state=42)

girdi_sayisi = X_train_smote.shape[1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(girdi_sayisi,)))
model.add(tf.keras.layers.Dense(128, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_smote, y_train_smote,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

y_tahmin_olasilik = model.predict(X_test_scaled)
y_tahmin = np.argmax(y_tahmin_olasilik, axis=1)

basari_orani = accuracy_score(y_test, y_tahmin)
print(f"\nAccuracy: %{basari_orani * 100:.2f}")

hedef_isimleri = le.inverse_transform(range(len(le.classes_)))

print("\nClassification Report:")
print(classification_report(y_test, y_tahmin, target_names=hedef_isimleri))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_tahmin, display_labels=hedef_isimleri, cmap=plt.cm.Blues)
plt.show()

model.save("tf_ids_model.keras")