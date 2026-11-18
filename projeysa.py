import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Hamilelik', 'Glikoz', 'Tansiyon', 'CiltKalinligi', 'Insuilin', 'BMI', 'GenetikSkor', 'Yas', 'Sonuc']
data = pd.read_csv(url, names=columns)

cols_to_fix = ['Glikoz', 'Tansiyon', 'BMI']
for col in cols_to_fix:
    data[col] = data[col].replace(0, data[col].mean())

X = data.drop('Sonuc', axis=1)
y = data['Sonuc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(12, input_dim=8, activation='relu'), 
    Dense(8, activation='relu'),               
    Dense(1, activation='sigmoid')             
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1, verbose=0)

plt.plot(history.history['loss'])
plt.title('Model Kayıp (Loss) Grafiği')
plt.ylabel('Hata Oranı')
plt.xlabel('Epoch (Tur)')
plt.show()

predictions = (model.predict(X_test) > 0.5).astype("int32")
print("\n--- PERFORMANS SONUÇLARI ---")
print(classification_report(y_test, predictions))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
plt.show()