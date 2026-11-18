# Diabetes Risk Prediction Using Artificial Neural Networks (ANN)

*(English description below)*

---


Bu proje, klinik sağlık verilerini analiz ederek bireylerin diyabet riskini erken evrede tahmin etmek amacıyla geliştirilmiş **Yapay Sinir Ağı (YSA)** tabanlı bir ikili sınıflandırma (binary classification) modelidir[cite: 5].

### Veri Seti ve Ön İşleme Süreci
Çalışmada literatürde yaygın olarak kullanılan ve 768 hastaya ait 8 temel klinik parametreyi (Hamilelik Sayısı, Glikoz, Kan Basıncı, Cilt Kalınlığı, İnsülin, BMI, Diyabet Soyağacı Fonksiyonu, Yaş) içeren **Pima Indians Diabetes Dataset** kullanılmıştır[cite: 5].
* **Eksik Veri Yönetimi:** Biyolojik olarak sıfır olması mümkün olmayan Glikoz, Kan Basıncı ve BMI sütunlarındaki hatalı `0` değerleri, veri bütünlüğünü korumak amacıyla ilgili sütunların ortalama (mean) değerleri ile doldurulmuştur[cite: 5].
* **Normalizasyon:** Farklı ölçeklerdeki klinik verilerin model ağırlıklarını yanıltmasını önlemek için **StandardScaler** uygulanarak veriler ortalaması 0, standart sapması 1 olacak şekilde ölçeklendirilmiştir[cite: 5].
* **Veri Bölünmesi:** Veri seti %80 Eğitim (Train) ve %20 Test olarak ayrılmıştır (`random_state=42`)[cite: 5].

### Model Mimarisi ve Eğitim Parametreleri
Model, **TensorFlow / Keras** kullanılarak `Sequential` yapıda tasarlanmıştır[cite: 5]:
* **Giriş Katmanı:** 8 klinik özellik[cite: 5]
* **1. Gizli Katman:** 12 nöron, `ReLU` aktivasyon fonksiyonu[cite: 5]
* **2. Gizli Katman:** 8 nöron, `ReLU` aktivasyon fonksiyonu[cite: 5]
* **Çıkış Katmanı:** 1 nöron, `Sigmoid` aktivasyon fonksiyonu (0-1 arası risk olasılığı)[cite: 5]
* **Derleme ve Eğitim:** Kayıp fonksiyonu olarak `Binary Cross-Entropy`, optimizasyon algoritması olarak `Adam` seçilmiş; model `100 epoch` ve `10 batch size` parametreleriyle eğitilmiştir[cite: 5].

### Performans Sonuçları
154 test verisi üzerinde yapılan değerlendirmede elde edilen metrikler[cite: 5]:
* **Genel Doğruluk (Accuracy):** %72[cite: 5]
* **Sağlıklı Sınıfı (0) F1-Score:** 0.78 (Precision: 0.78, Recall: 0.79)[cite: 5]
* **Diyabetli Sınıfı (1) F1-Score:** 0.61 (Precision: 0.61, Recall: 0.60)[cite: 5]

### Kullanılan Teknolojiler
* Python, TensorFlow, Keras[cite: 5]
* Scikit-learn, Pandas, NumPy[cite: 5]
* Matplotlib, Seaborn[cite: 5]

---

## English Description

This project is an **Artificial Neural Network (ANN)** based binary classification model developed to predict diabetes risk at an early stage by analyzing clinical health metrics[cite: 5].

### Dataset & Data Preprocessing
The model is trained on the **Pima Indians Diabetes Dataset**, consisting of 768 patient records with 8 clinical features (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age)[cite: 5].
* **Missing Value Imputation:** Biologically impossible `0` entries in Glucose, Blood Pressure, and BMI columns were treated as missing data and replaced with their respective column means[cite: 5].
* **Feature Scaling:** **StandardScaler** was applied to normalize features to zero mean and unit variance, preventing scale disparities from biasing model weights[cite: 5].
* **Train-Test Split:** The dataset was split into 80% training and 20% testing sets (`random_state=42`)[cite: 5].

### Model Architecture & Training
Built with **TensorFlow / Keras** using a `Sequential` architecture[cite: 5]:
* **Input Layer:** 8 clinical features[cite: 5]
* **Hidden Layer 1:** 12 neurons with `ReLU` activation[cite: 5]
* **Hidden Layer 2:** 8 neurons with `ReLU` activation[cite: 5]
* **Output Layer:** 1 neuron with `Sigmoid` activation (probability output between 0 and 1)[cite: 5]
* **Compilation & Training:** Compiled with `Binary Cross-Entropy` loss and `Adam` optimizer; trained over `100 epochs` with a `batch size of 10`[cite: 5].

### Evaluation Results
Evaluated on 154 unseen test samples[cite: 5]:
* **Overall Accuracy:** 72%[cite: 5]
* **Class 0 (Healthy) F1-Score:** 0.78 (Precision: 0.78, Recall: 0.79)[cite: 5]
* **Class 1 (Diabetic) F1-Score:** 0.61 (Precision: 0.61, Recall: 0.60)[cite: 5]

### Tech Stack
* Python, TensorFlow, Keras[cite: 5]
* Scikit-learn, Pandas, NumPy[cite: 5]
* Matplotlib, Seaborn[cite: 5]

---

## Developer / Geliştirici
* **Nazlıcan Bozdağ** — Software Engineer / Yazılım Mühendisi  
  *Istanbul Gelisim University / İstanbul Gelişim Üniversitesi*[cite: 5]
