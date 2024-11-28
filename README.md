# Laporan Proyek Machine Learning - Ahmad Chandra

## Domain Proyek
Kepuasan pelanggan dalam layanan jasa sangatlah penting dalam keberlangsungan perusahaan. Kemampuan untuk dapat memprediksi kepuasan pelanggan dapat membantu perusahaan meningkatkan layanannya kepada para pelanggan. Selain itu, hal ini juga dapat membantu perusahaan mengambil keputusan berdasarkan data dalam memilih aspek yang perlu ditingkatkan terlebih dahulu berdasarkan urgensi dan pemahaman industri perusahaan. Kemampuan memprediksi ini sekarang memungkinkan karena berkembangnya machine-learning.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, rincian masalahnya adalah 
- Bagaimana mengolah data agar bisa masuk ke pemodelan prediksi agar dapat meningkatkan kepuasan pelanggan?
- Bagaimana membuat sistem prediksi kepuasan pelanggan maskapai yang dengan peforma seakurat mungkin (minimal akurasi 85%) agar sedikit kesalahan prediksi yang mengakibatkan biaya antisipasi banyak terbuang sia-sia ?

### Goals

Untuk menangani rincian masalah di atas, maka tujuan yang saya ajukan adalah membuat sistem prediksi kekuatan kompresif beton dengan akurasi minimal 85% sehingga mengerti bagaimana antisipasi yang dilakukan agar tidak menimbulkan masalah lingkungan yang lebih besar.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
- Membandingkan 3 algoritma sekaligus dalam bentuk tabel yang berisi metrik evaluasi.
- Melakukan hyperparameter tuning terhadap 1 algoritma yang memiliki nilai paling unggul di metrik evaluasi.

## Batasan Problem
Lebih dari 80% pelanggan merupakan pelanggan berumur 20-60. Sehingga, pemodelan digunakan difokuskan pada pelanggan dengan rentang umur 20-60.

## Data Understanding
Dataset yang digunakan pada proyek kali ini dibuat oleh Delta_Sierra452  yang di upload ke Kaggle. Sumber dataset: [Airline Passenger Satisfaction Predictive Analysis](https://www.kaggle.com/datasets/deltasierra452/airline-pax-satisfaction-survey). Pada dataset ini terdiri dari 103904 baris dan 25 kolom data. Kondisi khusus dari data:
- Data memiliki baris atau kolom yang nilai hilang (dilakukan pembersihan data terlebih dahulu)
- Data tidak memiliki baris yang terduplikasi
- Data SR dan id adalah data pribadi yang berhubungan dengan pelanggan dan tidak akan mempengaruhi pemodelan, sehingga kedua variabel tersebut tidak digunakan.

### Variabel-variabel pada dataset adalah sebagai berikut:
|  Nama  | Jenis  |  Keterangan  |  Variabel  |
| --- | ----- | ------ | ------ |
|  Gender  |  Kategorikal  | Gender of the passengers (Female, Male)  |  Dependent  |
|  Customer Type  |  Kategorikal  | The customer type (Loyal customer, disloyal customer)  |  Dependent  |
|  Age  |  Continous  | The actual age of the passengers  |  Dependent  |
|  Type of Travel  |  Kategorikal  | Purpose of the flight of the passengers (Personal Travel, Business Travel)  |  Dependent  |
|  Class  |  Kategorikal  | Travel class in the plane of the passengers (Business, Eco, Eco Plus)  |  Dependent  |
|  Flight Distance  |  Continous  | The flight distance of this journey  |  Dependent  |
|  Inflight wifi service  |  Kategorikal  | Satisfaction level of the inflight wifi service Scale(0-5)  |  Dependent  |
|  Departure/Arrival time convenient  |  Kategorikal  | Satisfaction level of Departure/Arrival time convenient Scale(0-5)  |  Dependent  |
|  Ease of Online booking  |  Kategorikal  | Satisfaction level of online booking Scale(0-5)  |  Dependent  |
|  Gate location  |  Kategorikal  | Satisfaction level of Gate location Scale(0-5)  |  Dependent  |
|  Food and drink  |  Kategorikal  | Satisfaction level of Food and drink service Scale(0-5)  |  Dependent  |
|  Online boarding  |  Kategorikal  | Satisfaction level of online boarding Scale(0-5)  |  Dependent  |
|  Seat comfort  |  Kategorikal  | Satisfaction level of Seat comfort Scale(0-5)  |  Dependent  |
|  Inflight entertainment  |  Kategorikal  | Satisfaction level of inflight entertainment Scale(0-5)  |  Dependent  |
|  On-board service  |  Kategorikal  | Satisfaction level of On-board service Scale(0-5)  |  Dependent  |
|  Leg room service  |  Kategorikal  | Satisfaction level of Leg room service Scale(0-5)  |  Dependent  |
|  Baggage handling  |  Kategorikal  | Satisfaction level of baggage handling (1,2,3,4,5/ 1=Least Satisfied to 5=Most Satisfied)  |  Dependent  |
|  Checkin service  |  Kategorikal  | Satisfaction level of Check-in service Scale(0-5)  |  Dependent  |
|  Inflight service  |  Kategorikal  | Satisfaction level of inflight service Scale(0-5)  |  Dependent  |
|  Cleanliness  |  Kategorikal  | Satisfaction level of Cleanliness Scale(0-5)  |  Dependent  |
|  Departure Delay in Minutes  |  Continous  | Minutes delayed when departure  |  Dependent  |
|  Arrival Delay in Minutes  |  Continous  | Minutes delayed when arrival  |  Dependent  |
|  Satisfaction  |  Kategorikal  | Airline satisfaction level ('satisfied', 'neutral or dissatisfied')  | Independent  |

Keterangan: 
Scale(0-5): (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)

### Exploratory Data Analysis
### Univariate
Menggunakan 1.5xIQR rule, ditemukan 3 variabel numerikal mengandung outl
ier data, diantaranya:
![image](https://github.com/user-attachments/assets/eba71836-c4dc-42d0-84ac-5c2d0aee806b)



### Multivariate

Multivariative analysis tidak dilakukan karena hubungan data kontinu dengan variabel target kategorikal memiliki hubungan yang ambigu.


## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:

- Menghilangkan nilai-nilai outlier yang tidak termasuk pada 1.5 x IQR Rule

Variabel Satisfaction merupakan target dalam proyek ini. Hasil penghilangan pencilan data dari data numerik menggunakan 1.5xIQR rule menghasilkan data yang relatif berimbang. Hal ini dapat mengurangi overfitting dalam pemodelan.

![image](https://github.com/user-attachments/assets/0ecc17b8-d821-48d7-a068-23eaad6cfc04)

-  Melakukan encoding pada variabel-variabel categorical
  
  |  Kolom  |  Encoding  |  Alasan  |
  | ------- | ------ | ------ |
  |  'Gender'  | One-hot Encoding  | nilai menunjukkan tidak ada urutan khusus  |
  |  'Type_of_Travel'  | One-hot Encoding  | nilai menunjukkan tidak ada urutan khusus  |
  |  'Class'  | One-hot Encoding  | nilai menunjukkan tidak ada urutan khusus  |
  |  'Customer_Type'  | One-hot Encoding  | nilai menunjukkan tidak ada urutan khusus  |
  |  'Satisfaction'  | Label Encoding  | nilai perlu menunjukkan puas/tidaknya pelanggan  |

- Memisahkan data menjadi dua jenis menggunakan [Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).


  | Jenis | Persentase | Jumlah Baris |
  | --- | ----- | ------ |
  | Train | 80% | 57600 |
  | Test | 20% | 14401 |

- Melakukan pengskalaan standar (Standard Scaler) data pada masing-masing data training dan test secara terpisah. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)


<img width="727" alt="StandardScaling" src="https://github.com/user-attachments/assets/d12795c3-39b7-48bc-b343-55d9b1c6943a">

## Modeling
Setelah data siap diproses lebih lanjut, maka akan dilanjutkan pada memilih metode terbaik untuk dapat memprediksi cuaca seakurat mungkin dengan bereksperimen menggunakan 7 metode berikut ini:

| Nama | Kelebihan | Kekurangan |
| --- | ----- | ------ |
| K Nearest Neighbors	 | sederhana, adaptif, fleksibel dengan metrik yang bervariasi | sensitif dengan jumlah n tetangga dan fitur yang irelevan, bias terhadap kelas mayoritas, dan sulit dengan dimensi yang tinggi  |
| Random Forest | Akurat, model toleran dengan overfitting dan outliers, dan dapat menampung data yang besar  | Membutuhkan banyak memori karena menghasilkan banyak decision tree |
| Ada Boost | sederhana, adaptif, toleran terhadap overfitting  | sensitif terhadap noise dan outliers karena mempengaruhi pemberian bobot yang tidak sesuai pada iterasi berikutnya| 


Tahapan yang dilakukan:
- Melakukan looping for mencari algoritma machine learning dengan parameter default yang memiliki performa paling unggul dalam memprediksi kategori di dataset ini
- Setelah menemukan algoritma machine learning paling unggul, maka algoritma machine learning tersebut akan dimasukkan ke hyperparameter tuning dengan [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) untuk mencari parameter terbaik yang meningkatkan performa model


Berikut merupakan penjelasan setiap parameter yang digunakan:
1. K Nearest Neighbors
- n_neighbors=15, menentukan jumlah tetangga terdekat yang akan digunakan untuk melakukan prediksi, outlier kurang bisa mempengaruhi hasil prediksi karena jika n_neighbors=1 bisa menyebabkan overfitting.
- metric='minkowski', perhitungan jarak mempengaruhi cara jarak dihitung dan bagaimana kesamaan antara titik data diukur, metrik jarak lebih sensitif terhadap perbedaan dalam skala fitur.
- weights='uniform', menentukan apakah semua tetangga memiliki bobot yang sama atau apakah tetangga yang lebih dekat memiliki pengaruh lebih besar, untuk menghindari overfitting bisa memberikan bobot yang sama terhadap semua tetangga terdekat

2. Random Forest
- n_estimators=100, jumlah pohon keputusan dalam hutan, 100 karena angka besar tapi tidak terlalu membebankan waktu komputasi
- max_depth=None, kedalaman maksimum pohon, karena tidak membatasi kondisi yang terbagi untuk bisa sedetail mungkin
- max_features='sqrt', jumlah fitur maksimum yang dipertimbangkan untuk membagi setiap node,  dengan mempertimbangkan hanya akar kuadrat dari total fitur pada setiap split, setiap pohon dalam hutan cenderung melihat subset fitur yang berbeda
- criterion="gini", fungsi ini mengukur kualitas dari split.  “gini” menggunakan Gini impurity sedangkan “log_loss” dan “entropy” untuk shannon information gain.

3. Ada Boost
- learning_rate=0.1, laju pembelajaran, 0.1 agar pembelajaran semakin teliti
- n_estimators=50, jumlah pohon keputusan dalam hutan, berbeda dengan random forest memiliki kinerja lebih cepat, ada boost lebih kompleks sehingga memilih 50 untuk berada di tengah-tengah antara analisis model secara detail dan general
- estimator=None, model dasar, tidak menggunakan model dasar yang lain untuk bisa mempercepat komputasi

## Evaluation
Proses evaluasi model pada proyek ini menggunakan 4 metrik berikut ini

| Metrik | Pengertian | Rumus | 
| --- | ----- | ------ | 
| Akurasi | mengukur seberapa sering model prediksi benar secara keseluruhan | $`Jumlah Prediksi Benar / Jumlah Seluruh Prediksi`$ |
| Precision | mengukur seberapa tepat model dalam memprediksi kelas positif | $`True Positive / True Positive + False Positive`$  |
| Recall | mengukur seberapa baik model dalam menemukan semua contoh kelas positif | $`True Positive / True Positive + False Negative`$ |
| F1 Score | mengukur keseimbangan antara precision dan recall | $`2 * ((precision * recall)/(precision+recall))`$ |

Hasil eksperimen semua model:
<img width="602" alt="Screenshot 2024-11-28 at 17 16 36" src="https://github.com/user-attachments/assets/506fe1b2-d866-45ef-80a3-cc4cdd7c14f1">




Algoritma Random Forest mendapatkan nilai performa yang unggul dibanding dengan metode lain, sehingga untuk proses peningkatan performa menggunakan hyperparameter tuning, perlu berfokus pada algoritma Random Forest saja. Berikut merupakan parameter-parameter yang dikombinasikan supaya mendapatkan performa terbaik. *berikut merupakan konfigurasi yang digunakan dalam hyperparameter tuning menggunakan GridSearchCV*.

| Parameter | Nilai | Modul |
| --- | ----- | ------ |
| n_estimators  |  50,100,200, 500  |  RandomForestClassifier  |
| max_features  |  'sqrt', 'log2'  |  RandomForestClassifier  |
| max_depth  |   None  |  RandomForestClassifier  |
| criterion  |  'gini', 'entropy'  |  RandomForestClassifier  |

Setelah mengkombinasikan parameter-parameter yang ada sebanyak 80 kali, maka diperoleh parameter sebagai berikut

| Parameter | Nilai | Modul |
| --- | ----- | ------ |
| n_estimators  |   200  |  RandomForestClassifier  |
| max_features  |  'sqrt',  |  RandomForestClassifier  |
| max_depth  |   None  |  RandomForestClassifier  |
| criterion  |  'entropy'  |  RandomForestClassifier  |

Setelah menerapkan parameter-parameter tersebut dalam model Random Forest, maka diperoleh metrik performa sebagai berikut:

![image](https://github.com/user-attachments/assets/d1736191-bdc8-420a-8608-6251e0f2556f)



Berikut merupakan perbandingan sebelum dan sesudah dilakukan hyperparameter tuning terhadap model Gradient Boost.

| Metrik | Sebelum | Skor | 
| --- | ----- | ------ | 
| Accuracy | 0.960905 | 0.961530 |
| Precision |0.969561 | 0.971367 |
| Recall | 0.950233 | 0.949668 |
| F1 Score | 0.959800 | 0.960395 |

Proyek ini menggunakan balanced dataset sehingga metrik performa menunjukkan nilai yang sama semua. Untuk perbandingan sebelum dan sesudah hyperparameter bisa disimpulkan bahwa peningkatan performa tidak signifikan karena nilai peningkatannya sangat kecil.

**Catatan:**
- solusi sudah menjawab problem statement karena telah membuat model untuk memprediksi jenis cuaca yang akan datang berdasarkan data yang ada
- sudah mencapai goals yang diharapkan karena berhasil membangun model yang memiliki akurasi lebih dari 85%
- solusi yang direncakan berdampak pada hasil karena dapat mengetahui mana model yang paling maksimal untuk dataset ini dalam tugas prediksi kategorikal

