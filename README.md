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

## Data Understanding
Dataset yang digunakan pada proyek kali ini dibuat oleh Delta_Sierra452  yang di upload ke Kaggle. Sumber dataset: [Airline Passenger Satisfaction Predictive Analysis](([https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength](https://www.kaggle.com/datasets/deltasierra452/airline-pax-satisfaction-survey))). Pada dataset ini terdiri dari 1030 baris dan 8 kolom data. Kondisi khusus dari data:
- Data tidak memiliki baris atau kolom yang nilai hilang
- Data tidak memiliki baris yang terduplikasi

### Variabel-variabel pada dataset adalah sebagai berikut:
|  Nama  | Jenis  |  Keterangan  |  Variabel  |
| --- | ----- | ------ | ------ |
|  Gender  |  Feature |  Categorical  |  Jenis Kelamin Pelanggan  |  Dependent  |   
|  Blast Furnace Slag  |  Feature  |  kg/m^3  |  Dependent  |   
|  Fly Ash  Feature  |  Continuous  |  kg/m^3  |  Dependent  |
|  Water  Feature  |  Continuous  |  kg/m^3  |  Dependent  |
|  Superplasticizer  |  Feature  |  kg/m^3  |  Dependent  |
|  Coarse Aggregate  |  Feature  |  kg/m^3  |  Dependent  |
|  Fine Aggregate  |  Feature  |  kg/m^3  |  Dependent  |
|  Age  |  Feature|  kg/m^3  |  Dependent  |
|  Concrete compressive strength  |  Target  |  MPa  |  Independent  |


Gender: Gender of the passengers (Female, Male)
Customer Type: The customer type (Loyal customer, disloyal customer)
Age: The actual age of the passengers
Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
Flight Distance: The flight distance of this journey
Inflight wifi service: Satisfaction level of the inflight wifi service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Ease of Online booking: Satisfaction level of online booking (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Gate location: Satisfaction level of Gate location (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Food and drink: Satisfaction level of Food and drink service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Online boarding: Satisfaction level of online boarding (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Seat comfort: Satisfaction level of Seat comfort (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Inflight entertainment: Satisfaction level of inflight entertainment (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
On-board service: Satisfaction level of On-board service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Leg room service: Satisfaction level of Leg room service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Baggage handling: Satisfaction level of baggage handling (1,2,3,4,5/ 1=Least Satisfied to 5=Most Satisfied)
Checkin service: Satisfaction level of Check-in service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Inflight service: Satisfaction level of inflight service (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Cleanliness: Satisfaction level of Cleanliness (0,1,2,3,4,5/ 0=Not Applicable; 1=Least Satisfied to 5=Most Satisfied)
Departure Delay in Minutes: Minutes delayed when departure
Arrival Delay in Minutes: Minutes delayed when arrival
Satisfaction: /output column/ Airline satisfaction level ('satisfied', 'neutral or dissatisfied')
### Exploratory Data Analysis
### Univariate
Menggunakan 1.5xIQR rule, ditemukan 4 variabel mengandung outlier data, diantaranya:

![Outlier data](https://github.com/arachanx21/dicoding-submission/blob/origin/Assets/outliers.png)


Dataset ini tidak memiliki fitur kategorikal.



### Multivariate

![corr](https://github.com/arachanx21/dicoding-submission/blob/53b7199d750f066949c923c13d33b879943e2e6a/Assets/confusion_matrix1.png)


Berdasarkan heatmap diatas dapat diketahui bahwa

| Kolom | Korelasi | Skor | 
| --- | ----- | ------ | 
|  Cement  |  Positif  |   0.48  |
|  Water  |  Negatif  |  -0.37  |
|  Water - Superplasticizer  |  Negatif  | -0.64  | 
|  Superplasticizer  | Positif  |  0.40  |
|  Age  |  Positif  |  0.52  |



## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:
- Menghilangkan nilai-nilai outlier yang tidak termasuk pada 1.5 x IQR Rule
- Memisahkan data menjadi dua jenis menggunakan [Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).


  | Jenis | Persentase | Jumlah Baris |
  | --- | ----- | ------ |
  | Train | 80% | 752 |
  | Test | 20% | 189 |

- Melakukan pengskalaan standar (Standard Scaler) data pada masing-masing data training dan test secara terpisah. [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

## Modeling
Setelah data siap diproses lebih lanjut, maka akan dilanjutkan pada memilih metode terbaik untuk dapat memprediksi cuaca seakurat mungkin dengan bereksperimen menggunakan 7 metode berikut ini:

| Nama | Kelebihan | Kekurangan |
| --- | ----- | ------ |
| Decision Tree | Mampu menangani hubungan non-linear antara fitur dan target | Struktur menjadi kompleks dan lambat untuk dibangun jika menangani data besar |
| Random Forest | Mengurangi varians dan meningkatkan generalisasi model dengan menggabungkan prediksi dari banyak pohon | Membutuhkan banyak memori karena menghasilkan banyak decision tree |
| Support Vector Machine | Memiliki fleksibilitas dalam menangani data kompleks karena penggunaan kernel tricks | Kurang optimal pada dataset yang tidak seimbang karena cenderung fokus pada margin yang memaksimalkan kelas mayoritas | 
| K Nearest Neighbors	 | dapat menangani data multikelas tanpa perlu modifikasi khusus | mudah terpengaruh oleh data yang noisy dan outliers yang dapat mengurangi akurasi model |
| Gradient Boosting | Menghasilkan estimasi pentingnya fitur selama masa pembelajaran | Sensitif terhadap fitur-fitur yang tidak berkorelasi. | 
| Ada Boost | Mudah beradaptasi dengan data baru dan berubah dari waktu ke waktu karena sifatnya yang iteratif | sensitif terhadap noise dan outliers karena mempengaruhi pemberian bobot yang tidak sesuai pada iterasi berikutnya| 
| Extra Trees | lebih cepat dalam pelatihan karena membagi node berdasarkan split points yang dipilih secara acak tanpa melakukan pencarian split optimal | kurang optimal pada dataset yang tidak seimbang tanpa penyesuaian tambahan | 

Tahapan yang dilakukan:
- Melakukan looping for mencari algoritma machine learning dengan parameter default yang memiliki performa paling unggul dalam memprediksi kategori di dataset ini
- Setelah menemukan algoritma machine learning paling unggul, maka algoritma machine learning tersebut akan dimasukkan ke hyperparameter tuning dengan [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) untuk mencari parameter terbaik yang meningkatkan performa model
- Selain melakukan hyperaparemeter tuning untuk mencari parameter terbaik pada algoritma machine learning, dilakukan pencarian fitur terbaik yang paling berperan pada penentuan prediksi menggunakan [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html).
- Seleksi fitur dan pencarian parameter terbaik untuk algoritma machine learning dilakukan secara bersamaan menggunakan [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

Berikut merupakan penjelasan setiap parameter yang digunakan:
1. Decision Tree
- criterion='gini', teknik pemisahan data untuk kategorisasi, 'gini' sering kali lebih cepat dalam komputasi
- max_depth=None, kedalaman maksimum pohon, karena tidak membatasi kondisi yang terbagi untuk bisa sedetail mungkin
- min_samples_split=2, jumlah minimum sampel yang diperlukan untuk membagi node internal, karena 2 mengindikasi kondisi yang detail dan tidak bisa dibagi lagi.
2. Random Forest
- n_estimators=100, jumlah pohon keputusan dalam hutan, 100 karena angka besar tapi tidak terlalu membebankan waktu komputasi
- max_depth=None, kedalaman maksimum pohon, karena tidak membatasi kondisi yang terbagi untuk bisa sedetail mungkin
- max_features='sqrt', jumlah fitur maksimum yang dipertimbangkan untuk membagi setiap node,  dengan mempertimbangkan hanya akar kuadrat dari total fitur pada setiap split, setiap pohon dalam hutan cenderung melihat subset fitur yang berbeda
3. Support Vector Machine
- C=1.0, pengontrol seberapa banyak kesalahan yang diizinkan dalam model yang mempengaruhi trade-off antara bias dan varians, memberikan keseimbangan yang wajar antara memaksimalkan margin pemisah dan meminimalkan kesalahan klasifikasi
- kernel='rbf', menentukan ruang fitur di mana data akan diproyeksikan untuk menemukan hyperplane yang memisahkan kelas, rbf memiliki mekanisme menangani data yang tidak dapat dipisahkan secara linear dengan memproyeksikannya ke ruang dimensi yang lebih tinggi
- gamma='scale', menentukan seberapa jauh pengaruh dari satu contoh pelatihan, memiliki mekanisme menyesuaikan nilai gamma berdasarkan jumlah fitur dalam data.
4. K Nearest Neighbors
- n_neighbors=5, menentukan jumlah tetangga terdekat yang akan digunakan untuk melakukan prediksi, outlier kurang bisa mempengaruhi hasil prediksi karena jika n_neighbors=1 bisa menyebabkan overfitting.
- metric='minkowski', perhitungan jarak mempengaruhi cara jarak dihitung dan bagaimana kesamaan antara titik data diukur, metrik jarak lebih sensitif terhadap perbedaan dalam skala fitur.
- weights='uniform', menentukan apakah semua tetangga memiliki bobot yang sama atau apakah tetangga yang lebih dekat memiliki pengaruh lebih besar, untuk menghindari overfitting bisa memberikan bobot yang sama terhadap semua tetangga terdekat
5. Gradient Boosting
- max_features=None, jumlah fitur maksimum yang dipertimbangkan untuk membagi setiap node,  untuk tidak membatasi fitue sebanyak-banyak dipertimbangkan dalam membagi setiap node
- learning_rate=0.1, laju pembelajaran, 0.1 agar pembelajaran semakin teliti
- loss='log_loss', mengukur seberapa baik probabilitas yang diprediksi sesuai dengan label kelas yang sebenarnya, diperuntukkan untuk klasifikasi
- max_depth=3, kedalaman maksimum dalam membagi node, karena memelurkan komputasi yang cepat dan analisis model yang general
6. Ada Boost
- learning_rate=0.1, laju pembelajaran, 0.1 agar pembelajaran semakin teliti
- n_estimators=50, jumlah pohon keputusan dalam hutan, berbeda dengan random forest memiliki kinerja lebih cepat, ada boost lebih kompleks sehingga memilih 50 untuk berada di tengah-tengah antara analisis model secara detail dan general
- estimator=None, model dasar, tidak menggunakan model dasar yang lain untuk bisa mempercepat komputasi
7. Extra Trees
- n_estimators=100, jumlah pohon keputusan dalam hutan, 100 karena angka besar tapi tidak terlalu membebankan waktu komputasi
- max_features='sqrt', jumlah fitur maksimum yang dipertimbangkan untuk membagi setiap node,  dengan mempertimbangkan hanya akar kuadrat dari total fitur pada setiap split, setiap pohon dalam hutan cenderung melihat subset fitur yang berbeda
- max_depth=None, kedalaman maksimum pohon, karena tidak membatasi kondisi yang terbagi untuk bisa sedetail mungkin

## Evaluation
Proses evaluasi model pada proyek ini menggunakan 4 metrik berikut ini

| Metrik | Pengertian | Rumus | 
| --- | ----- | ------ | 
| Akurasi | mengukur seberapa sering model prediksi benar secara keseluruhan | $`Jumlah Prediksi Benar / Jumlah Seluruh Prediksi`$ |
| Precision | mengukur seberapa tepat model dalam memprediksi kelas positif | $`True Positive / True Positive + False Positive`$  |
| Recall | mengukur seberapa baik model dalam menemukan semua contoh kelas positif | $`True Positive / True Positive + False Negative`$ |
| F1 Score | mengukur keseimbangan antara precision dan recall | $`2 * ((precision * recall)/(precision+recall))`$ |

Hasil eksperimen semua model:

![table](https://github.com/user-attachments/assets/15baf439-3002-435d-829e-f8a0872fc806)


Gradient boost mendapatkan nilai performa yang unggul dibanding dengan metode lain, sehingga untuk proses peningkatan performa menggunakan hyperparameter tuning, perlu berfokus pada Gradient boost saja. Berikut merupakan parameter-parameter yang dikombinasikan supaya mendapatkan performa terbaik. *berikut merupakan konfigurasi yang digunakan dalam hyperparameter tuning menggunakan GridSearchCV*.

| Parameter | Nilai | Modul |
| --- | ----- | ------ |
| k | 10, 11, 12, 13 | SelectKBest |
| learning_rate | 1.0, 0.1 | GradientBoostingClassifier|
| criterion | 'friedman_mse', 'squared_error' | GradientBoostingClassifier|
| max_features | 'sqrt', 'log2' | GradientBoostingClassifier|
| loss | 'log_loss', 'exponential' | GradientBoostingClassifier|
| max_depth | 3, 4 | GradientBoostingClassifier|

Setelah mengkombinasikan parameter-parameter yang ada sebanyak 288 kali, maka diperoleh parameter sebagai berikut

| Parameter | Nilai | Modul |
| --- | ----- | ------ |
| k | 12 ('Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 'Location', 'Spring', 'Summer', 'Winter'], dtype='object') | SelectKBest |
| learning_rate | 1.0, 0.1 | GradientBoostingClassifier|
| criterion | 'squared_error' | GradientBoostingClassifier|
| max_features | 'sqrt' | GradientBoostingClassifier|
| loss | 'log_loss' | GradientBoostingClassifier|
| max_depth | 3 | GradientBoostingClassifier|

Setelah menerapkan parameter-parameter tersebut dalam model Gradient Boost, maka diperoleh metrik performa sebagai berikut:

![confusionmatrix](https://github.com/user-attachments/assets/365b2438-6c75-4f2c-bc29-6839243f2423)


Pada proyek ini, metrik performa menggunakan rata-rata **micro** karena ingin mengetahui performa secara global dan general saja. Berikut merupakan perbandingan sebelum dan sesudah dilakukan hyperparameter tuning terhadap model Gradient Boost.

| Metrik | Sebelum | Skor | 
| --- | ----- | ------ | 
| Accuracy | 0.901515 | 0.902272 |
| Precision | 0.901515 | 0.902272 |
| Recall | 0.901515 | 0.902272 |
| F1 Score | 0.901515 | 0.902272 |

Proyek ini menggunakan balanced dataset sehingga metrik performa menunjukkan nilai yang sama semua. Untuk perbandingan sebelum dan sesudah hyperparameter bisa disimpulkan bahwa peningkatan performa tidak signifikan karena hanya bertambah 0,000757 saja.

**Catatan:**
- solusi sudah menjawab problem statement karena telah membuat model untuk memprediksi jenis cuaca yang akan datang berdasarkan data yang ada
- sudah mencapai goals yang diharapkan karena berhasil membangun model yang memiliki akurasi lebih dari 85%
- solusi yang direncakan berdampak pada hasil karena dapat mengetahui mana model yang paling maksimal untuk dataset ini dalam tugas prediksi kategorikal

Jika tidak tertampil gambarnya, mohon dibuka di https://github.com/oktaviacitra/dicoding-submission/blob/main/README.md


**Referensi**
[1] [Rhasyid, D. Y. L. A., Pramudita, B. A., & Istiqomah, I. (2023). Sistem Pemantauan Cuaca Berdasarkan Kecepatan Angin, Suhu dan Kelembaban Udara Berbasis Internet of Things. eProceedings of Engineering, 10(4).](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/download/20764/20289) 
**---Ini adalah bagian akhir laporan---**

