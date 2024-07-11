# Submission 1: Plant Growth Classification

Nama: Tasya Putri Aliya

Username dicoding: tasyaputrialiya

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Plan Growth Data Classification](https://www.kaggle.com/datasets/gorororororo23/plant-growth-data-classification) |
| Masalah | Pertumbuhan tanaman merupakan hal yang tidak dapat diprediksi jumlah tetapnya karena sama seperti manusia yang tumbuh kembangnya dipengaruhi oleh banyak faktor seperti makanan, gen, dan aktivitas fisik, tanaman juga bertumbuh sesuai dengan beberapa faktor. Namun, faktor apa saja yang akan mempengaruhi pertumbuhan tanaman masih abu-abu dan titik apa yang membuat faktor tersebut akan dikatakan baik. Oleh karena itu, pengklasifikasian pertumbuhan tanaman dalam beberapa faktor perlu untuk dilakukan. Faktor yang akan digunakan, antara lain Fertilizer_Type, Humidity, Soil_Type, Sunlight_Hours, Temperature, dan Water_Frequency. |
| Solusi machine learning | Untuk menyelesaikan masalah tersebut maka dibuat model klasifikasi untuk menentukan apakah pertumbuhan tanaman baik pada beberapa kondisi tertentu untuk melihat kondisi apa yang paling sesuai |
| Metode pengolahan | Proyek ini menggunakan beberapa metode yaitu melakukan transformasi data dengan mengubah seluruh isi data menjadi lowercase, memastikan tipe data sesuai, dan melakukan pembagian data menjadi 80% training data dan 20% testing data. |
| Arsitektur model | Model menggunakan arsitektur model berupa 1 layer Concatenate, dan 2 layer Dense dengan unit 64 dan 32 dengan activation relu, serta satu layer output dengan Dense 1 karena menghasilkan binary dengan activation sigmoid. |
| Metrik evaluasi | Metrik evaluasi yang digunakan pada proyek ini adalah ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, serta BinaryAccuracy.  |
| Performa model | Berdasarkan nilai metrik yang diperoleh dari Evaluator, didapatkan AUC secara keseluruhan sebesar 47,4%, dengan Binary Accuracy mencapai 48,8%. Untuk False Negatives, False Positives, True Negatives, dan True Positives, terdapat masing-masing 12, 9, 10, dan 10 dari total 41 Example Count. Pada pengujian menggunakan tf-serving, model kurang mampu untuk menghasilkan model yang optimal, hal ini kemungkinan disebabkan oleh jumlah data yang sangat sedikit sehingga model tidak dapat berlatih dengan baik.|
