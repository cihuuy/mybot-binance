#requirements
1. pip install yfinance ta-lib scikit-learn binance
#process and fitur
1. Mengumpulkan Data Pasar: Data pasar untuk simbol tertentu (misalnya, BTC-USD) dikumpulkan menggunakan yfinance. Data ini mencakup harga historis dalam interval tertentu (misalnya, 1 jam selama 1 bulan).

2. Analisis Teknikal: Indikator teknikal seperti Simple Moving Average (SMA) dan Relative Strength Index (RSI) ditambahkan untuk membantu dalam analisis pasar.

3. Pelatihan Model AI: Data diolah untuk melatih model RandomForest yang memprediksi apakah harga akan naik atau turun. Model ini dilatih pada 80% data dan diuji pada 20% sisanya, dengan hasil akurasi ditampilkan.

4. Pengambilan Keputusan: Bot mengambil keputusan untuk beli atau jual berdasarkan prediksi model AI terhadap data terbaru.

5. Eksekusi Order: Bot mengeksekusi order beli atau jual melalui API Binance, tergantung pada keputusan yang diambil.

#use
1. python3 bot.py

# warning
Risiko: Trading kripto sangat berisiko, dan bot trading dapat menghasilkan kerugian.
Regulasi: Pastikan bahwa Anda mematuhi regulasi lokal mengenai trading otomatis.
Optimisasi: Sebelum digunakan dalam trading nyata, sebaiknya optimalkan model ini dengan data historis yang lebih besar dan lakukan backtesting.
