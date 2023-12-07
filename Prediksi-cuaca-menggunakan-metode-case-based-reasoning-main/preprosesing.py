import pandas as pd

# Membaca dataset dari file CSV dengan encoding 'latin1'
data = pd.read_csv('cuaca1.csv', encoding='latin1')

# Memisahkan kolom "kelembaban_persen" menjadi dua kolom "kelembaban_min" dan "kelembaban_max"
data[['kelembaban_min', 'kelembaban_max']] = data['kelembaban_persen'].str.split(' - ', expand=True)

# Menghapus spasi pada kolom "kelembaban_min" dan "kelembaban_max"
data['kelembaban_min'] = data['kelembaban_min'].str.strip()
data['kelembaban_max'] = data['kelembaban_max'].str.strip()

# Mengubah tipe data kolom "kelembaban_min" dan "kelembaban_max" menjadi numerik
data['kelembaban_min'] = pd.to_numeric(data['kelembaban_min'])
data['kelembaban_max'] = pd.to_numeric(data['kelembaban_max'])

# Memisahkan kolom "suhu_derajat_celcius" menjadi dua kolom "suhu_min" dan "suhu_max"
data[['suhu_min', 'suhu_max']] = data['suhu_derajat_celcius'].str.split(' - ', expand=True)

# Menghapus spasi pada kolom "suhu_min" dan "suhu_max"
data['suhu_min'] = data['suhu_min'].str.strip()
data['suhu_max'] = data['suhu_max'].str.strip()

# Mengubah tipe data kolom "suhu_min" dan "suhu_max" menjadi numerik
data['suhu_min'] = pd.to_numeric(data['suhu_min'])
data['suhu_max'] = pd.to_numeric(data['suhu_max'])

# Menyimpan dataset yang telah diperbaiki ke file CSV baru
data.to_csv('dataset_cuaca_diperbaiki.csv', index=False)
