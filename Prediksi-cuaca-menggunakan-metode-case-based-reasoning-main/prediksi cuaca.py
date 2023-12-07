import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Daftar nama file dataset
dataset_files = ['dataset_diperbaiki.csv']

# Membaca dan menggabungkan dataset
dfs = []
for file in dataset_files:
    df = pd.read_csv(file)
    dfs.append(df)

data = pd.concat(dfs)

# Pemrosesan kolom waktu
data['waktu'] = data['waktu'].map({'Pagi': 0, 'Siang': 1, 'Malam': 2, 'Dini Hari': 3})

# Pemrosesan kolom wilayah
data['wilayah'] = data['wilayah'].map({'Jakarta Selatan': 0, 'Jakarta Barat': 1, 'Jakarta Utara': 2, 'Jakarta Timur': 3, 'Jakarta Pusat': 4, 'Kepulauan Seribu': 5})

# Pemrosesan kolom cuaca
le = LabelEncoder()
data['cuaca'] = le.fit_transform(data['cuaca'])

# Memisahkan fitur dan target
X = data[['waktu', 'wilayah', 'kelembaban_min', 'kelembaban_max', 'suhu_min', 'suhu_max']]
y = data['cuaca']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat objek imputer
imputer = SimpleImputer(strategy='mean')

# Mengisi nilai NaN pada data latih
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Mengisi nilai NaN pada data uji
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Menerapkan penskalaan fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

# Membuat model K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)

# Menghitung akurasi model
y_pred = knn.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') * 100
recall = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

def cbr_knn():
    waktu = waktu_entry.get()
    wilayah = wilayah_entry.get()
    kelembaban_min = kelembaban_min_entry.get()
    kelembaban_max = kelembaban_max_entry.get()
    suhu_min = suhu_min_entry.get()
    suhu_max = suhu_max_entry.get()

    # Validate input fields
    if not waktu or not wilayah or not kelembaban_min or not kelembaban_max or not suhu_min or not suhu_max:
        result_label.configure(text="Harap lengkapi semua field input.")
        return

    try:
        kelembaban_min = float(kelembaban_min)
        kelembaban_max = float(kelembaban_max)
        suhu_min = float(suhu_min)
        suhu_max = float(suhu_max)
    except ValueError:
        result_label.configure(text="Input harus berupa angka untuk kelembaban dan suhu.")
        return

    new_data = pd.DataFrame({
        'waktu': [waktu],
        'wilayah': [wilayah],
        'kelembaban_min': [kelembaban_min],
        'kelembaban_max': [kelembaban_max],
        'suhu_min': [suhu_min],
        'suhu_max': [suhu_max]
    })

    # Mengisi nilai NaN pada data baru
    new_data_imputed = pd.DataFrame(imputer.transform(new_data), columns=new_data.columns)
    
    # Melakukan penskalaan fitur pada data baru
    new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)

    # Menghitung jarak antara contoh kasus baru dengan contoh kasus pada data latih
    distances, indices = knn.kneighbors(new_data_scaled)

    # Mengambil k-neighbors terdekat
    k_neighbors = y_train.iloc[indices[0]]
    
    # Mendapatkan prediksi cuaca berdasarkan mayoritas kelas
    predicted_weather = k_neighbors.mode().values[0]
    predicted_weather_label = le.inverse_transform([predicted_weather])

    result_label.configure(text="Prediksi Cuaca: " + predicted_weather_label[0])
    accuracy_label.configure(text="Akurasi Model: " + str(round(accuracy_score(y_test, y_pred) * 100, 2)) + "%")
    precision_label.configure(text="Presisi: " + str(round(precision, 2)) + "%")
    recall_label.configure(text="Recall: " + str(round(recall, 2)) + "%")
    f1_label.configure(text="F1-Score: " + str(round(f1, 2)) + "%")

    # Mengosongkan nilai pada elemen input
    reset_input()

def reset_input():
    waktu_entry.delete(0, tk.END)
    wilayah_entry.delete(0, tk.END)
    kelembaban_min_entry.delete(0, tk.END)
    kelembaban_max_entry.delete(0, tk.END)
    suhu_min_entry.delete(0, tk.END)
    suhu_max_entry.delete(0, tk.END)

def show_weather_distribution():
    plt.figure(figsize=(8, 6))
    weather_counts = data['cuaca'].value_counts()
    weather_labels = le.inverse_transform(weather_counts.index)
    plt.bar(weather_labels, weather_counts)
    plt.xlabel('Cuaca')
    plt.ylabel('Jumlah Data')
    plt.title('Distribusi Cuaca')
    plt.show()

# Membangun GUI menggunakan Tkinter
window = tk.Tk()
window.title("Aplikasi Prediksi Cuaca")

# Mengatur ikon aplikasi
window.iconbitmap('cuaca.ico')

# Gaya untuk tampilan GUI
style = ttk.Style(window)
style.configure('TLabel', font=('Helvetica', 12))
style.configure('TButton',font=('Helvetica', 12))

# Frame Utama
main_frame = ttk.Frame(window, padding=20)
main_frame.grid(row=0, column=0, sticky='nsew')

# Frame Input
input_frame = ttk.LabelFrame(main_frame, text='Input Data')
input_frame.pack(padx=10, pady=10, fill='both')

# Label dan Entry untuk memasukkan data
waktu_label = ttk.Label(input_frame, text="Waktu (Pagi, Siang, Malam, Dini Hari):", style='TLabel')
waktu_label.grid(row=0, column=0, sticky='w')
waktu_entry = ttk.Entry(input_frame)
waktu_entry.grid(row=0, column=1)

wilayah_label = ttk.Label(input_frame, text="Wilayah (Jakarta Selatan, Jakarta Barat, Jakarta Utara, Jakarta Timur, Jakarta Pusat, Kepulauan Seribu):", style='TLabel')
wilayah_label.grid(row=1, column=0, sticky='w')
wilayah_entry = ttk.Entry(input_frame)
wilayah_entry.grid(row=1, column=1)

kelembaban_min_label = ttk.Label(input_frame, text="Kelembaban Minimum:", style='TLabel')
kelembaban_min_label.grid(row=2, column=0, sticky='w')
kelembaban_min_entry = ttk.Entry(input_frame)
kelembaban_min_entry.grid(row=2, column=1)

kelembaban_max_label = ttk.Label(input_frame, text="Kelembaban Maksimum:", style='TLabel')
kelembaban_max_label.grid(row=3, column=0, sticky='w')
kelembaban_max_entry = ttk.Entry(input_frame)
kelembaban_max_entry.grid(row=3, column=1)

suhu_min_label = ttk.Label(input_frame, text="Suhu Minimum:", style='TLabel')
suhu_min_label.grid(row=4, column=0, sticky='w')
suhu_min_entry = ttk.Entry(input_frame)
suhu_min_entry.grid(row=4, column=1)

suhu_max_label = ttk.Label(input_frame, text="Suhu Maksimum:", style='TLabel')
suhu_max_label.grid(row=5, column=0, sticky='w')
suhu_max_entry = ttk.Entry(input_frame)
suhu_max_entry.grid(row=5, column=1)

# Frame Hasil
result_frame = ttk.LabelFrame(main_frame, text='Hasil Prediksi')
result_frame.pack(padx=10, pady=10, fill='both')

result_label = ttk.Label(result_frame, text="", style='TLabel')
result_label.pack(pady=5)

accuracy_label = ttk.Label(result_frame, text="", style='TLabel')
accuracy_label.pack(pady=5)

precision_label = ttk.Label(result_frame, text="", style='TLabel')
precision_label.pack(pady=5)

recall_label = ttk.Label(result_frame, text="", style='TLabel')
recall_label.pack(pady=5)

f1_label = ttk.Label(result_frame, text="", style='TLabel')
f1_label.pack(pady=5)

# Frame Tombol
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=10)

predict_button = ttk.Button(button_frame, text="Prediksi CBR", command=cbr_knn, style='TButton')
predict_button.pack(side='left', padx=5)

show_distribution_button = ttk.Button(button_frame, text="Tampilkan Distribusi Cuaca", command=show_weather_distribution, style='TButton')
show_distribution_button.pack(side='left', padx=5)

# Mengatur tampilan Grid
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
input_frame.grid_columnconfigure(1, weight=1)

window.mainloop()

