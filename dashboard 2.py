import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

st.title('Proyek Analisis Data: Bike Sharing Data')
st.header('ML-67 | Ida Ayu Pradwita Nashanti')

dfhour_numeric = pd.read_csv(r'D:\101 itu nomor kami\7th on 101\Dataset Dashboard\dfhournum.csv')
filtered = pd.read_csv(r'D:\101 itu nomor kami\7th on 101\Dataset Dashboard\filtered.csv')

filtered['dteday'] = pd.to_datetime(filtered['dteday'])
min_date = filtered["dteday"].min()
max_date = filtered["dteday"].max()
with st.sidebar:
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
# Convert start_date and end_date from date_input (datetime.date) to pandas datetime (datetime64)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered = filtered[(filtered["dteday"] >= start_date) & 
                (filtered["dteday"] <= end_date)]

tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Pertanyaan 1", "Pertanyaan 2", "Analisis Lanjutan"])
with tab1:
    st.write('Data berikut merupakan dataset dari Bike Sharing Data yang telah melalui serangkaian data preprocessing.')
    st.dataframe(data=filtered)

with tab2:
    weathersit = filtered.groupby(by="weathersit").mean()[['casual','registered','cnt']]
    st.subheader('Bagaimana pengaruh kondisi lingkungan terhadap jumlah penyewa sepeda?')
    # Create subplots in separate figures and display in Streamlit columns
    plt.figure(figsize=(8, 6))
    plt.plot(weathersit.index, weathersit['cnt'], marker='o', linestyle='-', label='Line Plot')
    plt.scatter(weathersit.index, weathersit['cnt'], color='red', label='Scatter Plot')

    # Adding title and labels
    plt.title('Rata-rata Jumlah Total Penyewa terhadap Situasi Cuaca')
    plt.xlabel('Situasi Cuaca (1: Cuaca Sangat Baik, 4: Cuaca Sangat Buruk)')
    plt.ylabel('Rata-rata Jumlah Penyewa')
    plt.legend()
    st.pyplot(plt)

    with st.expander("Interpretasi Data"):
        st.write(
            """Cuaca yang buruk mengakibatkan penurunan jumlah penyewa sepeda.
        """
        )

    col1, col2 = st.columns(2)

    # Subplot 1
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))  # Create a figure for the third plot
        ax.scatter(filtered['hum'], filtered['cnt'])
        ax.set_title('Sebaran Jumlah Total Penyewa terhadap Kelembapan')
        ax.set_xlabel('Kelembapan Ternormalisasi')
        ax.set_ylabel('Jumlah Penyewa')
        st.pyplot(fig)  # Show the third plot in Streamlit

    # Subplot 2
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))  # Create a figure for the second plot
        ax.scatter(filtered['temp'], filtered['cnt'])
        ax.set_title('Sebaran Jumlah Total Penyewa terhadap Suhu')
        ax.set_xlabel('Suhu Ternormalisasi (Celcius)')
        ax.set_ylabel('Jumlah Penyewa')
        st.pyplot(fig)  # Show the second plot in Streamlit

    with st.expander("Interpretasi Data"):
        st.write(
            """- Jumlah penyewa sepeda meningkat seiring bertambahnya suhu hingga nilai suhu normalisasi pada kisaran 0.7 (sekitar 28 C), namun untuk suhu yang lebih tinggi jumlah penyewanya cenderung menurun. Hal ini menunjukkan bahwa pengguna sepeda menurun ketika cuaca sedang panas.
- Jumlah penyewa sepeda cenderung merata di rentang kelembapan normalisasi 0.2-0.8, lalu menurun di kelembapan yang lebih tinggi.
        """
        )

with tab3:
    st.subheader('Bagaimana pengaruh aspek seasonal terhadap jumlah sewa sepeda?')
    col1, col2 = st.columns(2)
    with col1:
        weekday = filtered.groupby(by="weekday").mean()[['casual','registered','cnt']]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(weekday.index,weekday['casual'], label = 'Casual')
        ax.plot(weekday.index,weekday['registered'], label = 'Registered')
        ax.set_xlabel('Weekday (0 = Sunday, 6 = Saturday)', fontweight='bold')
        ax.set_ylabel('Average Count')
        ax.set_title('Average Users by Weekday')
        ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        width = 0.25  # Width of each bar
        workingday = filtered.groupby(by="workingday").mean()[['casual','registered','cnt']]
        
        # X-axis positions
        r1 = np.arange(len(workingday))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]

        # Plotting
        ax.bar(r1, workingday['casual'], color='pink', width=width, edgecolor='grey', label='Casual')
        ax.bar(r2, workingday['registered'], color='mediumseagreen', width=width, edgecolor='grey', label='Registered')
        ax.bar(r3, workingday['cnt'], color='skyblue', width=width, edgecolor='grey', label='Total Count')

        # Add labels
        ax.set_xlabel('Day Category', fontweight='bold')
        ax.set_xticks([r + width for r in range(len(workingday))], ['Weekend/Holiday','Workingday'])
        ax.set_ylabel('Average Count')
        ax.set_title('Average Users by Workingday')
        st.pyplot(fig)
    with st.expander("Interpretasi Data"):
        st.write(
            """- Penyewa yang berasal dari kelompok "Casual" atau belum terdaftar cenderung lebih banyak menyewa sepeda di akhir pekan.
- Penyewa yang berasal dari kelompok "Registered" atau telah terdaftar cenderung lebih banyak menyewa sepeda di hari kerja. Hal ini menunjukkan bahwa pelanggan yang terdaftar umumnya menyewa sepeda untuk keperluan bekerja.
        """
        )
with tab4:
    st.subheader('Analisis Lanjutan')
    st.write('Analisis lanjutan dilakukan untuk mengetahui sebaran permintaan sewa sepeda pada 4 kategori rentang jam dalam sehari. Analisis lanjutan juga dilakukan untuk mengetahui sebaran permintaan pada 4 kategori suhu. Besaran suhu dipilih karena merupakan besaran terkait kondisi lingkungan yang paling mudah dirasakan dan memiliki nilai korelasi paling tinggi dengan jumlah penyewa, jika dibandingkan dengan sesama besaran terkait kondisi lingkungan.')
    bins = [0, 6, 12, 18, 24]  # batas rentang jam
    labels = ['0-5', '6-11', '12-17', '18-23']  # label untuk setiap rentang waktu
    filtered['range'] = pd.cut(filtered['hr'], bins = bins, labels=labels, right=False)

    # Hitung rata-rata jumlah penyewa untuk setiap rentang waktu
    user_per_range = filtered.groupby('range')['cnt'].mean()
    casual_per_range = filtered.groupby('range')['casual'].mean()
    registered_per_range = filtered.groupby('range')['registered'].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        user_per_range.plot(kind='bar', color='skyblue')
        ax.set_title('Rata-rata Jumlah Total Penyewa per Rentang Waktu')
        ax.set_xlabel('Rentang Waktu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(user_per_range)))
        ax.set_xticklabels(user_per_range.index, rotation=0)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        casual_per_range.plot(kind='bar', color='pink')
        ax.set_title('Rata-rata Jumlah Penyewa Casual per Rentang Waktu')
        ax.set_xlabel('Rentang Waktu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(casual_per_range)))
        ax.set_xticklabels(casual_per_range.index, rotation=0)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 5))
        registered_per_range.plot(kind='bar', color='mediumseagreen')
        ax.set_title('Rata-rata Jumlah Penyewa Registered per Rentang Waktu')
        ax.set_xlabel('Rentang Waktu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(registered_per_range)))
        ax.set_xticklabels(registered_per_range.index, rotation=0)
        st.pyplot(fig)
    with st.expander("Interpretasi Data"):
        st.write(
            """- Permintaan sewa sepeda tertinggi terjadi di rentang jam 12-17 baik untuk kategori penyewa Casual ataupun Registered.
        """
        )

    bins = [0, 0.34, 0.5, 0.66, 1]  # batas rentang suhu
    labels = ['0-0.34', '0.34-0.50', '0.50-0.66', '0.66-1']  # label untuk setiap rentang suhu
    filtered['range'] = pd.cut(filtered['temp'], bins = bins, labels=labels, right=False)

    # Hitung rata-rata jumlah penyewa untuk setiap rentang waktu
    user_per_range = filtered.groupby('range')['cnt'].mean()
    casual_per_range = filtered.groupby('range')['casual'].mean()
    registered_per_range = filtered.groupby('range')['registered'].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        user_per_range.plot(kind='bar', color='skyblue')
        ax.set_title('Rata-rata Jumlah Total Penyewa per Rentang Suhu')
        ax.set_xlabel('Rentang Suhu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(user_per_range)))
        ax.set_xticklabels(user_per_range.index, rotation=0)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        casual_per_range.plot(kind='bar', color='pink')
        ax.set_title('Rata-rata Jumlah Penyewa Casual per Rentang Suhu')
        ax.set_xlabel('Rentang Suhu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(casual_per_range)))
        ax.set_xticklabels(casual_per_range.index, rotation=0)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 5))
        registered_per_range.plot(kind='bar', color='mediumseagreen')
        ax.set_title('Rata-rata Jumlah Penyewa Registered per Rentang Suhu')
        ax.set_xlabel('Rentang Suhu')
        ax.set_ylabel('Rata-rata Jumlah Penyewa')
        ax.set_xticks(range(len(registered_per_range)))
        ax.set_xticklabels(registered_per_range.index, rotation=0)
        st.pyplot(fig)
    with st.expander("Interpretasi Data"):
        st.write(
            """
- Rata-rata jumlah penyewa sepeda meningkat mulai dari rentang suhu rendah hingga suhu tinggi. Rata-rata tertinggi dicapai ketika suhu normalisasi berada di rentang 0.66-1 (27-41 C).
        """
        )
    


