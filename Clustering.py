import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk preprocessing dan rekayasa fitur
def preprocess_and_engineer_features(df):
    df['price_actual'].fillna(df['price_actual'].median(), inplace=True)
    df['price_ori'].fillna(df['price_ori'].median(), inplace=True)
    df.fillna('unknown', inplace=True)
    df = df[df['price_actual'] < 1e6]

    numeric_columns = ['item_rating', 'total_rating', 'total_sold', 'favorite']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['revenue'] = df['price_actual'] * df['total_sold']
    df['discount_rate'] = (df['price_ori'] - df['price_actual']) / df['price_ori']

    features = df[['price_actual', 'item_rating', 'total_rating', 'total_sold', 'favorite', 'revenue', 'discount_rate']]
    features.fillna(features.mean(), inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return df, features_scaled

# Fungsi untuk menghapus outlier menggunakan Local Outlier Factor (LOF)
def remove_outliers_lof(df, features_scaled):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    outliers = lof.fit_predict(features_scaled)
    df_cleaned = df[outliers == 1]
    features_scaled_cleaned = features_scaled[outliers == 1]
    return df_cleaned, features_scaled_cleaned

# Fungsi untuk memberikan deskripsi dan strategi bisnis yang dinamis
def get_cluster_description_stats(df, cluster_num):
    cluster_data = df[df['cluster'] == cluster_num]
    description = ""
    strategy = ""

    if cluster_data['price_actual'].mean() < df['price_actual'].mean():
        description += "Pelanggan dengan harga pembelian rendah."
        strategy += "Tingkatkan loyalitas dengan program loyalitas dan diskon khusus."
    else:
        description += "Pelanggan dengan harga pembelian tinggi."
        strategy += "Fokus pada peningkatan nilai produk dan pengalaman pelanggan."

    if cluster_data['item_rating'].mean() > df['item_rating'].mean():
        description += " Rating tinggi."
        strategy += " Gunakan ulasan positif untuk pemasaran dan tingkatkan interaksi."
    else:
        description += " Rating rendah."
        strategy += " Perbaiki kualitas produk dan layanan untuk meningkatkan kepuasan pelanggan."

    if cluster_data['total_sold'].mean() > df['total_sold'].mean():
        description += " Penjualan tinggi."
        strategy += " Luncurkan kampanye pemasaran untuk menarik pelanggan baru."
    else:
        description += " Penjualan rendah."
        strategy += " Fokus pada retensi pelanggan dan analisis umpan balik."

    return description, strategy

# Fungsi untuk mencari nilai k optimal menggunakan elbow method
def calculate_optimal_k(features_scaled):
    sse = []
    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        sse.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, sse, marker='o')
    ax.set_title('Elbow Method untuk Menentukan Nilai Optimal k')
    ax.set_xlabel('Jumlah Kluster (k)')
    ax.set_ylabel('Sum of Squared Errors (SSE)')
    st.pyplot(fig)
    
    # Menentukan nilai optimal k berdasarkan elbow method
    optimal_k = k_values[sse.index(min(sse, key=lambda x: abs(x - (sse[-1] + sse[0])/2)))]
    return optimal_k

def main():
    st.set_page_config(page_title="Segmentasi Pelanggan", page_icon=":bar_chart:", layout="wide")
    st.image("logo.png", width=160)
    st.title('Segmentasi Pelanggan menggunakan K-Means Clustering')

    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Unggah CSV", "Hasil Clustering"])

    if page == "Beranda":
        st.header("Selamat Datang di Alat Segmentasi Pelanggan")
        st.write("""
            Alat ini dirancang untuk membantu Anda melakukan segmentasi pelanggan menggunakan metode K-Means Clustering. 
            Dengan menggunakan alat ini, Anda dapat:
            - Mengunggah file CSV yang berisi data pelanggan.
            - Melihat hasil clustering yang memisahkan pelanggan Anda ke dalam beberapa segmen berdasarkan fitur-fitur tertentu.
            - Mendapatkan wawasan tentang karakteristik setiap segmen dan strategi bisnis yang disarankan untuk setiap segmen tersebut.
        """)

    if page == "Unggah CSV":
        st.header("Unggah File CSV")
        st.write("Unggah file CSV Anda untuk mulai melakukan segmentasi pelanggan.")
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required_features = ['price_actual', 'price_ori', 'item_rating', 'total_rating', 'total_sold', 'favorite']
            missing_features = [feature for feature in required_features if feature not in df.columns]

            if missing_features:
                st.warning(f"File CSV yang diunggah harus memiliki kolom berikut: {', '.join(missing_features)}")
            else:
                st.subheader("Pratinjau Data")
                st.write("5 baris terakhir dari file yang diunggah:")
                st.write(df.tail())

                df, features_scaled = preprocess_and_engineer_features(df)
                st.session_state.data = df
                st.session_state.features_scaled = features_scaled

                min_data_points = 50
                max_data_points = 50000

                if df.shape[0] < min_data_points:
                    st.warning(f"Jumlah data minimal adalah {min_data_points}. Data yang diunggah hanya memiliki {df.shape[0]} data.")
                elif df.shape[0] > max_data_points:
                    st.warning(f"Jumlah data maksimal adalah {max_data_points}. Data yang diunggah memiliki {df.shape[0]} data. Pertimbangkan untuk mengurangi ukuran dataset.")
                else:
                    st.success('File berhasil diunggah dan diproses!')

    elif page == "Hasil Clustering":
        st.header("Hasil Clustering")
        st.write("Lihat hasil segmentasi pelanggan dan strategi bisnis berdasarkan klustering.")

        if 'data' in st.session_state and 'features_scaled' in st.session_state:
            df = st.session_state.data
            features_scaled = st.session_state.features_scaled

            df_cleaned_lof, features_scaled_cleaned_lof = remove_outliers_lof(df, features_scaled)
            st.write(f"Jumlah data setelah menghapus outlier (LOF): {df_cleaned_lof.shape[0]}")

            if df_cleaned_lof.shape[0] > 0:
                # Menentukan nilai optimal k menggunakan elbow method
                st.sidebar.subheader("Konfigurasi Klustering")
                st.sidebar.write("Menentukan nilai optimal k menggunakan elbow method")
                optimal_k = calculate_optimal_k(features_scaled_cleaned_lof)
                st.sidebar.write(f"Nilai optimal k berdasarkan elbow method: {optimal_k}")

                # Pilihan untuk memilih nilai k
                k = st.sidebar.number_input("Jumlah Kluster (k)", min_value=2, max_value=10, value=optimal_k, step=1)

                kmeans = KMeans(n_clusters=k, random_state=42)
                df_cleaned_lof['cluster'] = kmeans.fit_predict(features_scaled_cleaned_lof)

                st.subheader('Visualisasi Klustering')
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='price_actual', y='total_sold', hue='cluster', data=df_cleaned_lof, palette='viridis', ax=ax)
                ax.set_title(f'Segmentasi Pelanggan dengan {k} Kluster')
                ax.set_xlabel('Harga Aktual')
                ax.set_ylabel('Total Terjual')
                st.pyplot(fig)

                st.subheader('Statistik Kluster')
                numeric_columns = df_cleaned_lof.select_dtypes(include=[np.number]).columns
                cluster_stats = df_cleaned_lof.groupby('cluster')[numeric_columns].mean()
                cluster_counts = df_cleaned_lof['cluster'].value_counts().sort_index()
                cluster_stats['count'] = cluster_counts
                st.dataframe(cluster_stats.style.background_gradient(cmap='viridis'))

                st.subheader('Deskripsi dan Strategi Bisnis untuk Setiap Kluster')
                for cluster in range(k):
                    description, strategy = get_cluster_description_stats(df_cleaned_lof, cluster)
                    st.markdown(f"**Kluster {cluster}**: {description}")
                    st.markdown(f"**Strategi Bisnis**: {strategy}")

            else:
                st.error('Tidak cukup data setelah menghapus outlier untuk melakukan clustering.')

        else:
            st.warning('Unggah file CSV dan lakukan preprocessing untuk melihat hasil clustering.')

if __name__ == '__main__':
    main()