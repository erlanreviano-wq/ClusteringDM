import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Clustering Penjualan Tiket Pesawat",
    layout="wide"
)

st.title("🔵 Clustering Penjualan Tiket Pesawat (DBSCAN)")

st.markdown("""
Aplikasi ini menampilkan hasil clustering penjualan tiket pesawat
menggunakan metode **DBSCAN**.
""")
st.sidebar.header("📂 Upload Data Hasil Clustering")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil di-upload")
else:
    df = pd.read_csv("hasil_clustering_dbscan.csv")
    st.info("Menggunakan file default: hasil_clustering_dbscan.csv")
st.subheader("📊 Data Hasil Clustering")
st.dataframe(df.head())
st.subheader("📈 Distribusi Cluster")

cluster_counts = df['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)
st.subheader("🔍 Visualisasi Clustering")

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster]
    ax.scatter(
        subset['Ticket_Price'],
        subset['Total'],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

ax.set_xlabel("Ticket Price")
ax.set_ylabel("Total Transaction")
ax.set_title("Scatter Plot Hasil Clustering DBSCAN")
ax.legend()
ax.grid(True)

st.pyplot(fig)
st.subheader("📋 Rata-rata Setiap Cluster")

cluster_summary = df.groupby('Cluster')[[
    'Ticket_Quantity',
    'Ticket_Price',
    'Total'
]].mean()

st.dataframe(cluster_summary)
st.info("""
**Keterangan:**
- Setiap warna menunjukkan satu cluster.
- Cluster dibentuk berdasarkan pola transaksi (jumlah tiket, harga, dan total).
- Metode DBSCAN tidak menggunakan K-Means dan tidak memerlukan jumlah cluster di awal.
""")
