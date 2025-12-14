import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Clustering Penjualan Tiket Pesawat (DBSCAN)",
    layout="wide"
)

st.title("✈️ Clustering Penjualan Tiket Pesawat (DBSCAN)")
st.info("Aplikasi ini menampilkan hasil clustering DBSCAN yang telah diproses sebelumnya.")

st.subheader("📂 Dataset Hasil Clustering")

df = pd.read_csv("hasil_clustering_dbscan.csv")
st.dataframe(df.head())

st.subheader("📊 Distribusi Cluster")

cluster_dist = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
    .rename(columns={"index": "Cluster", "Cluster": "Jumlah Data"})
)

st.dataframe(cluster_dist)

st.subheader("📋 Rata-rata Setiap Cluster")

cluster_mean = (
    df
    .groupby("Cluster")[["Ticket_Quantity", "Ticket_Price", "Total"]]
    .mean()
    .reset_index()
)

st.dataframe(cluster_mean)

st.subheader("📈 Visualisasi Scatter Plot")

plt.figure(figsize=(8, 6))

for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    plt.scatter(
        subset["Ticket_Price"],
        subset["Total"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

plt.xlabel("Ticket Price")
plt.ylabel("Total Transaction")
plt.title("Scatter Plot Hasil Clustering DBSCAN")
plt.legend()
plt.grid(True)

st.pyplot(plt)

st.subheader("📝 Keterangan")
st.markdown("""
- Clustering dilakukan menggunakan **DBSCAN** pada tahap preprocessing (Google Colab).
- File `hasil_clustering_dbscan.csv` merupakan **output final clustering**.
- Aplikasi Streamlit digunakan untuk **visualisasi dan analisis hasil cluster**.
- Setiap cluster menunjukkan pola transaksi berbeda berdasarkan jumlah tiket, harga, dan total.
""")
