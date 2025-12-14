import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="Clustering Penjualan Tiket Pesawat",
    layout="wide"
)

st.title("✈️ Clustering Penjualan Tiket Pesawat (DBSCAN)")

st.subheader("📂 Dataset")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info("Menggunakan dataset upload")
else:
    df = pd.read_csv("penjualan_tiket_pesawat.csv")
    st.info("Menggunakan dataset default")

st.dataframe(df.head())

df_clust = df.copy()
df_clust["Date"] = pd.to_datetime(df_clust["Date"])

# Hapus missing value & duplikasi
df_clust = df_clust.dropna()
df_clust = df_clust.drop_duplicates()

encoder_cols = ["City", "Gender", "Airline", "Payment_Method"]
for col in encoder_cols:
    le = LabelEncoder()
    df_clust[col] = le.fit_transform(df_clust[col])

# Feature rasio
df_clust["Total_per_Ticket"] = (
    df_clust["Total"] / df_clust["Ticket_Quantity"]
)

# Feature waktu
df_clust["Month"] = df_clust["Date"].dt.month
df_clust["DayOfWeek"] = df_clust["Date"].dt.dayofweek

X = df_clust[
    ["Ticket_Quantity", "Ticket_Price", "Total", "Total_per_Ticket"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("🔵 Clustering DBSCAN")

eps = st.slider("Nilai eps", 0.1, 1.0, 0.4, 0.05)
min_samples = st.slider("Min samples", 3, 10, 5)

dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

df_clust["Cluster"] = dbscan.fit_predict(X_scaled)

st.subheader("📊 Distribusi Cluster")

cluster_dist = (
    df_clust["Cluster"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Cluster", "Cluster": "Count"})
)

st.dataframe(cluster_dist)

st.subheader("📋 Rata-rata Setiap Cluster")

cluster_summary = (
    df_clust
    .groupby("Cluster")[["Ticket_Quantity", "Ticket_Price", "Total"]]
    .mean()
    .reset_index()
)

st.dataframe(cluster_summary)

st.subheader("📈 Visualisasi Hasil Clustering")

fig, ax = plt.subplots(figsize=(8, 6))

for c in sorted(df_clust["Cluster"].unique()):
    subset = df_clust[df_clust["Cluster"] == c]
    ax.scatter(
        subset["Ticket_Price"],
        subset["Total"],
        label=f"Cluster {c}",
        alpha=0.6
    )

ax.set_xlabel("Ticket Price")
ax.set_ylabel("Total Transaction")
ax.set_title("Scatter Plot Hasil Clustering DBSCAN")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.subheader("💾 Simpan Hasil Clustering")

if st.button("Simpan ke CSV"):
    df_clust.to_csv("hasil_clustering_dbscan.csv", index=False)
    st.success("File hasil_clustering_dbscan.csv berhasil disimpan")
st.info(
    """
    **Keterangan:**
    - Setiap cluster menunjukkan kelompok transaksi dengan pola yang mirip.
    - Clustering dilakukan menggunakan DBSCAN tanpa menentukan jumlah cluster di awal.
    - Feature engineering dan preprocessing lanjutan diterapkan sebelum clustering.
    """
)
