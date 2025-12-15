import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="Clustering Transaksi (DBSCAN)",
    layout="wide"
)

st.title("📊 Clustering Transaksi Penjualan (DBSCAN)")
st.write(
    "Aplikasi ini mengelompokkan transaksi penjualan menggunakan "
    "**DBSCAN** berdasarkan pola jumlah, harga, dan nilai transaksi."
)

df = pd.read_csv("catatan.csv")

st.subheader("📄 Dataset Awal")
st.dataframe(df.head())

df_proc = df.copy()

# Parsing tanggal AMAN
df_proc["Tanggal"] = pd.to_datetime(
    df_proc["Tanggal"],
    dayfirst=True,
    errors="coerce"
)

# Buang tanggal invalid
df_proc = df_proc.dropna(subset=["Tanggal"])

# Encoding kolom kategorikal
for col in df_proc.select_dtypes(include="object").columns:
    df_proc[col] = LabelEncoder().fit_transform(df_proc[col])

# Cleaning akhir
df_proc = df_proc.dropna().drop_duplicates()

X = df_proc[[
    "Terjual",
    "Harga",
    "Pemasukan"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=0.4,
    min_samples=5
)

clusters = dbscan.fit_predict(X_scaled)
df_proc["Cluster"] = clusters

st.subheader("📌 Distribusi Cluster")
cluster_counts = df_proc["Cluster"].value_counts().sort_index()
st.dataframe(cluster_counts.rename("Jumlah Data"))

st.subheader("📊 Rata-rata Setiap Cluster")
cluster_mean = df_proc.groupby("Cluster")[[
    "Terjual",
    "Harga",
    "Pemasukan"
]].mean()
st.dataframe(cluster_mean)

st.subheader("🎯 Visualisasi Hasil Clustering")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=df_proc,
    x="Harga",
    y="Pemasukan",
    hue="Cluster",
    palette="tab10",
    ax=ax
)

ax.set_title("Scatter Plot Clustering DBSCAN")
ax.set_xlabel("Harga")
ax.set_ylabel("Pemasukan")

st.pyplot(fig)

st.markdown(
    """
### 📝 Keterangan:
- **DBSCAN** mengelompokkan data berdasarkan kepadatan.
- Tidak perlu menentukan jumlah cluster di awal.
- Nilai `-1` (jika ada) menunjukkan **noise / outlier**.
- Clustering dibentuk dari pola **Terjual, Harga, dan Pemasukan**.
"""
)
