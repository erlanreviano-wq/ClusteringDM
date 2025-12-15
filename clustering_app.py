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

st.title("🔵 Clustering Transaksi Menggunakan DBSCAN")
st.write(
    "Aplikasi ini melakukan **clustering transaksi** menggunakan metode "
    "**DBSCAN** berdasarkan pola penjualan dan keuangan."
)

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file catatan.csv",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Menggunakan dataset upload")
else:
    df = pd.read_csv("catatan.csv")
    st.sidebar.info("Menggunakan dataset default")

st.subheader("📄 Dataset Awal")
st.dataframe(df.head())

df_proc = df.copy()

# Tanggal → fitur waktu
df_proc["Tanggal"] = pd.to_datetime(df_proc["Tanggal"])
df_proc["Year"] = df_proc["Tanggal"].dt.year
df_proc["Month"] = df_proc["Tanggal"].dt.month
df_proc["Day"] = df_proc["Tanggal"].dt.day
df_proc.drop(columns=["Tanggal"], inplace=True)

# Encoding kategorikal
for col in df_proc.select_dtypes(include="object").columns:
    df_proc[col] = LabelEncoder().fit_transform(df_proc[col])

# Cleaning
df_proc = df_proc.dropna().drop_duplicates()

features = [
    "Terjual",
    "Harga",
    "Modal Satuan",
    "Pemasukan",
    "Pengeluaran"
]

X = df_proc[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

clusters = dbscan.fit_predict(X_scaled)
df_proc["Cluster"] = clusters

st.subheader("📊 Distribusi Cluster")

cluster_counts = df_proc["Cluster"].value_counts().sort_index()
st.dataframe(cluster_counts.rename("Jumlah Data"))

fig, ax = plt.subplots()
cluster_counts.plot(kind="bar", ax=ax)
ax.set_title("Jumlah Data per Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Jumlah Transaksi")

st.pyplot(fig)

st.subheader("📋 Rata-rata Setiap Cluster")

cluster_summary = df_proc.groupby("Cluster")[features].mean()
st.dataframe(cluster_summary)

st.subheader("🔍 Visualisasi Cluster (Scatter Plot)")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df_proc,
    x="Harga",
    y="Pemasukan",
    hue="Cluster",
    palette="tab10",
    ax=ax
)

ax.set_title("Clustering berdasarkan Harga dan Pemasukan")
st.pyplot(fig)

st.markdown(
    """
### 📝 Keterangan:
- Setiap warna menunjukkan **satu cluster**.
- Clustering dibentuk berdasarkan **pola transaksi dan keuangan**.
- **DBSCAN tidak memerlukan jumlah cluster di awal** dan tidak menggunakan K-Means.
"""
)
