import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Clustering K-Means",
    layout="wide"
)

st.title("📊 Clustering Transaksi Penjualan (K-Means)")

df = pd.read_csv("catatan.csv")

st.subheader("📄 Dataset Awal")
st.dataframe(df.head())

df_proc = df.copy()

df_proc["Tanggal"] = pd.to_datetime(
    df_proc["Tanggal"],
    dayfirst=True,
    errors="coerce"
)

df_proc = df_proc.dropna(subset=["Tanggal"])

for col in df_proc.select_dtypes(include="object").columns:
    df_proc[col] = LabelEncoder().fit_transform(df_proc[col])

df_proc = df_proc.dropna().drop_duplicates()

X = df_proc[["Terjual", "Harga", "Pemasukan"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("⚙️ Parameter K-Means")
k = st.sidebar.slider(
    "Jumlah Cluster (k)",
    min_value=2,
    max_value=6,
    value=3
)

kmeans = KMeans(
    n_clusters=k,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
df_proc["Cluster"] = clusters

st.subheader("📌 Distribusi Cluster")
st.dataframe(df_proc["Cluster"].value_counts().sort_index())

st.subheader("📊 Rata-rata Setiap Cluster")
st.dataframe(
    df_proc.groupby("Cluster")[["Terjual", "Harga", "Pemasukan"]].mean()
)

st.subheader("🎯 Visualisasi Hasil Clustering")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df_proc,
    x="Harga",
    y="Pemasukan",
    hue="Cluster",
    palette="tab10",
    ax=ax
)

ax.set_title("Scatter Plot Clustering K-Means")
ax.set_xlabel("Harga")
ax.set_ylabel("Pemasukan")
ax.grid(True)

st.pyplot(fig)

st.markdown(
    """
### 📝 Keterangan:
- K-Means mengelompokkan data berdasarkan **jarak ke centroid**.
- Jumlah cluster ditentukan di awal (`k`).
- Setiap cluster menunjukkan pola transaksi yang berbeda.
"""
)
