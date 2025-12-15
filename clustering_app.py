import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

features = ["Terjual", "Harga", "Pemasukan"]
X = df_proc[features]

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

df_proc["Cluster"] = kmeans.fit_predict(X_scaled)

st.subheader("📌 Distribusi Cluster")
st.dataframe(df_proc["Cluster"].value_counts().sort_index())

st.subheader("📊 Rata-rata Setiap Cluster")
st.dataframe(
    df_proc.groupby("Cluster")[features].mean()
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
ax.grid(True)
st.pyplot(fig)

st.subheader("🧮 Input Data Baru (Prediksi Cluster)")

col1, col2, col3 = st.columns(3)

with col1:
    input_terjual = st.number_input("Jumlah Terjual", min_value=0, value=10)

with col2:
    input_harga = st.number_input("Harga", min_value=0, value=5000)

with col3:
    input_pemasukan = st.number_input("Pemasukan", min_value=0, value=50000)

if st.button("Prediksi Cluster"):
    input_df = pd.DataFrame(
        [[input_terjual, input_harga, input_pemasukan]],
        columns=features
    )

    input_scaled = scaler.transform(input_df)
    predicted_cluster = kmeans.predict(input_scaled)[0]

    st.success(f"📌 Data tersebut masuk ke **Cluster {predicted_cluster}**")

st.markdown(
    """
### 📝 Keterangan:
- Model K-Means dilatih menggunakan data historis.
- Data baru yang diinput akan dipetakan ke cluster terdekat.
- Proses ini **tidak melatih ulang model**, hanya melakukan prediksi cluster.
"""
)
