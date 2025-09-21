# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="KMeans + PCA Deployment", layout="wide")
sns.set(style="whitegrid")

st.title("📊 Customer Segmentation")

# -------------------------
# Helpers & cached loaders
# -------------------------
@st.cache_data
def load_model(path):
    return joblib.load(path)

def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error baca CSV: {e}")
        return None

def generate_interpretation(cluster_profile):
    """
    Buat ringkasan sederhana: untuk setiap cluster sebutkan 2 fitur
    tertinggi dan 2 fitur terendah dibanding rata-rata keseluruhan.
    """
    overall = cluster_profile.mean()
    interpretations = {}
    for cid, row in cluster_profile.iterrows():
        diff = row - overall
        top_pos = diff.sort_values(ascending=False).head(2).index.tolist()
        top_neg = diff.sort_values(ascending=True).head(2).index.tolist()
        interpretations[cid] = {
            "strong_points": top_pos,
            "weak_points": top_neg,
            "desc": f"Cluster {cid} memiliki nilai relatif tinggi pada {', '.join(top_pos)} "
                    f"dan relatif rendah pada {', '.join(top_neg)}."
        }
    return interpretations



# -------------------------
# Sidebar: Load models & data paths
# -------------------------
st.sidebar.header("Model & Data Paths")
scaler_path = st.sidebar.text_input("Path Scaler .pkl", value="Model/scaler.pkl")
kmeans_path = st.sidebar.text_input("Path KMeans .pkl", value="Model/kmeans_final.pkl")
pca_path = st.sidebar.text_input("Path PCA .pkl ", value="Model/pca_model.pkl")
default_csv = st.sidebar.text_input("Path CSV (Optional) ", value="Data/segment.csv")

load_models_btn = st.sidebar.button("Load Models")

scaler_model = None
kmeans_model = None
pca_model = None

if load_models_btn:
    # Load scaler
    try:
        scaler_model = load_model(scaler_path)
        st.sidebar.success(f"Scaler loaded: {scaler_path}")
    except Exception as e:
        st.sidebar.error(f"Gagal load Scaler: {e}")

    # Load KMeans
    try:
        kmeans_model = load_model(kmeans_path)
        st.sidebar.success(f"KMeans loaded: {kmeans_path}")
    except Exception as e:
        st.sidebar.error(f"Gagal load KMeans: {e}")

    # Load PCA (optional)
    try:
        pca_model = load_model(pca_path)
        st.sidebar.success(f"PCA loaded: {pca_path}")
    except Exception:
        pca_model = None
        st.sidebar.info("PCA not loaded / optional.")

# Silent load models if not loaded yet (for convenience)
if scaler_model is None:
    try:
        scaler_model = load_model("Model/scaler.pkl")
    except Exception:
        scaler_model = None

if kmeans_model is None:
    try:
        kmeans_model = load_model("Model/kmeans_final.pkl")
    except Exception:
        kmeans_model = None

if pca_model is None:
    try:
        pca_model = load_model("Model/pca_model.pkl")
    except Exception:
        pca_model = None




# -------------------------
# Sidebar: Input data upload
# -------------------------
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV untuk diprediksi", type=["csv"])
use_default = False
if uploaded_file is None and default_csv:
    try:
        _ = open(default_csv, "r")
        use_default = st.sidebar.checkbox("Gunakan default CSV", value=False)
    except Exception:
        use_default = False

if uploaded_file is not None:
    df = safe_read_csv(uploaded_file)
elif use_default:
    try:
        df = pd.read_csv(default_csv)
        st.sidebar.info(f"Memakai Data Default: {default_csv}")
    except Exception as e:
        st.error(f"Gagal memuat default CSV: {e}")
        df = None
else:
    df = None

if df is None:
    st.info("Silakan load model (sidebar) dan upload CSV atau pilih default CSV.")
    st.stop()

# Preview data
st.subheader("🗂️ The Data")
st.dataframe(df.head())




# -------------------------
# Sidebar: Feature selection
# -------------------------
st.sidebar.header("Feature Selection")

all_cols = df.columns.tolist()
selected_features = st.sidebar.multiselect(
    "Pilih fitur yang dipakai untuk transform (urut wajib sama seperti training)",
    options=all_cols,
    default=all_cols
)

if len(selected_features) == 0:
    st.error("Pilih minimal 1 fitur.")
    st.stop()

X_raw = df[selected_features].copy()

# Check non-numeric
non_numeric = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    st.warning(f"Ada kolom non-numeric pada fitur terpilih: {non_numeric}. Silakan ubah/encode atau pilih fitur numeric saja.")
    st.stop()




# -------------------------
# Data transform & predict
# -------------------------
try:
    # Apply scaler
    if scaler_model is not None:
        X_scaled = scaler_model.transform(X_raw)
    else:
        X_scaled = X_raw.values  # fallback raw values

    # Apply PCA if available
    if pca_model is not None:
        X_transformed = pca_model.transform(X_scaled)
        # Prepare 2D projection for plotting
        if X_transformed.shape[1] >= 2:
            pca_for_plot = X_transformed[:, :2]
        else:
            pca_for_plot = np.hstack([X_transformed, np.zeros((X_transformed.shape[0], 1))])
    else:
        X_transformed = X_scaled
        # Create PCA 2D for plotting only
        from sklearn.decomposition import PCA as _PCA
        tmp_pca = _PCA(n_components=2)
        pca_for_plot = tmp_pca.fit_transform(X_scaled)
except Exception as e:
    st.error(f"Gagal transform data: {e}")
    st.stop()

try:
    # Predict clusters
    clusters = kmeans_model.predict(X_transformed)
except Exception as e:
    st.error(f"Gagal predict: {e}")
    st.stop()




# -------------------------
# Prepare result dataframe
# -------------------------
df_result = df.copy()
df_result["Cluster"] = clusters

df_pca_plot = pd.DataFrame(pca_for_plot, columns=["PC1", "PC2"])
df_pca_plot["Cluster"] = clusters




# -------------------------
# Sidebar: Cluster labeling (optional)
# -------------------------
st.sidebar.header("👨‍👩‍👧‍👦 Labeling Cluster")
n_clusters = getattr(kmeans_model, "n_clusters", len(np.unique(clusters)))
default_labels = {i: f"Segment {i}" for i in range(n_clusters)}

cluster_labels = {}
for i in range(n_clusters):
    cluster_labels[i] = st.sidebar.text_input(f"Label cluster {i}", value=default_labels.get(i, f"Cluster {i}"))

df_result["Cluster_Label"] = df_result["Cluster"].map(cluster_labels)



# -------------------------
# Display clustering results
# -------------------------
st.subheader("✅ Hasil Clustering")
st.dataframe(df_result.head())

# Cluster profile (mean values)
st.subheader("📝 Profil Rata-rata per Cluster")
numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
# Remove cluster and PCA cols for profiling
for col_to_remove in ["Cluster", "PC1", "PC2"]:
    if col_to_remove in numeric_cols:
        numeric_cols.remove(col_to_remove)

cluster_profile = df_result.groupby("Cluster")[numeric_cols].mean().round(3)
st.dataframe(cluster_profile)

# Auto-interpretation
st.subheader("🔍 Auto-Interpretasi Sederhana Per Cluster")
interpretations = generate_interpretation(cluster_profile)
for cid, info in interpretations.items():
    label = cluster_labels.get(cid, f"Cluster {cid}")
    st.markdown(f"**{label} (Cluster {cid})** — {info['desc']}")

# Sidebar: Business strategy per cluster label
st.sidebar.header("🎯Strategi Bisnis per Label (Berdasarkan interpretasi)")
strategy_map = {}
for i in range(n_clusters):
    label = cluster_labels[i]
    default_strategy = ""
    strategy_map[label] = st.sidebar.text_area(f"Strategi untuk '{label}'", value=default_strategy, height=80)

st.subheader("🎯 Strategi Bisnis")
for label, strat in strategy_map.items():
    if strat.strip():
        st.markdown(f"**{label}**: {strat}")
    else:
        st.markdown(f"**{label}**: _Belum disetel. Tambahkan strategi di sidebar._")




# -------------------------
# Visualisasi PCA 2D
# -------------------------
st.subheader("Visualisasi PCA 2D (Cluster)")
fig1, ax1 = plt.subplots(figsize=(8, 5))
palette = sns.color_palette(n_colors=n_clusters)
sns.scatterplot(data=df_pca_plot, x="PC1", y="PC2", hue="Cluster", palette=palette, ax=ax1, legend="full")
ax1.set_title("PCA projection colored by KMeans cluster")
ax1.legend(title="Cluster", loc="best")
st.pyplot(fig1)




# -------------------------
# Visualisasi Per-feature comparison (bar chart means)
# -------------------------
st.subheader("📌Perbandingan Rata-rata Fitur Antara Cluster")
features_to_compare = st.multiselect(
    "Pilih fitur dengan skala yang sama (mean)", options=numeric_cols, default=numeric_cols[:4]
)

if features_to_compare:
    profile_for_plot = cluster_profile[features_to_compare].T  # features x clusters
    fig2, ax2 = plt.subplots(figsize=(10, 4 + max(0, len(features_to_compare)//3)))
    profile_for_plot.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Mean Value")
    ax2.set_xlabel("Features")
    ax2.set_title("Mean per Feature by Cluster")
    ax2.legend([f"{cluster_labels.get(int(x), x)} (#{int(x)})" for x in profile_for_plot.columns], title="Cluster")
    st.pyplot(fig2)




# -------------------------
# Visualisasi Boxplot per feature
# -------------------------
st.subheader("Distribusi (Boxplot) Per Fitur per Cluster")
box_feature = st.selectbox("Pilih fitur untuk boxplot", options=numeric_cols, index=0)
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_result, x="Cluster_Label", y=box_feature, ax=ax3)
ax3.set_title(f"Distribusi {box_feature} per Cluster")
ax3.set_xlabel("Cluster")
st.pyplot(fig3)




# -------------------------
# Inspect & filter data by segment
# -------------------------
st.subheader("Inspect & Filter Data")
segment_options = ["All"] + list(dict.fromkeys(df_result["Cluster_Label"].tolist()))
selected_label = st.selectbox("Pilih segment untuk tampilkan detail", options=segment_options)

if selected_label == "All":
    st.dataframe(df_result.head(200))
else:
    st.dataframe(df_result[df_result["Cluster_Label"] == selected_label].head(200))



# -------------------------
# Download results
# -------------------------
st.subheader("Download hasil clustering")
csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV hasil clustering", data=csv, file_name="clustered_result.csv", mime="text/csv")

st.markdown("---")
st.caption("Catatan: Pastikan fitur yang kamu pilih untuk prediksi sama persis (kolom & urutan) seperti saat training model. Jika pipeline training menggunakan scaling / encoding / PCA, pastikan object yang sama disimpan dan diload di aplikasi ini.")
