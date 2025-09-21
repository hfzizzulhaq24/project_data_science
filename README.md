# Customer Hotel Segmentation
This Streamlit app segments customers into different groups using:
- **KMeans Clustering**
- **Principal Component Analysis (PCA)** for visualization
- **Scaler** (StandardScaler or similar)

## 📂 Folders

- `Model/`: Contains `scaler.pkl`, `kmeans_final.pkl`, and optional `pca_model.pkl`
- `Data/`: Contains default sample data (`segment.csv`)
- `app.py`: Streamlit app

## 🚀 Features

- Upload your own CSV
- Use provided `.pkl` models
- Visualize clustering results with PCA 2D plot
- Auto-interpret cluster characteristics
- Add business strategy for each segment
- Download result
