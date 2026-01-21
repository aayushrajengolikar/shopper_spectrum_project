import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/online_retail_cleaned.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 🔥 FEATURE ENGINEERING
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df

df = load_data()

# -----------------------------
# RFM FEATURE ENGINEERING
# -----------------------------
@st.cache_data
def compute_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm

rfm = compute_rfm(df)

# -----------------------------
# SCALING & CLUSTERING
# -----------------------------
@st.cache_data
def train_kmeans(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm, scaler, kmeans

rfm, scaler, kmeans = train_kmeans(rfm)

# -----------------------------
# CLUSTER LABELING
# -----------------------------
cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

def label_cluster(row):
    if row["Recency"] < cluster_profile["Recency"].median() and \
       row["Frequency"] > cluster_profile["Frequency"].median() and \
       row["Monetary"] > cluster_profile["Monetary"].median():
        return "High-Value"
    elif row["Frequency"] >= cluster_profile["Frequency"].median():
        return "Regular"
    elif row["Recency"] > cluster_profile["Recency"].median():
        return "At-Risk"
    else:
        return "Occasional"

cluster_profile["Segment"] = cluster_profile.apply(label_cluster, axis=1)
cluster_map = cluster_profile["Segment"].to_dict()
rfm["Segment"] = rfm["Cluster"].map(cluster_map)

# -----------------------------
# PRODUCT SIMILARITY
# -----------------------------
@st.cache_data
def build_similarity(df):
    pivot = df.pivot_table(
        index="Description",
        columns="CustomerID",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )

    similarity = cosine_similarity(pivot)
    return similarity, pivot.index

similarity_matrix, product_names = build_similarity(df)

# -----------------------------
# STREAMLIT UI
# -----------------------------
tab1, tab2 = st.tabs(["🎯 Product Recommendation", "👥 Customer Segmentation"])

# -----------------------------
# PRODUCT RECOMMENDATION
# -----------------------------
with tab1:
    st.header("🔍 Product Recommendation")

    product_input = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        if product_input not in product_names:
            st.error("Product not found. Please enter a valid product name.")
        else:
            idx = list(product_names).index(product_input)
            scores = list(enumerate(similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

            st.success("Top 5 Similar Products:")
            for i, _ in scores:
                st.write(f"• {product_names[i]}")

# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------
with tab2:
    st.header("👤 Customer Segmentation")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency", min_value=0)
    monetary = st.number_input("Monetary Value", min_value=0.0)

    if st.button("Predict Cluster"):
        user_data = np.array([[recency, frequency, monetary]])
        user_scaled = scaler.transform(user_data)
        cluster = kmeans.predict(user_scaled)[0]
        segment = cluster_map[cluster]

        st.success(f"Customer Segment: **{segment}**")
