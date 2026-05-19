# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation System

An end-to-end E-Commerce Analytics project that performs customer segmentation using RFM analysis and KMeans clustering, builds a collaborative filtering-based product recommendation system, and predicts future customer purchases using machine learning models.

---

# 📌 Project Overview

The global e-commerce industry generates massive amounts of customer transaction data every day. Analyzing this data helps businesses:

- Understand customer purchasing behavior
- Identify high-value and at-risk customers
- Recommend relevant products
- Improve customer retention
- Optimize marketing strategies

This project focuses on solving these business problems using machine learning and data analytics techniques.

---

# 🎯 Objectives

## 1️⃣ Customer Segmentation
Segment customers based on:
- Recency
- Frequency
- Monetary value (RFM)

Using:
- KMeans Clustering

---

## 2️⃣ Product Recommendation System
Recommend similar products using:
- Item-based Collaborative Filtering
- Cosine Similarity

---

## 3️⃣ Future Purchase Prediction
Predict future customer spending using:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 4️⃣ Interactive Web Application
Deploy a user-friendly dashboard using:
- Streamlit

---

# 🧠 Machine Learning Techniques Used

| Category | Technique |
|---|---|
| Clustering | KMeans |
| Recommendation System | Collaborative Filtering |
| Similarity Metric | Cosine Similarity |
| Regression Models | Linear Regression, Decision Tree, Random Forest |
| Feature Scaling | StandardScaler |
| Evaluation Metrics | RMSE, MAE, R² Score |

---

# 📂 Dataset Information

Dataset: Online Retail Transaction Dataset

## Dataset Features

| Column | Description |
|---|---|
| InvoiceNo | Transaction ID |
| StockCode | Product Code |
| Description | Product Name |
| Quantity | Quantity Purchased |
| InvoiceDate | Transaction Date |
| UnitPrice | Price Per Unit |
| CustomerID | Unique Customer ID |
| Country | Customer Country |

---

# 🔧 Data Preprocessing

The following preprocessing steps were performed:

- Removed missing Customer IDs
- Removed cancelled invoices
- Removed negative/zero quantities
- Removed negative/zero prices
- Converted date columns to datetime format
- Created `TotalAmount` feature

```python
TotalAmount = Quantity × UnitPrice