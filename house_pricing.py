import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction with Clustering")
st.write("This app uses **KMeans clustering + Random Forest regression**")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():    

return pd.read_csv("housing.csv")


df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.sample(5))

st.subheader("❓ Missing Values")
st.write(df.isnull().sum())

# -----------------------------
# KMeans Clustering
# -----------------------------
st.subheader("🌍 Location-based Clustering")

df = df[['longitude', 'latitude']]

ss = StandardScaler()
scale = ss.fit_transform(df)

fig1, ax1 = plt.subplots()
ax1.scatter(scale[:, 0], scale[:, 1], s=5)
ax1.set_title("Before Clustering")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
st.pyplot(fig1)

# Elbow method
wcss_error = []
for k in range(1, 10):
    model = KMeans(n_clusters=k)
    model.fit(scale)
    wcss_error.append(model.inertia_)

fig2, ax2 = plt.subplots()
ax2.plot(range(1, 10), wcss_error)
ax2.set_title("Elbow Method")
ax2.set_xlabel("Number of clusters")
ax2.set_ylabel("WCSS ERROR")
st.pyplot(fig2)

# Train final KMeans
model = KMeans(n_clusters=4)
pred = model.fit_predict(scale)

df["clusters"] = pred 

fig3, ax3 = plt.subplots()
for c in df["clusters"].unique():
    temp = df[df["clusters"] == c]
    ax3.scatter(temp["longitude"], temp["latitude"], label=f"Cluster {c}", s=5)

ax3.set_title("After Clustering")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.legend()
st.pyplot(fig3)

# -----------------------------
# Prepare data for ML
# -----------------------------
st.subheader("🤖 Model Training")

df1 = pd.read_csv("housing.csv")

# Strip spaces first to be safe
df1.columns = df1.columns.str.strip()

df1['total_bedrooms'] = df1['total_bedrooms'].fillna(df1['total_bedrooms'].median())

df1['clusters']=pred

X=df1[['housing_median_age','total_rooms','total_bedrooms','population','households','median_income','clusters']]

y=df1['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ Model trained successfully")
st.metric("R² Score", round(r2, 3))

# -----------------------------
# User Prediction
# -----------------------------
st.subheader("🏡 Predict House Price")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Housing Median Age", 0, 100, 34)
    rooms = st.number_input("Total Rooms", 0, 10000, 3000)
    bedrooms = st.number_input("Total Bedrooms", 0, 5000, 600)
    population = st.number_input("Population", 0, 10000, 1400)

with col2:
    households = st.number_input("Households", 0, 5000, 500)
    income = st.number_input("Median Income", 0.0, 15.0, 2.575)
    cluster = st.selectbox("Cluster", [0, 1, 2, 3])

if st.button("🔮 Predict Price"):
    input_data = [[age, rooms, bedrooms, population, households, income, cluster]]
    input_scaled = ss.transform(input_data)
    prediction = rf.predict(input_scaled)

    st.success(f"💰 Predicted House Value: **${prediction[0]:,.2f}**")

