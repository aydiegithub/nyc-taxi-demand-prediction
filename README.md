# New York Taxi Demand Prediction 🚕📈

This project builds a machine learning model to predict hourly demand for yellow taxis in New York City using historical trip data. It aims to optimize taxi fleet distribution and enhance the passenger experience.

---

## 🚀 **Objective**

- Predict taxi demand across different NYC regions at 10-minute intervals.  
- Support dispatch systems, dynamic pricing, and urban traffic management.  
- Deploy a scalable, retrainable ML model ready for production (MLOps).

---

## 📚 **Dataset**

- **Source:** NYC Yellow Taxi Trip Data  
- **Data Range:** January 2016 – March 2016  
- **Records:** 35.5 million+  
- **Features:** Pickup/dropoff time and location, trip distance, fare amount, passenger count, etc.

---

## 🛠️ **Tools & Technologies**

- **Languages:** Python  
- **Libraries:** Polars, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Version Control:** GitHub  
- **Notebook Environment:** Jupyter Notebooks

---

## 🧹 **Data Preprocessing**

- Outlier removal based on NYC bounding box (latitude and longitude).  
- Feature engineering: Trip time, trip distance, speed calculation.  
- Unix time binning into 10-minute intervals.  
- Clustering using MiniBatchKMeans on pickup locations.

---

## 🧠 **Feature Engineering**

- Lag features (previous 5 time intervals of pickups)  
- Cluster latitude, longitude, ID  
- Weekday encoding  
- Exponential moving averages  
- Timestamp-based features (10-minute bins)

---

## 📈 **Modeling Approach**

- **Input Features:** ft_1, ft_2, ft_3, ft_4, ft_5, latitude, longitude, weekday, exp_avg, cluster_id  
- **Target Variable:** Number of pickups per cluster per 10-minute interval  
- **Model:** Supervised Learning Models (RandomForestRegressor used)

---

## 📊 **Evaluation Metrics**

- Mean Absolute Percentage Error (MAPE)  
- Mean Squared Error (MSE)

---

## 🌐 **Deployment (Planned)**

- Simple **Flask** app (`app.py`) and basic frontend (`index.html`) for predictions.

---
