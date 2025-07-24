# Flight Price Prediction with Deep Learning

This project explores the dataset extracted from "Ease My Trip" website, and is used to predict flight ticket prices using real-world data. It compares **two versions of a neural network model**: one basic, and one enhanced with smart, domain-inspired feature engineering as well as answering the proposed questions from Kaggle. 

Tech Stack

| Component               | Tool/Library          | Description                                |
| ----------------------- | --------------------- | ------------------------------------------ |
| Programming Language | Python 3.x            | Core language for all processing and ML    |
| Data Handling        | Pandas, NumPy         | Dataframes, numerical arrays               |
| Visualization        | Matplotlib, Seaborn   | Plots, KDEs, error/residual analysis       |
| Deep Learning        | TensorFlow / Keras    | Model building, training, and evaluation   |
| Model Evaluation     | Scikit-learn          | Metrics: MAE, RMSE, R², Explained Variance |
| Categorical Handling | Embedding Layers      | Efficient representation of categories     |
| Feature Engineering  | Manual via Pandas     | Custom domain-driven transformations       |
| Hardware Accelerator  | **NVIDIA Tesla P100** | GPU used for model training (via Kaggle)   |

---

## Objective

The goal of this project is not just to build an accurate model, but also to answer these proposed airline pricing questions:

**a)** Does price vary with different **airlines**?  
**b)** How is the price affected when tickets are bought just **1 or 2 days before departure**?  
**c)** Does the ticket price change based on **departure** and **arrival time**?  
**d)** How does price change between different **source** and **destination** combinations?  
**e)** How does the ticket price vary between **Economy** and **Business class**?

These questions are approached by building interpretable models and using smart features to capture these patterns.

---

## Dataset

The dataset comes directly from Kaggle's "Flight Price Prediction" challenge and contains details like:
- Airline, source/destination cities  
- Departure/arrival time categories  
- Number of stops, travel class, days left until flight  
- Duration of the flight  
- Price (target variable)

---

## Model Version 1: Basic Deep Learning Regressor

### What it does:
This version encodes all categorical variables numerically and feeds them into an embedding-based model, along with numerical features.

### 🔧 Key Code:
```python
cat_cols_v1 = ['airline', 'source_city', 'departure_time', 'stops', 
               'arrival_time', 'destination_city', 'class']

# Encode categoricals
for col in cat_cols_v1:
    df[col] = df[col].astype('category').cat.codes
```

Then create **embedding layers for each categorical variable**, flatten them, and concatenate them with the numeric features.

```python
embedding_sizes = {
    col: (df[col].nunique(), min(50, (df[col].nunique() + 1) // 2)) 
    for col in cat_cols_v1
}

x = Concatenate()(embedded_inputs + [numerical_input])
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1)(x)
```

---

## Model Version 2: With Feature Engineering

### What’s new here:
This version introduces several domain-inspired features designed to help answer the research questions more explicitly:

- **Departure hour** (mapped from `departure_time`)
- **Weekend indicator** (`is_weekend`)
- **Route frequency** (`route_count`)
- **Class + duration combo** (`economy_long_haul`)
- **Booking window category** (to explore last-minute pricing)
- **Duration binning** (short vs long-haul)
- **Peak season indicator**

### Feature Engineering Snippet:
```python
df['departure_hour'] = df['departure_time'].map({...})
df['is_weekend'] = df['departure_time'].isin(['Evening', 'Night']).astype(int)
df['booking_window'] = pd.cut(df['days_left'], bins=[0,3,7,14,30,365], labels=[...])
df['duration_bin'] = pd.cut(df['duration'], bins=[0,2,4,6,24], labels=[...])
```

The architecture remains the same: embeddings for categoricals + dense layers for regression.

---

## Results & Comparison

After training and evaluating both models:

| Metric             | V1        | V2        | Best Model    |
|--------------------|-----------|-----------|---------------|
| **MAE**            | ₹3207.09  | ₹3233.95  | V1 (slightly) |
| **RMSE**           | ₹5377.61  | ₹5191.02  | V2          |
| **R²**             | 0.9439    | 0.9477    | V2          |
| **Expl. Variance** | 0.9440    | 0.9478    | V2          |

---

## Visual Insights

### 📊 Error Distribution (V1 vs V2)

<img width="541" height="358" alt="image" src="https://github.com/user-attachments/assets/4a3150eb-2859-46cb-8c2f-ad26a3242759" />

> *Distribution of prediction errors for both models. V2 appears to have a tighter distribution with fewer extreme outliers.*

---

### 📦 Errors by Price Range

<img width="532" height="451" alt="image" src="https://github.com/user-attachments/assets/043f93c0-ff4e-4789-a38f-64678a00ad77" />

> *Absolute errors by price range. Both models show larger errors on expensive flights, but V2 performs slightly better.*

---

### 📉 Residual Analysis (V2)

<img width="1120" height="314" alt="image" src="https://github.com/user-attachments/assets/bc2efe07-6f50-4481-8171-09ff7d7db5c0" />

> *Residual analysis of V2. Most predictions are close to actual values, but the model struggles a bit with high-priced flights.*

---

## Learning

Now, connecting back to the **research questions**:

| Question | Findings |
|----------|-------------|
| **a. Airlines** | The `airline` variable had a strong impact on price, as seen in its embedding size and learned weights. |
| **b. Days before departure** | Price increases significantly when `days_left` is low. The `booking_window` feature showed clear patterns, especially in last-minute bins. |
| **c. Departure/Arrival Time** | Features like `departure_hour` and `is_weekend` helped capture time-based pricing differences. |
| **d. Source/Destination** | The `route_count` and city combinations revealed significant price differences. |
| **e. Class type** | Economy vs. Business was one of the most influential factors — clearly reflected in predictions. |

---

## Final Thoughts

Predicting flight prices isn't easy — airlines are unpredictable!  
But with the right data and a thoughtful mix of **embeddings + features**, one can get surprisingly close.
