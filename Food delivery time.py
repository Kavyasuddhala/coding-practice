import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

# 2. Explore the Data

#check first few rows
print(df.head())

#Basic info
print(df.info())

#statistical summary
print(df.describe())

# 3. Visualize Relationships

# Distance vs Delivery Time

plt.figure()
plt.scatter(df['distance_km'], df['delivery_time_min'])
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (min)')
plt.title('Distance vs Delivery Time')
plt.show()

# Prep Time vs Delivery Time

plt.figure()
plt.scatter(df['prep_time_min'], df['delivery_time_min'])
plt.xlabel('Preparation Time (min)')
plt.ylabel('Delivery Time (min)')
plt.title('Prep Time vs Delivery Time')
plt.show()


# 4. Prepare the data for ML 

X = df[['distance_km', 'prep_time_min']]  # Two features
y = df['delivery_time_min']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Multiple Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Model Coefficients

print("\nModel Equation:")
print(f"Delivery Time = {model.coef_[0]:.2f} * Distance + "
      f"{model.coef_[1]:.2f} * Prep Time + {model.intercept_:.2f}")

print("\nCoefficients Interpretation:")
print(f"Distance coefficient: {model.coef_[0]:.2f}")
print(f"Prep time coefficient: {model.coef_[1]:.2f}")

# 7. Prediction for Vikram's Question

new_data = np.array([[7.0, 15]])  # 7 km distance, 15 min prep time
predicted_time = model.predict(new_data)

print(f"\nExpected delivery time for 7 km & 15 min prep: "
      f"{int(predicted_time[0])} minutes")
