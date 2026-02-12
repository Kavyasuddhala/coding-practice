import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

# Explore Data set

print("First 10 rows")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

# Visualize CTR vs Views

plt.figure()
plt.scatter(df['ctr'], df['total_views'])
plt.xlabel('Thumbnail CTR (%)')
plt.ylabel('Total Views after 30 Days')
plt.title('CTR vs Total Views')
plt.show()

# Prepare the model

X = df[['ctr']]        
y = df['total_views']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=46
)

# Creating a model

model = LinearRegression()
model.fit(X_train, y_train)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Predict views for 8% CTR

ctr_value = np.array([[8.0]])
predicted_views = model.predict(ctr_value)

print("Predicted views for 8% CTR:", int(predicted_views[0]))

# visualize

plt.figure()
plt.scatter(df['ctr'], df['total_views'], label='Actual Data')
plt.plot(df['ctr'], model.predict(X), label='Regression Line')
plt.xlabel('Thumbnail CTR (%)')
plt.ylabel('Total Views after 30 Days')
plt.title('Linear Regression: CTR vs Views')
plt.legend()
plt.show()
