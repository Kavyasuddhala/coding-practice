import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

# 2. Explore the Data

#check first few rows
print(df.head())

# Basic info
print(df.info())

# Statistical summary
print(df.describe())

# 3. Visualize Relationships

plt.figure()
plt.scatter(df['ram_gb'], df['price_inr'])
plt.xlabel('RAM (GB)')
plt.ylabel('Price (INR)')
plt.title('RAM vs Price')
plt.show()

plt.figure()
plt.scatter(df['storage_gb'], df['price_inr'])
plt.xlabel('Storage (GB)')
plt.ylabel('Price (INR)')
plt.title('Storage vs Price')
plt.show()

plt.figure()
plt.scatter(df['processor_ghz'], df['price_inr'])
plt.xlabel('Processor Speed (GHz)')
plt.ylabel('Price (INR)')
plt.title('Processor Speed vs Price')
plt.show()

# 4. Prepare Data for ML

X = df[['ram_gb', 'storage_gb', 'processor_ghz']]
y = df['price_inr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# 6. Model Coefficients

print("\nModel Equation:")
print(f"Price = {model.coef_[0]:.2f}*RAM + "
      f"{model.coef_[1]:.2f}*Storage + "
      f"{model.coef_[2]:.2f}*Processor + "
      f"{model.intercept_:.2f}")

print("\nCoefficient Impact:")
print(f"RAM coefficient: {model.coef_[0]:.2f}")
print(f"Storage coefficient: {model.coef_[1]:.2f}")
print(f"Processor coefficient: {model.coef_[2]:.2f}")

# 7. R² Score

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR² Score: {r2:.2f}")

# 8. Predict Meera's Laptop Price

meera_laptop = np.array([[16, 512, 3.2]])
predicted_price = model.predict(meera_laptop)

print(f"\nPredicted price for 16GB RAM, 512GB storage, 3.2GHz processor:")
print(f"₹{int(predicted_price[0])}")

# 9. Bonus: Check Overpricing

bonus_laptop = np.array([[8, 512, 2.8]])
predicted_bonus_price = model.predict(bonus_laptop)

actual_price = 55000

print("\nBonus Analysis:")
print(f"Predicted fair price: ₹{int(predicted_bonus_price[0])}")
print(f"Actual listed price: ₹{actual_price}")

if actual_price > predicted_bonus_price[0]:
    print("⚠️ This laptop appears OVERPRICED.")
else:
    print("✅ This laptop is reasonably priced.")

