import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("house_price_dataset.csv")

# Dataset preview
print("Dataset Preview:")
print(data.head())

# Features and target
X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict prices
predictions = model.predict(X_test)

print("\nPredicted Prices:")
print(predictions)

# Predict new house price
new_house = [[2000, 3, 2]]
predicted_price = model.predict(new_house)

print("\nPredicted price for new house:", predicted_price)

# Model accuracy
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# Graph visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.show()
