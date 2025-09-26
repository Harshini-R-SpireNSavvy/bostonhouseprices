import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# CSV path
csv_file = r"C:\endproject\bostonhouseprices\Boston House Prices.csv"

# Check if CSV exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} not found!")

# Load CSV
df = pd.read_csv(csv_file)

# Correct column names
X = df[['Rooms', 'Distance']]
y = df['Value']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
regmodel = LinearRegression()
regmodel.fit(X_scaled, y)

# Save model and scaler
pickle.dump(regmodel, open('regmodel.pkl', 'wb'))
pickle.dump(scaler, open('scaling.pkl', 'wb'))

print("Training complete. regmodel.pkl and scaling.pkl are saved!")
