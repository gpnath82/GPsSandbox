import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('merged_data.csv')  # Replace 'your_file.csv' with your actual file name

# Select relevant features and target variable
X = df[['Sensex_Price', 'Sensex_Open', 'Sensex_High', 'Sensex_Low', 'Sensex_Vol.', 'Sensex_Change %', 
         'NYSE_Price', 'NYSE_Open', 'NYSE_High', 'NYSE_Low',  'NYSE_Change %',
         'TCS_Price', 'TCS_Open', 'TCS_High', 'TCS_Low', 'TCS_Vol.', 'TCS_Change %']]
y = df['USD_Price']

# Handle missing values (NaN) using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict USD price for a new data point
 # Replace with your new data point values in the same order as X
new_data = [[691645.3, 691998.78, 692151.02, 691574.89, 11.49, 0, 14652.7, 14487.3, 14653.2, 14485.8, 0.01, 3854.15, 3820, 3904.9, 3805.05, 2.36, 0.01]]

predicted_usd_price = model.predict(new_data)[0]

print(f'Predicted USD Price: {predicted_usd_price}')