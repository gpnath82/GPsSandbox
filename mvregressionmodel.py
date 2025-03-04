
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Load the data.  Specify the date format explicitly!
data = pd.read_csv("merged_data.csv", parse_dates=['Date'], date_format='%Y-%m-%d')

def create_and_evaluate_model(X_cols, y_col, model_name):
    """Creates, trains, and evaluates a linear regression model."""
    X = data[X_cols].copy()
    y = data[y_col]
    # Feature engineering
    X['Date'] = pd.to_datetime(X['Date'], format='%d/%m/%y', errors='coerce')  
    X['Day of Week'] = X['Date'].dt.dayofweek
    X['Year'] = X['Date'].dt.year
    X['Month'] = X['Date'].dt.month
    X['Day'] = X['Date'].dt.day 
    X = X.drop('Date', axis=1) #Remove the original date column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #print(f"Mean Squared Error: {model_name} {mse:.6f}")
    print(f"R-squared: {model_name} {r2:.2f}")
    return model


# Create and evaluate models
USDmodel = create_and_evaluate_model(['Date', 'Sensex_Price', 'NYSE_Price', 'TCS_Price'], 'USD_Price', 'USDmodel')
Sensexmodel = create_and_evaluate_model(['Date', 'NYSE_Price', 'TCS_Price', 'USD_Price'], 'Sensex_Price', 'Sensexmodel')
NYSEmodel = create_and_evaluate_model(['Date', 'Sensex_Price', 'TCS_Price', 'USD_Price'], 'NYSE_Price', 'NYSEmodel')
TCSmodel = create_and_evaluate_model(['Date', 'Sensex_Price', 'NYSE_Price', 'USD_Price'], 'TCS_Price', 'TCSmodel')



def predict_price(model_name,model, date_str, sensex_price, nyse_price, tcs_price, usd_price):
    """Generic prediction function."""
    try:
        date = datetime.strptime(date_str, '%d/%m/%Y')
    except ValueError:
        return "Invalid date format. Use DD/MM/YYYY"
    
    input_data = pd.DataFrame({
        'Date': [date.toordinal()],
        'Sensex_Price': [sensex_price],
        'NYSE_Price': [nyse_price],
        'TCS_Price': [tcs_price],
        'USD_Price': [usd_price],
        'Day of Week': [date.weekday()],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day]
    })
    df = pd.DataFrame(input_data)
    
    # Create a copy to avoid modifying the original dataframe. Important for iterative predictions.
    df_copy = df.copy()
    
    if model_name == 'Sensex':
        features = ['NYSE_Price', 'TCS_Price', 'USD_Price', 'Day of Week', 'Year', 'Month', 'Day']
    elif model_name == 'NYSE':
        features = ['Sensex_Price', 'TCS_Price', 'USD_Price', 'Day of Week', 'Year', 'Month', 'Day']
    elif model_name == 'TCS':
        features = ['Sensex_Price', 'NYSE_Price', 'USD_Price', 'Day of Week', 'Year', 'Month', 'Day']
    elif model_name == 'USD':
        features = ['Sensex_Price', 'NYSE_Price', 'TCS_Price', 'Day of Week', 'Year', 'Month', 'Day']
    else:
        return "Invalid model name"
    try:
      prediction = model.predict(df_copy[features])[0]
      return prediction
    except ValueError as e:
      return f"Prediction error: {e}"


date = '05/03/2025'
sensex_price = 72989
nyse_price = 18296
tcs_price = 3533
usd_price = 87.26
NumberOfDaystobePredicted=7

for i in range(NumberOfDaystobePredicted):
    print(f"Predicted Date: {date}")
    sensex_price = predict_price('Sensex',Sensexmodel, date, sensex_price, nyse_price, tcs_price, usd_price) #0 is a placeholder
    print(f"Predicted Sensex Price: {sensex_price}")    

    nyse_price = predict_price('NYSE',NYSEmodel, date, sensex_price, nyse_price, tcs_price, usd_price)
    print(f"Predicted NYSE Price: {nyse_price}")

    tcs_price = predict_price('TCS',TCSmodel, date, sensex_price, nyse_price, tcs_price, usd_price)
    print(f"Predicted TCS Price: {tcs_price}")

    predicted_usd_price = predict_price('USD',USDmodel, date, sensex_price, nyse_price, tcs_price, usd_price)
    print(f"Predicted USD Price: {predicted_usd_price}")
    date = datetime.strptime(date, '%d/%m/%Y').date()
    date += timedelta(days=1)
    date = date.strftime('%d/%m/%Y')