import pandas as pd
import os

# Define the data folder path
data_folder = "data"

# Create an empty list to store the dataframes
dfs = []

# Iterate through the CSV files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(os.path.join(data_folder, filename))
        # Append the DataFrame to the list
        dfs.append(df)

# Merge the DataFrames based on the 'Date' column
merged_df = pd.merge(dfs[0], dfs[1], on='Date', how='outer')
for i in range(2, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on='Date', how='outer')

# Convert the 'Date' column to DD/MM/YYYY format
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%m/%d/%Y')
merged_df = merged_df.sort_values(by='Date', ascending=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_data.csv", index=False)



print("Combined CSV file created successfully.")