import pandas as pd

# Load sales data (with day-first format)
sales_df = pd.read_csv('../data/Sales_Datanew.csv')
sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, errors='coerce')

# Load weather data (with day-first format too)
weather_df = pd.read_csv('../data/weather_data.csv')
weather_df['date'] = pd.to_datetime(weather_df['date'], dayfirst=True, errors='coerce')

# Merge on 'date'
merged_df = pd.merge(sales_df, weather_df, on='date', how='left')

merged_df = merged_df.drop(columns=['customer_type', 'payment'])
# Save the result
merged_df.to_csv('../data/sales_weather_data.csv', index=False)

print("Merged CSV file successfully created")
