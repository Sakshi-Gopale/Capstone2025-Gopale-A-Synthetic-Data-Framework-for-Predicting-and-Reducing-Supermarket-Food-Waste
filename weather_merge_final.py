import pandas as pd

#Loading sales data (with day-first format)
sales_df = pd.read_csv('data/Sales_Datanew.csv')
sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, errors='coerce')

#Loading weather data (with day-first format too)
weather_df = pd.read_csv('data/weather_data.csv')
weather_df['date'] = pd.to_datetime(weather_df['date'], dayfirst=True, errors='coerce')

#Merging on 'date'
merged_df = pd.merge(sales_df, weather_df, on='date', how='left')

merged_df = merged_df.drop(columns=['customer_type', 'payment'])

#Saving the result
merged_df.to_csv('data/sales_weather_data.csv', index=False)

print("Merged CSV file successfully created")
