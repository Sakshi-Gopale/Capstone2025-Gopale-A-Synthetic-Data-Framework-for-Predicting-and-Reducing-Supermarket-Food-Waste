import pandas as pd

#Reading the CSV file 
df = pd.read_csv("Sales_Dataset_Train.csv")

#Ensuring quantity_sold is numeric
df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce')

#Converting expiry_date to datetime for sorting or consistency
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df['expiry_date'] = pd.to_datetime(df['expiry_date'], dayfirst=True, errors='coerce')

#Calculating remaining shelf life in days
df['remaining_shelf_life'] = (df['expiry_date'] - df['date']).dt.days

#Grouping by expiry_date, category, and product, then summing quantity_sold
result = df.groupby(['expiry_date', 'category', 'product'])['quantity_sold'].sum().reset_index()

#Renaming the column for clarity
result.rename(columns={'quantity_sold': 'total_quantity_sold'}, inplace=True)

extra_cols = df[['expiry_date', 'date', 'category', 'product', 'event', 'holiday', 'shelf_life','remaining_shelf_life']].drop_duplicates()
result = pd.merge(result, extra_cols, on=['expiry_date', 'category', 'product'], how='left')

#Displaying the result
print(result)

#Saving to a new CSV
result.to_csv("quantity_by_category_and_expiry.csv", index=False)