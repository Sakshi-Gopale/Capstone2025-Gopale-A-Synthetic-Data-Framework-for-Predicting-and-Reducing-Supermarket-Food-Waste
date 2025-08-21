import pandas as pd

#Read the CSV file (replace with your actual file path)
df = pd.read_csv("Sales_Dataset.csv")

#Ensure quantity_sold is numeric
df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce')

#Optional: Convert expiry_date to datetime for sorting or consistency
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df['expiry_date'] = pd.to_datetime(df['expiry_date'], dayfirst=True, errors='coerce')

#Calculate remaining shelf life in days
df['remaining_shelf_life'] = (df['expiry_date'] - df['date']).dt.days

#Group by expiry_date, category, and product, then sum quantity_sold
result = df.groupby(['expiry_date', 'category', 'product'])['quantity_sold'].sum().reset_index()

#Rename the column for clarity (optional)
result.rename(columns={'quantity_sold': 'total_quantity_sold'}, inplace=True)

extra_cols = df[['expiry_date', 'date', 'category', 'product', 'event', 'holiday', 'shelf_life','remaining_shelf_life']].drop_duplicates()
result = pd.merge(result, extra_cols, on=['expiry_date', 'category', 'product'], how='left')

#Display the result
print(result)

#Optionally, save to a new CSV
result.to_csv("quantity_by_category_and_expiry.csv", index=False)
