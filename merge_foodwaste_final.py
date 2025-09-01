import pandas as pd

#Loading sales data
sales_df = pd.read_csv("quantity_by_category_and_expiry.csv")

#Loading waste data
waste_df = pd.read_csv("Inventory_Dataset_Train.csv")

#Converting expiry_date to datetime in both dataframes
sales_df['expiry_date'] = pd.to_datetime(sales_df['expiry_date'], errors='coerce')
waste_df['expiry_date'] = pd.to_datetime(waste_df['expiry_date'], format='%d-%m-%Y', errors='coerce')

#Merging on product, category, and expiry_date
merged_df = pd.merge(
    sales_df,
    waste_df,
    how='left',
    on=['product', 'category', 'expiry_date']
)

#Filling missing waste quantities with 0
merged_df['quantity'] = pd.to_numeric(merged_df['quantity'], errors='coerce')
merged_df['total_quantity_sold'] = pd.to_numeric(merged_df['total_quantity_sold'], errors='coerce')

#Calculating actual_quantity_to_be_waste
merged_df['waste_product_quantity'] = merged_df['quantity'] - merged_df['total_quantity_sold']

merged_df.rename(columns={'quantity': 'total_quantity'}, inplace=True)

#Reordering columns
desired_order = [
    'product', 'category', 'expiry_date', 'shelf_life','remaining_shelf_life',
    'total_quantity', 'waste_product_quantity', 'event', 'holiday', "date"
]

#Keeping only columns that exist (in case any is missing)
final_columns = [col for col in desired_order if col in merged_df.columns]
merged_df = merged_df[final_columns]

#Displaying final merged data
print(merged_df)

#Saving to CSV
merged_df.to_csv("wastage_data.csv", index=False)