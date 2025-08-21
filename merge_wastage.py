import pandas as pd

#Load sales data
sales_df = pd.read_csv("quantity_by_category_and_expiry.csv")

#Load waste data
waste_df = pd.read_csv("Inventory_Dataset.csv")

#Convert expiry_date to datetime in both dataframes
sales_df['expiry_date'] = pd.to_datetime(sales_df['expiry_date'], errors='coerce')
waste_df['expiry_date'] = pd.to_datetime(waste_df['expiry_date'], format='%d-%m-%Y', errors='coerce')

#Merge on product, category, and expiry_date
merged_df = pd.merge(
    sales_df,
    waste_df,
    how='left',
    on=['product', 'category', 'expiry_date']
)

#Rename for clarity
merged_df.rename(columns={'quantity': 'waste_quantity'}, inplace=True)

#Fill missing waste quantities with 0
merged_df['waste_quantity'] = pd.to_numeric(merged_df['waste_quantity'], errors='coerce').fillna(0)
merged_df['total_quantity_sold'] = pd.to_numeric(merged_df['total_quantity_sold'], errors='coerce')

#Calculate actual_quantity_to_be_sold
merged_df['actual_quantity_to_be_sold'] = merged_df['total_quantity_sold'] - merged_df['waste_quantity']

#Add quantity_to_be_sold based on condition
merged_df['quantity_to_be_sold'] = merged_df.apply(
    lambda row: row['total_quantity_sold'] if row['actual_quantity_to_be_sold'] == 0 else row['actual_quantity_to_be_sold'],
    axis=1
)

#Add waste_product_quantity only if actual_quantity_to_be_sold ≠ 0
merged_df['waste_product_quantity'] = merged_df.apply(
    lambda row: row['waste_quantity'] if row['actual_quantity_to_be_sold'] != 0 else 0,
    axis=1
)

#Drop unwanted columns
columns_to_drop = ['waste_flag', 'actual_quantity_to_be_sold', 'waste_quantity', 'quantity_to_be_sold']
merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns], inplace=True)

#Rename total_quantity_sold → total_quantity
merged_df.rename(columns={'total_quantity_sold': 'total_quantity'}, inplace=True)


#Reorder columns
desired_order = [
    'product', 'category', 'expiry_date', 'shelf_life','remaining_shelf_life',
    'total_quantity', 'waste_product_quantity', 'event', 'holiday', "date"
]

#Keep only columns that exist (in case any is missing)
final_columns = [col for col in desired_order if col in merged_df.columns]
merged_df = merged_df[final_columns]


#Display final merged data
print(merged_df)

#Save to CSV
merged_df.to_csv("wastage_data.csv", index=False)
