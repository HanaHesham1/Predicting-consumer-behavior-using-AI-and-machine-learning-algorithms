import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
#pd.set_option('max_columns', 200)
file_path = 'C:/Users/Lenovo/Downloads/EDS_DATA.xlsx'
df = pd.read_excel(file_path)
#print(df.shape)
#print (df.head(5))
#print(df.describe())
#df.drop(['university'],axis=1
#pd.to_datetime(df['Date'])
#df['orderId']=df['orderId'].astype(int)

df['orderId'] = pd.to_numeric(df['orderId'], errors='coerce')

# Convert NaN values to a specific placeholder value (e.g., -1)
df['orderId'] = df['orderId'].fillna(-1)

# Convert the 'Col1' column from float to integer
df['orderId'] = df['orderId'].astype(int)
df['GovbyNum'] = pd.to_numeric(df['GovbyNum'], errors='coerce')

# Convert NaN values to a specific placeholder value (e.g., -1)
df['GovbyNum'] = df['GovbyNum'].fillna(-1)
grouped=df.groupby('GovbyNum')

# Convert the 'Col1' column from float to integer
df['GovbyNum'] = df['GovbyNum'].astype(int)
df.drop(columns='subsyndicate',inplace=True)
df.drop(columns='university',inplace=True)
df.drop(columns='registrationNo',inplace=True)
# Verify the data types of the columns
print(df.dtypes)
#print(df.dtypes)
print(df.isna().sum())
#df['governorateName'].value_counts().head(2000).plot(kind='bar')
#df['Date'].value_counts().head(10).plot(kind='kde')
#df.plot(kind='scatter',x='Date', y='orderId' )
Months = {'Month':['Jan','Feb','Mar','Apr','Mai','June','July','Aug','Seb','Oct','Nov','Dec','Jan','feb','Mar'],
          'count':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}
Month = pd.DataFrame(Months)
countjan = df['Date'].str.contains('2023-01').sum()
count='count'
Month.loc[0,count]=countjan
countfeb = df['Date'].str.contains('2023-02').sum()
Month.loc[1,count]=countfeb
countmar = df['Date'].str.contains('2023-03').sum()
Month.loc[2,count]=countmar
countapr = df['Date'].str.contains('2023-04').sum()
Month.loc[3,count]=countapr
countmai = df['Date'].str.contains('2023-05').sum()
Month.loc[4,count]=countmai
countjun = df['Date'].str.contains('2023-06').sum()
Month.loc[5,count]=countjun
countjul = df['Date'].str.contains('2023-07').sum()
Month.loc[6,count]=countjul
countjul = df['Date'].str.contains('2023-08').sum()
Month.loc[7,count]=countjul
countjul = df['Date'].str.contains('2023-09').sum()
Month.loc[8,count]=countjul
countjul = df['Date'].str.contains('2023-10').sum()
Month.loc[9,count]=countjul
countjul = df['Date'].str.contains('2023-11').sum()
Month.loc[10,count]=countjul
countjul = df['Date'].str.contains('2023-12').sum()
Month.loc[11,count]=countjul
countjul = df['Date'].str.contains('2024-01').sum()
Month.loc[12,count]=countjul
countjul = df['Date'].str.contains('2024-02').sum()
Month.loc[13,count]=countjul
countjul = df['Date'].str.contains('2024-03').sum()
Month.loc[14,count]=countjul
Month.plot(kind='bar',x='Month', y='count' )
sumprices={'Month':['Jan','Feb','Mar','Apr','Mai','June','July','Aug','Seb','Oct','Nov','Dec','Jan','feb','Mar'],
           'Total Price':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]}
Price=pd.DataFrame(sumprices)
sum_result = df.loc[df['Date'].str.contains('2023-01',case=False) , 'productPrice'].sum()
Total_price='Total Price'
Price.loc[0,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-02',case=False) , 'productPrice'].sum()
Price.loc[1,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-03',case=False) , 'productPrice'].sum()
Price.loc[2,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-04',case=False) , 'productPrice'].sum()
Price.loc[3,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-05',case=False) , 'productPrice'].sum()
Price.loc[4,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-06',case=False) , 'productPrice'].sum()
Price.loc[5,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-07',case=False) , 'productPrice'].sum()
Price.loc[6,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-08',case=False) , 'productPrice'].sum()
Price.loc[7,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-09',case=False) , 'productPrice'].sum()
Price.loc[8,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-10',case=False) , 'productPrice'].sum()
Price.loc[9,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-11',case=False) , 'productPrice'].sum()
Price.loc[10,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2023-12',case=False) , 'productPrice'].sum()
Price.loc[11,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2024-01',case=False) , 'productPrice'].sum()
Price.loc[12,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2024-02',case=False) , 'productPrice'].sum()
Price.loc[13,Total_price]=sum_result
sum_result = df.loc[df['Date'].str.contains('2024-03',case=False) , 'productPrice'].sum()
Price.loc[14,Total_price]=sum_result
Price.plot(kind='bar',x='Month', y='Total Price' )
unique_count = df['governorateName'].nunique()
print("Number of unique values:", unique_count)
unique_values = df['governorateName'].unique()
print("Unique values:", unique_values)

def count_unique_values(df, column_name):
    # Get the unique values and their counts
    unique_values_counts = df[column_name].value_counts().reset_index()

    # Rename the columns in the new dataframe
    unique_values_counts.columns = ['Unique_Value', 'Count']

    return unique_values_counts


def sum_values_by_unique_column(df, unique_column, sum_column):
    # Group the dataframe by the unique column and calculate the sum of the values in the sum column
    grouped_data = df.groupby(unique_column)[sum_column].sum().reset_index()
    
    return grouped_data

# Example usage

# Call the function to calculate the sum of values in 'SumColumn' for each unique value in 'UniqueColumn'
result_df = sum_values_by_unique_column(df, 'governorateName', 'quantity')
sum_sales_per_gov = sum_values_by_unique_column(df, 'GovbyNum', 'Total_outcome')
print(sum_sales_per_gov)
print(result_df)
#plt.bar(sum_sales_per_gov['governorateName'], sum_sales_per_gov['productPrice'])
#plt.figure(figsize=(10, 6))
#plt.xticks(rotation=75)
#plt.xlabel('governorateName')
#plt.ylabel('sales')
#plt.title('sales per governmente')
#plt.show()

# Example usage
result_df = count_unique_values(df, 'governorateName')

# Print the resulting dataframe
print(result_df)
#plt.bar(result_df['Unique_Value'], result_df['Count'])
#plt.xlabel('Unique Values')
#plt.ylabel('Count')
#plt.title('Count of Unique Values')
#plt.show()
pivot_table = df.pivot_table(values='governorateName', index='productPrice', columns='Date')
#plt.figure(figsize=(40, 20))


#heatmap=sns.heatmap(pivot_table, annot=True, fmt=".0f")

#plt.yticks(rotation=0)
#plt.xlabel('Month')
#plt.ylabel('Year')
#plt.title('Heatmap of Close based on Year and Month')
#plt.show()
print(df.describe())
print(df.corr())
print(df.info())
#sns.boxplot(df['Total_outcome'])
#plt.show()

val2 = 1

# Value to find in 'col3'
val3 = 'القاهرة'

# Filter data based on conditions
filtered_df = df[(df['Weeks'] == val2) & (df['governorateName'] == val3)]

# Count the number of unique values in 'col1'
count = filtered_df['registrationNo'].nunique()

# Print the count
print(count)
