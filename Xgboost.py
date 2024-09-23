import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
#import statsmodels.api as sm
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split
#from arabic_reshaper import arabic_reshaper
#from bidi.algorithm import get_display
#import arabic_reshaper
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Enable Arabic text rendering
mpl.rcParams['font.family'] = 'DejaVu Sans'  # Specify a font with Arabic support



# Set the Arabic font

plt.style.use('ggplot')
#pd.set_option('max_columns', 200)
file_path = 'C:/Users/Lenovo/Downloads/EDS_DATA.xlsx'
file_path2='C:/Users/Lenovo/Downloads/Data_New.xlsx'
file_path3='C:/Users/Lenovo/Downloads/popultionsize.xlsx'
file_path4='C:/Users/Lenovo/Downloads/clinics.xlsx'
file_path5='C:/Users/Lenovo/Downloads/Govname_id.xlsx'
df = pd.read_excel(file_path)
df2= pd.read_excel(file_path2)
df3=pd.read_excel(file_path3)
df4=pd.read_excel(file_path4)
df5=pd.read_excel(file_path5)
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
df2['Shipping'] = df2['Shipping'].replace(r'^\s*$', np.nan, regex=True)

# Convert column 'A' to integer type
df2['Shipping'] = df2['Shipping'].astype(float).astype(pd.Int64Dtype())


df['GovbyNum'] = pd.to_numeric(df['GovbyNum'], errors='coerce')

# Convert NaN values to a specific placeholder value (e.g., -1)
df['GovbyNum'] = df['GovbyNum'].fillna(-1)

# Convert the 'Col1' column from float to integer
df['GovbyNum'] = df['GovbyNum'].astype(int)
df.drop(columns='subsyndicate',inplace=True)
df.drop(columns='university',inplace=True)
def count_unique_values_per_week(df):
    # Convert 'Date' column to datetime type


    # Group the DataFrame by 'governorateName', 'Week', and 'orderId'
    order_counts = df.groupby(['governorateName', 'Weeks'])['orderId'].nunique().reset_index()

    # Group the DataFrame by 'governorateName', 'Week', and 'registrationNo'
    registration_counts = df.groupby(['governorateName', 'Weeks'])['registrationNo'].nunique().reset_index()

    merged_df = pd.merge(order_counts, registration_counts, on=['governorateName', 'Weeks'], how='outer')

    return merged_df

merged_counts = count_unique_values_per_week(df)

# Print the merged DataFrame
#print(merged_counts)
def sum_values_by_unique_column(df, unique_column, sum_column,product):
    filtered_df = df[df['productName'] == product]
    # Group the dataframe by the unique column and calculate the sum of the values in the sum column
    grouped_data = filtered_df.groupby([unique_column, 'Weeks'])[sum_column].sum().reset_index()

    return grouped_data
Artinibsa = sum_values_by_unique_column(df, 'governorateName', 'quantity','ARTINIBSA 4% 1:100000')
ALEXADRICAINE = sum_values_by_unique_column(df, 'governorateName', 'quantity','ALEXADRICAINE 68MG + 17 MG 50 AMP')
ART_PHARMA= sum_values_by_unique_column(df, 'governorateName', 'quantity','ART PHARMA 4% 1:100,000')
MEPECAIN = sum_values_by_unique_column(df, 'governorateName', 'quantity','MEPECAINE CARPULES 3 % 50 1.80 ML% red')
merged_df = pd.merge(merged_counts, Artinibsa,on=['governorateName', 'Weeks'], how='left')
merged_df = pd.merge(merged_df, ALEXADRICAINE,on=['governorateName', 'Weeks'], how='left')
merged_df.rename(columns={'quantity_x': 'Artinibsa'}, inplace=True)
merged_df.rename(columns={'quantity_y': 'ALEXADRICAINE'}, inplace=True)
merged_df = pd.merge(merged_df, ART_PHARMA,on=['governorateName', 'Weeks'], how='left')
merged_df = pd.merge(merged_df, MEPECAIN,on=['governorateName', 'Weeks'], how='left')
merged_df.rename(columns={'quantity_x': 'ART_PHARMA'}, inplace=True)
merged_df.rename(columns={'quantity_y': 'MEPECAIN'}, inplace=True)
#print(merged_df)
def replace_null_with_zeros(df):
    # Replace all null values with zeros
    df_filled = df.fillna(0)
    
    return df_filled
#merged_df.to_excel('output.xlsx', index=False)
def average_values_by_unique_column(df, unique_column, sum_column):
    # Group the dataframe by the unique column and calculate the sum of the values in the sum column
    grouped_data =df.groupby([unique_column, 'Weeks'])[sum_column].mean().reset_index()

    return grouped_data

# Calculate the average shipping per week per governmente
grouped_data=average_values_by_unique_column(df2,'governorateName','Shipping')
#print(grouped_data)
merged_df = pd.merge(merged_df, grouped_data,on=['governorateName', 'Weeks'], how='left')
def mean_values_by_unique_column(df, unique_column, sum_column,product):
    filtered_df = df[df['productName'] == product]
    # Group the dataframe by the unique column and calculate the sum of the values in the sum column
    grouped_data = filtered_df.groupby([unique_column, 'Weeks'])[sum_column].mean().reset_index()

    return grouped_data
Artinibsa_price = mean_values_by_unique_column(df, 'governorateName', 'productPrice','ARTINIBSA 4% 1:100000')
merged_df = pd.merge(merged_df, Artinibsa_price,on=['governorateName', 'Weeks'], how='left')
merged_df = pd.merge(merged_df, df3,on=['governorateName'], how='left')
def count_values_by_unique_column(df, unique_column, sum_column,product):
    filtered_df = df[df['Name'].str.contains(product)]
    # Group the dataframe by the unique column and calculate the sum of the values in the sum column
    grouped_data = filtered_df.groupby([unique_column])[sum_column].nunique().reset_index()

    return grouped_data


grouped_data_clinics=count_values_by_unique_column(df4,'Gov Id','License No','عيادة')
grouped_data_centres=count_values_by_unique_column(df4,'Gov Id','License No','مركز')
merged_data = pd.merge(grouped_data_clinics, grouped_data_centres,on=['Gov Id'], how='left')
merged_data=replace_null_with_zeros(merged_data)
merged_data['seats']=merged_data['License No_y'] * 5 + merged_data['License No_x']
#merged_data.to_excel('output2.xlsx', index=False)
merged_data = pd.merge(merged_data, df5,on=['Gov Id'], how='left')
merged_df = pd.merge(merged_df, merged_data,on=['governorateName'], how='left')
merged_df=replace_null_with_zeros(merged_df)
merged_df .drop(columns='Gov Id',inplace=True)
merged_df.rename(columns={'License No_x': 'Number_Clinics'}, inplace=True)
merged_df.rename(columns={'License No_y': 'Number_Centres'}, inplace=True)

merged_df['productPrice'] = merged_df['productPrice'].replace(0, 600)
merged_df.rename(columns={'productPrice': 'Artinibsa_price'}, inplace=True)
merged_df.rename(columns={'orderId': 'Number_orders'}, inplace=True)
merged_df.rename(columns={'registrationNo': 'Number_doctors'}, inplace=True)
merged_df.drop(columns='Government by number',inplace=True)
correlation_matrix = merged_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Columns')
plt.show()

def sum_quantity_x_by_week(dataframe):
    grouped_data = dataframe.groupby('Weeks')['Artinibsa'].sum().reset_index()
    return grouped_data

# Example usage
def sum_quantity_y_by_week(dataframe):
    grouped_data = dataframe.groupby('Weeks')['ALEXADRICAINE'].sum().reset_index()
    return grouped_data
def sum_quantity_Z_by_week(dataframe):
    grouped_data = dataframe.groupby('Weeks')['ART_PHARMA'].sum().reset_index()
    return grouped_data
def sum_quantity_T_by_week(dataframe):
    grouped_data = dataframe.groupby('Weeks')['MEPECAIN'].sum().reset_index()
    return grouped_data
#reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in merged_df['governorateName']]

sum_by_weekx = sum_quantity_x_by_week(merged_df)
sum_by_weeky = sum_quantity_y_by_week(merged_df)
sum_by_weekz = sum_quantity_Z_by_week(merged_df)
sum_by_weekt = sum_quantity_T_by_week(merged_df)

sum_by_week = pd.merge(sum_by_weekx, sum_by_weeky,on=[ 'Weeks'], how='left')
sum_by_week = pd.merge(sum_by_week, sum_by_weekz,on=[ 'Weeks'], how='left')
sum_by_week = pd.merge(sum_by_week, sum_by_weekt,on=[ 'Weeks'], how='left')

weeks = sum_by_week['Weeks']
artinibsa = sum_by_week['Artinibsa']
alexandican = sum_by_week['ALEXADRICAINE']
ART_pharma=sum_by_week['ART_PHARMA']
MEPECAIN=sum_by_week['MEPECAIN']
plt.plot(weeks, artinibsa, marker='o', label='Artinibsa')
plt.plot(weeks, alexandican, marker='o', label='Alexandican')
plt.plot(weeks, ART_pharma, marker='o', label='ART_pharma')
plt.plot(weeks, MEPECAIN, marker='o', label='MEPECAIN')
plt.xlabel('Weeks')
plt.ylabel('Quantity')
plt.title('Line Graphs: Artinibsa, Alexandican, art pharma, mepecain')
plt.legend()
plt.grid(True)
plt.show()
#reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in merged_df['governorateName']]

doctors_per_week = merged_df.groupby('governorateName')['Number_doctors'].sum()
dr=pd.DataFrame(doctors_per_week)
dr.reset_index(inplace=True)
#reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in dr['governorateName']]
plt.figure(figsize=(10, 6))
#plt.bar(reshaped_words, dr['Number_doctors'])
plt.xlabel('governrate Name')
plt.ylabel('Sum of Number of Doctors')
plt.title('Bar Graph: Sum of Number of Doctors per Week')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

orders_per_week = merged_df.groupby('governorateName')['Number_orders'].sum()
do=pd.DataFrame(orders_per_week)
do.reset_index(inplace=True)
#reshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in do['governorateName']]
plt.figure(figsize=(10, 6))
#plt.bar(reshaped_words, do['Number_orders'])
plt.xlabel('governrate Name')
plt.ylabel('Sum of Number of orders')
plt.title('Bar Graph: Sum of Number of orders per Week')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

clinics_per_week = merged_df.groupby('governorateName')['Number_Clinics'].sum()
clinics=pd.DataFrame(clinics_per_week)
clinics.reset_index(inplace=True)
#eshaped_words = [get_display(arabic_reshaper.reshape(word)) for word in clinics['governorateName']]
plt.figure(figsize=(10, 6))
#plt.bar(reshaped_words, clinics['Number_Clinics'])
plt.xlabel('governrate Name')
plt.ylabel('Sum of Number of clinics')
plt.title('Bar Graph: Sum of Number of clinics')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

centres_per_week = merged_df.groupby('governorateName')['Number_Centres'].sum()
centres=pd.DataFrame(centres_per_week)
centres.reset_index(inplace=True)
plt.figure(figsize=(10, 6))
#plt.bar(reshaped_words, centres['Number_Centres'])
plt.ylabel('Sum of Number of Centres')
plt.title('Bar Graph: Sum of Number of Centres')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

seats_per_week = merged_df.groupby('governorateName')['seats'].sum()
seats=pd.DataFrame(seats_per_week)
plt.figure(figsize=(10, 6))
seats.reset_index(inplace=True)
#plt.bar(reshaped_words, seats['seats'])
plt.xlabel('governrate Name')
plt.ylabel('Sum of Number of seats')
plt.title('Bar Graph: Sum of Number of seats')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

merged_df['Shipping'] = merged_df['Shipping'].astype(int)
merged_df['populationsize'] = merged_df['populationsize'].astype(int)
merged_df['Artinibsa'] = merged_df['Artinibsa'].astype(int)
merged_df['seats'] = merged_df['seats'].astype(int)
correlation = merged_df['Shipping'].corr(merged_df['Number_orders'])

# Plot the scatter plot
sns.scatterplot(data=merged_df, x='Shipping', y='Number_orders')
plt.title(f'Correlation: {correlation:.2f}')
plt.show()

correlation = merged_df['populationsize'].corr(merged_df['Number_orders'])
sns.scatterplot(data=merged_df, x='populationsize', y='Number_orders')
plt.title(f'Correlation: {correlation:.2f}')
plt.show()

correlation = merged_df['seats'].corr(merged_df['Number_orders'])
sns.scatterplot(data=merged_df, x='seats', y='Number_orders')
plt.title(f'Correlation: {correlation:.2f}')
plt.show()

correlation = merged_df['Number_doctors'].corr(merged_df['Number_orders'])
sns.scatterplot(data=merged_df, x='Number_doctors', y='Number_orders')
plt.title(f'Correlation: {correlation:.2f}')
plt.show()

X = merged_df.drop('Artinibsa', axis=1)
y = merged_df['Artinibsa']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53)
model = xgb.XGBClassifier()

# Set the hyperparameters
params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the model
model.fit(X_train, y_train, eval_metric='error', eval_set=[(X_test, y_test)])
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)