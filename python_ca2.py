import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data
df = pd.read_csv("Air_Quality.csv")

print(df)

#Obj1] To understand the structure and composition of the dataset, including data types, feature names, and summary statistics.

#Structure
print("\nShape of dataset:",df.shape)
print("\n Data_types:",df.dtypes)
print("\n Column names:",df.columns.tolist())

#preview the dataset
print("\n First 5 Rows:",df.head())

#summary
print("\n Summary:",df.describe())

#Obj2] Detect and Handle Missing Values, Duplicates, and Inconsistencies

print("\n Missing values:",df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="OrRd")
plt.title("Missing Value Heatmap")
plt.show()
plt.close()

# Fill 'Indicator ID' using mode
df['Indicator ID'] = df['Indicator ID'].fillna(df['Indicator ID'].mode()[0])
# Fill 'Measure Info' with placeholder
df['Measure Info'] = df['Measure Info'].fillna('Not Specified')
# Fill 'Geo Type Name' using mode
df['Geo Type Name'] = df['Geo Type Name'].fillna(df['Geo Type Name'].mode()[0])
# Fill 'Geo Join ID' using forward fill
df['Geo Join ID'] = df['Geo Join ID'].ffill()  
# Fill 'Data Value' using median 
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].median())

print("Missing values after filling:")
print(df.isnull().sum())


#Obj3] Examine Distribution of Features

plt.figure(figsize=(6, 4))
sns.histplot(df['Data Value'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Data Value")
plt.xlabel("Data Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='Time Period', hue='Time Period', data=df, palette='Set2', legend=False)
plt.title("Records Count per Time Period")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#Obj4] Identify Outliers and Unusual Data Points
plt.figure(figsize=(10, 5))
sns.boxplot(x='Geo Type Name', y='Data Value', data=df, palette='pastel', hue='Geo Type Name',legend=False)
plt.title("Data Value by Geo Type Name (Outlier Detection)")
plt.xlabel("Geo Type Name")
plt.ylabel("Data Value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='Measure', y='Data Value', data=df, palette='muted', hue='Measure',legend=False)
plt.title("Distribution and Outliers by Measure (Violin Plot)")
plt.xlabel("Measure")
plt.ylabel("Data Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Obj5] Explore Relationships Between Variables

plt.figure(figsize=(10, 6))
numeric_cols = df.select_dtypes(include='number').columns

corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
plt.show()

#Obj6] Analyze Temporal Trends (Daily/Monthly/Seasonal)
print(df.columns.tolist())

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')

# Drop invalid dates
df = df.dropna(subset=['Start_Date'])

# Extract Month and Year
df['Month'] = df['Start_Date'].dt.month
df['Year'] = df['Start_Date'].dt.year

# Plot average Data Value over time
plt.figure(figsize=(10, 4))
df.groupby('Start_Date')['Data Value'].mean().plot()
plt.title("Average Data Value Over Time")
plt.xlabel("Date")
plt.ylabel("Data Value")
plt.tight_layout()
plt.show()

#Obj 7] Compare Features Across Categories

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')

# Extract Month
df['Month'] = df['Start_Date'].dt.month

# Plot boxplot 
plt.figure(figsize=(8, 5))
sns.boxplot(x='Month', y='Data Value', data=df, hue='Month', palette='pastel', legend=False)
plt.title("Data Value Distribution by Month")
plt.tight_layout()
plt.show()

# Save the cleaned DataFrame
df.to_csv("Cleaned_Air_Quality.csv", index=False)

print("\n data saved successfully to 'Cleaned_Air_Quality.csv'")












