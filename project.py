import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df= pd.read_csv("project_dataset.csv")

#getting information about the dataset

print(df.head())
print(df.tail()) 

#describe 
print(df.describe())

#info 
print(df.info())

print("Shape of the dataset: ",df.shape)

# checking for missing values
print("Missing values: ",df.isnull().sum())
print("Unknown values in each column: ")
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, ":", (df[col] == 'unknown').sum())

# target variable (y) analysis
print("Count of yes/no: ",df['y'].value_counts())

# count plot 
sns.countplot(x='y',data=df)
plt.title("Target variable count(yes/no count)")
plt.show()

# histogram
num_cols = ['age','balance','day','campaign','pdays','previous']

for col in num_cols:
    plt.figure()
    plt.hist(df[col],bins=30,color='lightgreen',edgecolor='black') 
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()


# boxplot
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='y',y=col,data=df)
    plt.title(f"{col} by target (y)")
    plt.tight_layout()
    plt.show()

# pie chart
df['y'].value_counts().plot.pie(autopct = '%1.1f%%',colors=['skyblue','lightcoral'],startangle= 90)
plt.title("Target variable distribution (y)")
plt.ylabel('')
plt.show()
print(df['job'].nunique())

# pie chart comparison of housing loan vs personal loan
housing_loan_count = (df['housing loan']=='yes').sum()
personal_loan_count = (df['Personal loan']=='yes').sum()

plt.figure()
plt.pie([housing_loan_count,personal_loan_count],labels= ['Housing_loans','Personal_loans'],autopct = '%1.1f%%',startangle=90,colors=['skyblue','salmon'])
plt.title("Comparison of Housing Loan vs Personal Loan ")
plt.show()