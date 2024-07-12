import pandas as pd
import matplotlib.pyplot as plt
churn_data=pd.read_csv("Churn_Modelling.csv")
print(churn_data.info())
# correct Data format
churn_data['HasCrCard']=churn_data['HasCrCard'].astype('category')
churn_data['IsActiveMember']=churn_data['IsActiveMember'].astype('category')
churn_data['Exited']=churn_data['Exited'].astype('category')
churn_data['CustomerId']=churn_data['CustomerId'].astype('str')
churn_data['RowNumber']=churn_data['RowNumber'].astype('str')
print(churn_data.dtypes)
#To Display all collumn
pd.set_option('display.max_columns',None)
print(churn_data.describe())
# To Display Variables Histogram

plt.hist(churn_data['CreditScore'], color= 'Orange',bins=10)
plt.title('Histogram of Credit Score')
plt.xlabel("Credit Score")
plt.ylabel("Frequency")
plt.show()

plt.hist(churn_data['Age'],bins=10)
plt.title("Histogram of Age")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.hist(churn_data['Tenure'],bins=10)
plt.title('Histogram of Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()

plt.hist(churn_data["Balance"], bins=10)
plt.title("Histogram of Balance")
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

plt.hist(churn_data['NumOfProducts'],bins=4)
plt.title('Histogram of Number of Product')
plt.xlabel("number of Products")
plt.ylabel('Frequency')
plt.show()

plt.hist(churn_data["EstimatedSalary"],bins=10)
plt.title("Histogram of Estimated Salary")
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()

