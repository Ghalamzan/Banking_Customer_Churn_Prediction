import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


churn_dataa = pd.read_csv('Churn_Modelling.csv')

churn_data = churn_dataa.drop(columns=['RowNumber','CustomerId','Surname'])
customer_ID= churn_dataa['CustomerId']
#Display all the collumn
pd.set_option('display.max_columns',None)
print(churn_data.head())
print(churn_data.dtypes)

# variable_list=['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']

#modify Data type


churn_data['IsActiveMember']=churn_data['IsActiveMember'].astype('category')
churn_data['Exited']=churn_data['Exited'].astype('category')

#Handling Duplicate Data
duplicate=churn_data.duplicated()
num_duplicates= duplicate.sum()
print (f'number of duplicates are : {num_duplicates}')
#Drop Duplicate in the main Data frame by applying inplace=True
churn_data.drop_duplicates(inplace=True)
number_of_Rows= churn_data.shape[0]
print(f'number of Rows are:{number_of_Rows}')
#Handling missing values
missing_values= churn_data.isnull()
#which Row in the Data frame containt missing value, missing_values.any(axis=1) return the row number which contains missing values
# print('Rows with missing values are :')
# print (churn_data[missing_values.any(axis=1)])
mode_Geoghraphy = churn_data['Geography'].mode()[0]
mean_Age = churn_data['Age'].mean()
mode_HasCrCard = churn_data['HasCrCard'].mode()[0]
mode_IsActiveMember = churn_data['IsActiveMember'].mode()[0]

churn_data.fillna({'Geography': mode_Geoghraphy,
                   'Age': mean_Age,
                   'HasCrCard': mode_HasCrCard,
                   'IsActiveMember': mode_IsActiveMember}, inplace=True)


# missing_values_After_fill = churn_data.isnull()
# print(churn_data[missing_values_After_fill].any(axis=1))
# handling outliers


def remove_outlier(df, treshhold = 1.5):
    numerical_column = churn_data.select_dtypes(include=[np.number]).columns
    for column in numerical_column:
        Q1 = churn_data[column].quantile(.25)
        Q3= churn_data[column].quantile(.75)
        IQR =Q3-Q1
        lower_quartile = Q1 - (treshhold*IQR)
        upper_quartile = Q3 +(treshhold*IQR)
        churn_removed_outlier = churn_data[(churn_data[column]>=lower_quartile) & (churn_data[column]<=upper_quartile)]
        return(churn_removed_outlier)

churn_data_filtered = remove_outlier(churn_data)


# to Retrive the Customer ID to define each customer cluster

customer_ID=customer_ID[churn_data_filtered.index]

number_of_Rows= churn_data_filtered.shape[0]
print(f'number of Rows are:{number_of_Rows}')

# endcoding Data

churn_data_filtered_encoded = pd.get_dummies(churn_data_filtered,columns=["Geography","Gender"],prefix=['GEO','GEN'])
churn_data_filtered_encoded=churn_data_filtered_encoded.astype(int)

# Normalize the dataset
scaler =StandardScaler()
numeric_column = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
churn_data_filtered_encoded[numeric_column]=scaler.fit_transform(churn_data_filtered_encoded[numeric_column])
print(churn_data_filtered_encoded)

column_to_drop = ['GEN_Male', 'GEN_Female', 'GEO_Spain', 'GEO_France', 'GEO_Germany', 'Exited']
churn_data_filtered_encoded = churn_data_filtered_encoded.drop(columns=column_to_drop)
print(churn_data_filtered_encoded.columns)

# To choose the number of Cluster in Kmeans method Elbow Method is used to define the optimum number of clust.
# Within Cluster Sum of Square
wcss =[]
for k in range (2,12):
    kmeans=KMeans(n_clusters=k, random_state= 42)
    kmeans.fit(churn_data_filtered_encoded)
    wcss.append(kmeans.inertia_)
plt.plot(range(2,12),wcss,marker="o",linestyle="--")
plt.title('Figure: Elbow method',fontsize = 15)
plt.xlabel("number of cluster")
plt.ylabel("whithin cluster Sum of Squer")
plt.show()

# number of Cluster = 7
kmeans =KMeans(n_clusters=7,random_state=42)
cluste_label =kmeans.fit_predict(churn_data_filtered_encoded)




df_cluster_label =pd.DataFrame(cluste_label, columns= ["cluster"])
data_with_clusters = pd.concat([customer_ID.reset_index(drop=True), churn_data_filtered_encoded.reset_index(drop=True), df_cluster_label], axis=1)
Cluster_mean = data_with_clusters.groupby('cluster').mean()
print(f'cluster Mean is as follows:\n{Cluster_mean}')
Cluster_Number_of_Customer = data_with_clusters.groupby('cluster').count()
print(f'number of customer in each cluster is:\n {Cluster_Number_of_Customer}')

print(data_with_clusters.head())

kmeans_silhouet_average = silhouette_score(churn_data_filtered_encoded,cluste_label)
print(f'kmeans silouet_Average is :\n{round(kmeans_silhouet_average,2)}')
kmeans_db_score = davies_bouldin_score(churn_data_filtered_encoded,cluste_label)
print(f'Davis Boulding index is :\n{round(kmeans_db_score,2)}')



