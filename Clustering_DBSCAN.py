
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the data
churn_dataa = pd.read_csv('Churn_Modelling.csv')
churn_data = churn_dataa.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
pd.set_option('display.max_columns', None)
print(churn_data.head())
print(churn_data.dtypes)

# Modify data types
churn_data['IsActiveMember'] = churn_data['IsActiveMember'].astype('category')
churn_data['Exited'] = churn_data['Exited'].astype('category')

# Handle duplicates
churn_data.drop_duplicates(inplace=True)
print(f'Number of Rows are: {churn_data.shape[0]}')

# Handle missing values
mode_Geoghraphy = churn_data['Geography'].mode()[0]
mean_Age = churn_data['Age'].mean()
mode_HasCrCard = churn_data['HasCrCard'].mode()[0]
mode_IsActiveMember = churn_data['IsActiveMember'].mode()[0]

churn_data.fillna({'Geography': mode_Geoghraphy,
                   'Age': mean_Age,
                   'HasCrCard': mode_HasCrCard,
                   'IsActiveMember': mode_IsActiveMember}, inplace=True)

# Remove outliers
def remove_outlier(df, treshhold=1.5):
    numerical_column = df.select_dtypes(include=[np.number]).columns
    for column in numerical_column:
        Q1 = df[column].quantile(.25)
        Q3 = df[column].quantile(.75)
        IQR = Q3 - Q1
        lower_quartile = Q1 - (treshhold * IQR)
        upper_quartile = Q3 + (treshhold * IQR)
        df = df[(df[column] >= lower_quartile) & (df[column] <= upper_quartile)]
    return df

churn_data_filtered = remove_outlier(churn_data)
print(f'Number of Rows after removing outliers: {churn_data_filtered.shape[0]}')

# Encoding categorical variables
churn_data_filtered_encoded = pd.get_dummies(churn_data_filtered, columns=["Geography", "Gender"], prefix=['GEO', 'GEN'])

# Normalize the dataset
scaler = StandardScaler()
numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
churn_data_filtered_encoded[numeric_columns] = scaler.fit_transform(churn_data_filtered_encoded[numeric_columns])
print(churn_data_filtered_encoded.head())

# Drop unnecessary columns
column_to_drop = ['GEN_Male', 'GEN_Female', 'GEO_Spain', 'GEO_France', 'GEO_Germany', 'Exited']
churn_data_filtered_encoded = churn_data_filtered_encoded.drop(columns=column_to_drop)
print(churn_data_filtered_encoded.columns)

# DBSCAN clustering
Eps = [1,1.5, 2]  # Max distance between two samples
MinPts = [ 5, 7]  # Min number of samples in the neighborhood
cluster_counts = {}
results = []

for minPts in MinPts:
    for eps in Eps:
        dbscan = DBSCAN(eps=eps, min_samples=minPts)
        cluster_labels = dbscan.fit_predict(churn_data_filtered_encoded)
        churn_data_filtered_encoded[f'cluster_eps_{eps}_min_samples_{minPts}'] = cluster_labels

        # Count unique cluster labels (-1 is noise points)
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        # Identify core points
        core_points = np.zeros_like(cluster_labels, dtype=bool)
        core_points[dbscan.core_sample_indices_] = True

        # Identify border points (not noise and not core)
        border_points = (cluster_labels != -1) & ~core_points

        # Count number of core points and border points
        num_core_points = np.sum(core_points)
        num_border_points = np.sum(border_points)

        # the result
        results.append({
            'Eps': eps,
            'MinPts': minPts,
            'Num_Core_Points': num_core_points,
            'Num_Border_Points': num_border_points
        })

        # Store the result in the dictionary
        cluster_counts[f'Eps_{eps}_MinPts_{minPts}'] = unique_clusters

# Print the number of clusters for each combination of parameters
for key, value in cluster_counts.items():
    print(f'{key}: {value} clusters')

# Create DataFrame from results
results_df = pd.DataFrame(results)
print(results_df)


eps = 1
minPts = 7
dbscan = DBSCAN(eps=eps, min_samples=minPts)
cluster_labels = dbscan.fit_predict(churn_data_filtered_encoded)
churn_data_filtered_encoded['cluster_labels'] = cluster_labels

# Identify core points
core_points = np.zeros_like(cluster_labels, dtype=bool)
core_points[dbscan.core_sample_indices_] = True

# Identify noise and border points
noise_points = (cluster_labels == -1)
border_points = ~core_points & ~noise_points

# Create a DataFrame to summarize the results
summary_df = churn_data_filtered_encoded.copy()
summary_df['core_point'] = core_points
summary_df['border_point'] = border_points

# Group by cluster labels
grouped_summary = summary_df.groupby('cluster_labels').agg(
    num_core_points=('core_point', 'sum'),
    num_border_points=('border_point', 'sum')
).reset_index()

# Count noise points separately
num_noise_points = noise_points.sum()
noise_summary = pd.DataFrame({
    'cluster_labels': [-1],
    'num_core_points': [0],
    'num_border_points': [num_noise_points]
})

# Combine the summaries
final_summary = pd.concat([grouped_summary, noise_summary], ignore_index=True)

# Print the final summary
print(f'for EPS=1 and Minpts= 7 the final clusters number of core and border points are:\n {final_summary}')

