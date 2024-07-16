import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve , f1_score
from sklearn.neighbors import KNeighborsClassifier


churn_dataa = pd.read_csv('Churn_Modelling.csv')
churn_data = churn_dataa.drop(columns=['RowNumber','CustomerId','Surname'])
customer_ID = churn_dataa['CustomerId']

#Display all the collumn
pd.set_option('display.max_columns',None)
print(churn_data.head())
print(churn_data.dtypes)

# variable_list=['RowNumber','CustomerId','Surname','CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']

#modify Data type


# churn_data['IsActiveMember']=churn_data['IsActiveMember'].astype('category')
# churn_data['Exited']=churn_data['Exited'].astype('category')

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

customer_ID = customer_ID[churn_data_filtered.index]

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

column_to_drop = ['GEN_Male', 'GEN_Female', 'GEO_Spain', 'GEO_France', 'GEO_Germany']
churn_data_filtered_encoded = churn_data_filtered_encoded.drop(columns=column_to_drop)
print(churn_data_filtered_encoded.columns)

X = churn_data_filtered_encoded.drop('Exited', axis=1)
y= churn_data_filtered_encoded["Exited"]

#KNN Classifier
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size= .2 , random_state= 42 )

Number_of_neighbour = [4, 7,10]
Accuracy = {}
Precision ={}
Recall = {}
ROC_Curve ={}
F_Score ={}
for k in Number_of_neighbour:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_predict)
    precision= precision_score(y_test,y_predict)
    recall = recall_score(y_test,y_predict)
    roc = roc_auc_score(y_test,y_predict)
    f_Score= f1_score(y_test,y_predict)
    Accuracy[k]= round(accuracy,2)
    Precision[k] =round(precision,2)
    Recall[k] = round(recall,2)
    ROC_Curve[k]= round(roc,2)
    F_Score [k]=round(f_Score,2)
    conf_matrix = confusion_matrix(y_test,y_predict, labels=knn.classes_)
    disp= ConfusionMatrixDisplay( confusion_matrix=conf_matrix,display_labels=knn.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


print(f'Accuracy is :\n {Accuracy}')
print(f'Precision is :\n {{4: {Precision[4]}, 7: {Precision[7]}, 10: {Precision[10]}}}')
print(f'Recall is :\n {{4: {Recall[4]}, 7: {Recall[7]}, 10: {Recall[10]}}}')
print(f'ROC_Curve is :\n {{4: {ROC_Curve[4]}, 7: {ROC_Curve[7]}, 10: {ROC_Curve[10]}}}')
print(f'F_Score is :\n {{4: {F_Score[4]}, 7: {F_Score[7]}, 10: {F_Score[10]}}}')