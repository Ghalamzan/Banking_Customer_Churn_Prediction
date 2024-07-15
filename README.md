# Customer Churn Prediction in Banking

In the context of banking systems, where customer churn can lead to a potential loss of around 30–40% of revenues and margin erosion, this study delves into banking customer behavior. It employs clustering and classification techniques to predict churn, aiming to facilitate in-depth analysis and future initiatives aimed at minimizing churn.

## Dataset

The dataset used in this study is available in the GitHub repository under the name `Churn_Modeling.CSV`. This dataset, sourced from Kaggle, includes a total of 10,027 entries. After addressing missing and duplicate values, 9,985 observations were considered post-outlier handling.

## Data Preprocessing

- **Handling Missing and Duplicate Values**: Missing values were filled, and duplicates were removed.
- **Outlier Handling**: Outliers were addressed to ensure data quality.
- **Correlation Analysis**: Pairwise Pearson correlation showed no significant associations among numerical variables.
- **Standardization**: Z-score standardization was employed for data scalability.

## Features

The study focuses on two key attribute sets derived from literature:

### Demographic Attributes:

- Age
- Estimated Salary

### Banking Engagement Attributes:

- Tenure
- Balance
- Number of Products
- Credit Score
- Has Credit Card
- Is Active Member

## Clustering Methods

To categorize customers into distinct groups, various clustering methods were employed:

- k-means
- PAM
- DBSCAN
- OPTICS

The application of OPTICS revealed its unsuitability for the dataset, as the majority of objects were allocated to a single cluster. Based on the Silhouette Score and Davies-Bouldin Index, the Agglomerative method was favored over k-means and DBSCAN.

## Classification Methods

In the classification task, the following methods were compared:

- Decision Tree
- K-Nearest Neighbors (KNN)
- Naïve Bayes

The decision tree outperformed KNN and Naïve Bayes in predicting churn based on F1 Score and ROC AUC. The most important features in classifying customer churn were Age, Balance, and Estimated Salary.

## Results

- **Clustering**: The Agglomerative method was found to be the most effective based on the Silhouette Score and Davies-Bouldin Index.
- **Classification**: The decision tree provided the best performance in predicting churn, with Age, Balance, and Estimated Salary being the most influential features.

## Usage

To use this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Customer_Churn_Prediction.git
    ```
2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the analysis**:
    ```bash
    python analysis.py
    ```

## Conclusion

This study provides insights into banking customer behavior using clustering and classification techniques. The findings can be used to implement strategies aimed at minimizing customer churn, thereby reducing potential revenue loss and margin erosion.


## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

For more information, please contact `ghalamzan6286@gmail.com`.
