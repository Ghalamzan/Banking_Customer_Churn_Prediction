# Customer Churn Prediction in Banking

In the banking sector, customer churn can lead to a significant revenue loss of around 30–40%. This study aims to predict customer churn by analyzing banking customer behavior using clustering and classification techniques, providing insights that can help minimize churn.

## Repository Structure

```sh
│
├── Classification
│   └── (files related to classification tasks)
├── clustering
│   └──
│   └──
│   └──   
├── Churn_Modelling.csv
├── Data_Preprocessing.py
├── README.md
└── statistics_analysis.py
```



## Dataset

The dataset used in this study is available in the repository as `Churn_Modelling.csv`. Sourced from Kaggle, it includes 10,027 entries. After addressing missing and duplicate values, 9,985 observations were used post-outlier handling.

## Data Preprocessing

- **Handling Missing and Duplicate Values**: Filled missing values and removed duplicates.
- **Outlier Handling**: Addressed outliers to ensure data quality.
- **Correlation Analysis**: Conducted pairwise Pearson correlation, finding no significant associations among numerical variables.
- **Standardization**: Applied Z-score standardization for data scalability.

## Features

The study focuses on two key sets of attributes derived from literature:

### Demographic Attributes

- Age
- Estimated Salary

### Banking Engagement Attributes

- Tenure
- Balance
- Number of Products
- Credit Score
- Has Credit Card
- Is Active Member

## Clustering Methods

Various clustering methods were used to categorize customers:

- k-means
- PAM
- DBSCAN
- OPTICS

The OPTICS method was found unsuitable as it allocated most objects to a single cluster. The Agglomerative method was preferred based on the Silhouette Score and Davies-Bouldin Index.

## Sample Image

![This is a sample caption](image/images.png)

## Classification Methods

For predicting churn, the following classification methods were compared:

- Decision Tree
- K-Nearest Neighbors (KNN)
- Naïve Bayes

The decision tree outperformed KNN and Naïve Bayes in predicting churn, with the most important features being Age, Balance, and Estimated Salary.

## Results

- **Clustering**: The Agglomerative method was the most effective based on the Silhouette Score and Davies-Bouldin Index.
- **Classification**: The decision tree provided the best performance, with Age, Balance, and Estimated Salary being the most influential features.

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

This study provides insights into banking customer behaviour using clustering and classification techniques. The findings can help implement strategies to minimize customer churn, reducing potential revenue loss and margin erosion.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

For more information, please contact [ghalamzan6286@gmail.com](mailto:ghalamzan6286@gmail.com).
