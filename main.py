import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
from scipy.stats import t


# Function to load and clean data, then transpose
def load_clean_transpose_data(file_path):
    """
        Load, clean, and transpose data from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file containing the data.

        Returns:
        - df (pd.DataFrame): Original DataFrame with cleaned and imputed data.
        - transposed_df (pd.DataFrame): Transposed DataFrame.
    """
    df = pd.read_csv(file_path , header = None , names = ["Year" , "Time Code" , "Country Name" , "Country Code" ,
                                                    "Access to Electricity" , "Coal Rents (% of GDP)" ,
                                                    "Mineral Rents (% of GDP)" , "Natural Gas Rents (% of GDP)"])

    # Convert numeric columns to numeric values
    numeric_columns = ["Access to Electricity" , "Coal Rents (% of GDP)" , "Mineral Rents (% of GDP)"
        , "Natural Gas Rents (% of GDP)"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric , errors = 'coerce')

    # Impute missing values with mean
    imputer = SimpleImputer(strategy = 'mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Transpose the data
    transposed_df = df.transpose()

    return df , transposed_df


# Function for quadratic model
def model_func(x , a , b , c):
    """
        Compute the output of a quadratic function.

        Parameters:
        - x (float or array-like): Input value or array of input values.
        - a (float): Coefficient of the quadratic term.
        - b (float): Coefficient of the linear term.
        - c (float): Constant term.

        Returns:
        - float or array-like: Output value(s) of the quadratic function.
    """
    return a * x ** 2 + b * x + c


# Function to calculate confidence interval
def err_ranges(cov_matrix , x , model_func , params , confidence = 0.95):
    """
        Calculate confidence intervals for the predicted values of a model.

        Parameters:
        - cov_matrix (numpy.ndarray): Covariance matrix of the model parameters.
        - x (array-like): Input values for the model.
        - model_func (callable): Model function that takes 'x' and 'params' as arguments.
        - params (array-like): Model parameters.
        - confidence (float, optional): Confidence level for the intervals (default is 0.95).

        Returns:
        - lower_bound (numpy.ndarray): Lower bounds of the confidence intervals for each predicted value.
        - upper_bound (numpy.ndarray): Upper bounds of the confidence intervals for each predicted value.
    """
    alpha = 1 - confidence
    n = len(x)
    df = max(0 , n - len(params))
    t_val = np.abs(t.ppf(alpha / 2 , df))

    predicted_values = model_func(x , *params)
    residuals = x - predicted_values
    resid_std = np.std(residuals , ddof = len(params))
    margin_of_error = t_val * resid_std
    lower_bound = predicted_values - margin_of_error
    upper_bound = predicted_values + margin_of_error

    return lower_bound , upper_bound


# Function for K-Means clustering and visualization
def perform_kmeans_clustering(df , features , k = 3):
    df_features = df[features]

    # Normalize the data
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df_features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters = k , random_state = 42)
    df["Cluster"] = kmeans.fit_predict(df_normalized)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(df_normalized , df["Cluster"])
    print(f"Silhouette Score: {silhouette_avg}")

    # Explore characteristics of each cluster
    cluster_characteristics = df.groupby("Cluster")[features].mean().reset_index()
    print("Cluster Characteristics:")
    print(cluster_characteristics)

    # Inverse transform the cluster centers to the original scale
    original_scale_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Plotting the clusters and cluster centers
    plt.figure(figsize = (10 , 6))
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        plt.scatter(cluster_data["Access to Electricity"] ,
                    cluster_data["Coal Rents (% of GDP)"] ,
                    label = f'Cluster {cluster}' , alpha = 0.7)

    plt.scatter(original_scale_centers[: , 0] , original_scale_centers[: , 1] ,
                marker = 'X' , s = 200 , c = 'red' ,
                label = 'Cluster Centers (Original Scale)')
    plt.title("K-Means Clustering of Energy and Economic Data" , fontsize = 20)
    plt.xlabel("Access to Electricity" , fontsize = 20)
    plt.ylabel("Coal Rents (% of GDP)" , fontsize = 20)
    plt.legend()
    plt.show()


# Function for time series analysis and prediction visualization
def perform_time_series_analysis(df , country , target_feature):
    """
        Perform time series analysis on a DataFrame for a specific country and target feature.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the time series data.
        - country (str): The name of the country for analysis.
        - target_feature (str): The target feature for analysis.

        Returns:
        - None: Displays a plot of the actual data, the best-fitting function,
        and confidence intervals.
    """
    country_df = df[(df['Country Name'] == country) & (pd.to_numeric(df['Year'] ,
                                                                     errors = 'coerce') <= 2030)]

    years = country_df['Year'].astype(float)
    target_feature_values = country_df[target_feature].astype(float)

    params , covariance = curve_fit(model_func , years , target_feature_values)
    future_years = np.arange(2000 , 2031 , 1)
    predicted_values = model_func(future_years , *params)

    lower_bound , upper_bound = err_ranges(covariance , future_years , model_func , params ,
                                           confidence = 0.95)

    # Plotting
    plt.figure()
    plt.scatter(years , target_feature_values , label = 'Actual Data')
    plt.plot(future_years , predicted_values , label = 'Best Fitting Function' , color = 'red')
    plt.fill_between(future_years , lower_bound , upper_bound , color = 'red' , alpha = 0.2 ,
                     label = 'Confidence Interval (95%)')
    plt.xlabel('Year' , fontsize = 20)
    plt.ylabel(f'{target_feature} - {country}' , fontsize = 20)
    plt.title(f'{target_feature} Prediction with Confidence Interval - {country}' , fontsize = 20)
    plt.legend()
    plt.show()


# Main part of the script
data_path = '5d0592c2-cfe0-49f4-b938-e718c4a489cd_Data.csv'
original_data , cleaned_data = load_clean_transpose_data(data_path)

# K-Means Clustering
cluster_features = ["Access to Electricity" , "Coal Rents (% of GDP)" ,
                    "Mineral Rents (% of GDP)" , "Natural Gas Rents (% of GDP)"]
perform_kmeans_clustering(original_data , cluster_features)

# Time Series Analysis and Prediction Visualization
selected_countries = ['India' , 'Australia' , 'Canada' , 'China']
target_feature = 'Access to Electricity'  # Replace with the feature you want to predict

for country in selected_countries:
    perform_time_series_analysis(original_data , country , target_feature)
