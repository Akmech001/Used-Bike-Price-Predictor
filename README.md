# Used Bike Price Predictor

## Project Overview

The Used Bike Price Predictor is a machine learning project aimed at predicting the prices of used bikes based on various features such as bike name, brand, owner type, power, kilometers driven, city, and age. This project leverages data preprocessing, feature engineering, and linear regression modeling to provide accurate price predictions for used bikes.

## Dataset

The dataset used for this project is `Used_Bikes.csv`, which contains information about various used bikes. The key features in the dataset include:

- `bike_name`: Name of the bike
- `brand`: Brand of the bike
- `owner`: Type of owner (e.g., First Owner, Second Owner)
- `power`: Power of the bike
- `kms_driven`: Kilometers driven
- `city`: City where the bike is being sold
- `age`: Age of the bike
- `price`: Price of the bike (target variable)

## Data Analysis

### Initial Data Exploration

1. **Shape of the Data**: The dataset has a certain number of rows and columns, indicating the size and complexity of the data.
2. **Data Types and Missing Values**: Information about data types and presence of missing values.
3. **Value Counts**: Distribution of categorical variables like `brand`, `owner`, and `city`.

### Data Visualization

- **Brand Distribution**: A bar plot showing the distribution of different bike brands in the dataset.
  
  ```python
  brand_counts = bike1['brand'].value_counts()
  brand_counts_df = brand_counts.reset_index()
  brand_counts_df.columns = ['brand', 'count']
  plt.figure(figsize=(10, 6))
  sns.barplot(x='brand', y='count', data=brand_counts_df)
  plt.xticks(rotation=90)
  plt.xlabel('Brand')
  plt.ylabel('Count')
  plt.title('Distribution of Bike Brands')
  plt.show()
  ```

- **Scatter Plot**: A scatter plot showing the relationship between bike names and their prices.
  
  ```python
  sns.scatterplot(y=bike1['price'], x=bike1['bike_name'])
  ```

### Data Cleaning

- **Removing Outliers**: Removing bikes with prices above a certain threshold to avoid skewing the model.
  
  ```python
  bike1 = bike1[bike1['price'] < 1250000]
  ```

- **Handling Duplicates**: Removing duplicate entries from the dataset.
  
  ```python
  bike1.drop_duplicates(inplace=True)
  ```

- **Saving Cleaned Data**: The cleaned data is saved to `cleaned_bike.csv`.

## Model Building

### Feature Engineering

- **One-Hot Encoding**: Converting categorical features (`bike_name`, `brand`, `city`, `owner`) into numerical format using OneHotEncoder.

### Model Training

- **Train-Test Split**: Splitting the data into training and testing sets.
  
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  ```

- **Linear Regression Model**: Using Linear Regression to train the model.
  
  ```python
  from sklearn.linear_model import LinearRegression
  lr = LinearRegression()
  ```

- **Pipeline Creation**: Creating a pipeline to streamline the preprocessing and model training process.
  
  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.compose import make_column_transformer
  column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['bike_name', 'brand', 'city', 'owner']), remainder='passthrough')
  pipe = make_pipeline(column_trans, lr)
  pipe.fit(X_train, y_train)
  ```

### Model Evaluation

- **R2 Score**: Evaluating the model performance using R2 score.
  
  ```python
  from sklearn.metrics import r2_score
  y_pred = pipe.predict(X_test)
  r2_score(y_test, y_pred)
  ```

### Model Deployment

- **Saving the Model**: Saving the trained model using Pickle.
  
  ```python
  import pickle
  pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
  ```

- **Predicting Prices**: Function to predict the price of a bike based on user input.
  
  ```python
  def predict_price(bike_name, brand, owner, power, kms_driven, city, age):
      data = pd.DataFrame([[bike_name, brand, owner, power, kms_driven, city, age]], columns=['bike_name', 'brand', 'owner', 'power', 'kms_driven', 'city', 'age'])
      predicted_price = pipe.predict(data)
      return predicted_price[0]
  ```

## Usage

You can use the `predict_price` function to predict the price of a used bike by providing the necessary inputs.

```python
predict_price('Royal Enfield Himalayan', 'Royal Enfield', 'First Owner', '441', '16000', 'Chennai', '2')
```

## Conclusion

The Used Bike Price Predictor project demonstrates how machine learning can be applied to predict prices based on historical data. This project includes data cleaning, visualization, feature engineering, model training, and evaluation. The final model can be used to predict bike prices, helping buyers and sellers make informed decisions.
