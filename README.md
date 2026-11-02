# 🚗 Used Car Price Prediction | Machine Learning & Model Benchmarking

### Comparing Regression Models for Accurate and Efficient Vehicle Price Estimation

**Python • Pandas • NumPy • Scikit-learn • LightGBM • CatBoost • Regression • Model Evaluation**

## 📌 Project Overview

Estimating the market value of a used car is an important challenge for online automotive platforms. Accurate predictions can improve the customer experience and help users make informed buying and selling decisions.

This project was developed for **Rusty Bargain**, a fictional used-car sales platform creating an application that estimates vehicle market prices.

Using historical vehicle listings, the project trains and compares five regression models to predict car prices.

The analysis evaluates three important criteria:

- **Prediction quality:** How accurately does each model estimate vehicle prices?
- **Training speed:** How long does each model take to train?
- **Prediction speed:** How quickly can each model generate estimates?

The goal is to identify a model that offers a practical balance between accuracy and computational efficiency.

## 🎯 Project Objectives

- Explore and preprocess historical vehicle data.
- Identify and handle missing values and duplicate records.
- Analyze the distribution of vehicle prices.
- Prepare numerical and categorical features for machine learning.
- Train and compare multiple regression algorithms.
- Evaluate prediction quality using Root Mean Squared Error (RMSE).
- Measure model training and prediction times.
- Recommend a suitable model for the used-car pricing application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and machine learning |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Regression models, preprocessing, and evaluation |
| LightGBM | Gradient boosting regression |
| CatBoost | Gradient boosting regression |
| Jupyter Notebook | Analysis and documentation |

## 📂 Dataset Description

The project uses `car_data.csv`, containing historical used-car listings.

The dataset includes vehicle characteristics, listing information, and selling prices.

Important features include:

| Feature | Description |
|---|---|
| `Price` | Vehicle price (target variable) |
| `VehicleType` | Vehicle body type |
| `RegistrationYear` | Vehicle registration year |
| `Gearbox` | Transmission type |
| `Power` | Engine power |
| `Model` | Vehicle model |
| `Mileage` | Vehicle mileage |
| `RegistrationMonth` | Registration month |
| `FuelType` | Fuel type |
| `Brand` | Vehicle manufacturer |
| `NotRepaired` | Repair history indicator |

Additional listing-related columns were inspected during preprocessing.

## 🧹 Data Preparation

### 1. Initial Data Exploration

The dataset was examined using Pandas methods to understand its structure, variable types, descriptive statistics, and potential data quality issues.

### 2. Handling Missing Values

Missing values in selected categorical features were replaced with `"Unknown"`.

The affected columns included vehicle type, gearbox, model, fuel type, and repair history.

### 3. Removing Duplicates

Duplicate records were identified and removed to reduce redundant information.

### 4. Feature Selection

The following columns were excluded from model training:

- `DateCrawled`
- `DateCreated`
- `LastSeen`
- `PostalCode`
- `NumberOfPictures`

### 5. Categorical Encoding

Categorical features were transformed into numerical values using `OrdinalEncoder`.

### 6. Train-Test Split

The dataset was divided into:

- 75% training data
- 25% testing data

A fixed random state was used for reproducibility.

## 🤖 Machine Learning Models

Five regression algorithms were evaluated.

### 1. Linear Regression

Used as a simple baseline model to establish an initial benchmark for prediction accuracy and computational speed.

### 2. Decision Tree Regressor

A nonlinear regression model capable of capturing more complex relationships between vehicle characteristics and prices.

Different maximum tree depths were evaluated.

### 3. Random Forest Regressor

An ensemble algorithm that combines multiple decision trees to improve predictive accuracy.

The experiment tested different combinations of tree counts and maximum depths.

### 4. LightGBM Regressor

A gradient boosting algorithm designed for efficient training and strong predictive performance.

The notebook used 200 estimators and a learning rate of 0.1.

### 5. CatBoost Regressor

A gradient boosting algorithm evaluated for its prediction accuracy and execution speed.

The notebook used 300 iterations, a learning rate of 0.1, and a maximum depth of 8.

## 📊 Model Evaluation

### Evaluation Metric: RMSE

Root Mean Squared Error measures the average magnitude of prediction errors while penalizing larger errors more heavily.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Lower RMSE values indicate better predictive accuracy.

Training and prediction times were also recorded to evaluate computational efficiency.

## 📈 Model Comparison Results

| Model | RMSE | Training Time (s) | Prediction Time (s) |
|---|---:|---:|---:|
| Linear Regression | 4,097.19 | 0.048 | 0.003 |
| Decision Tree | 2,072.50 | 0.625 | 0.012 |
| **Random Forest** | **1,754.00** | 57.816 | 2.267 |
| LightGBM | 1,800.20 | 3.674 | 0.427 |
| CatBoost | 1,803.32 | 10.398 | 0.011 |

*These results are reported from the original notebook execution. Execution times may vary depending on hardware and software versions.*

### Key Observations

**Linear Regression**

Achieved the fastest training time but produced the highest prediction error.

**Decision Tree**

Substantially improved accuracy compared with Linear Regression while maintaining relatively low execution times.

**Random Forest**

Achieved the lowest RMSE of all evaluated models, but required the longest training and prediction times.

**LightGBM**

Produced predictions with an RMSE close to Random Forest while training approximately 16 times faster.

**CatBoost**

Delivered competitive accuracy and particularly fast prediction times in the recorded experiment.

## 🏆 Final Model Recommendation

**LightGBM is the recommended model for the Rusty Bargain application based on the reported accuracy and computational performance.**

The model achieved:

- RMSE: **1,800.20**
- Training time: **3.674 seconds**
- Prediction time: **0.427 seconds**

Although Random Forest achieved slightly better prediction accuracy, LightGBM offered substantially faster training and prediction.

CatBoost also emerged as a competitive alternative, particularly when prediction latency is the primary consideration.

## 💡 Business Impact

The project demonstrates how machine learning can support automotive pricing applications.

Potential benefits include:

**Automated Price Estimation**

Generate estimated market prices using vehicle characteristics.

**Improved Customer Experience**

Help users evaluate used-car listings more efficiently.

**Operational Efficiency**

Compare model speed alongside predictive accuracy to identify practical deployment candidates.

**Data-Driven Pricing**

Use historical listing data to support consistent price estimates.

These are potential applications; the notebook does not evaluate a deployed production system.

## ⚠️ Project Limitations

- The dataset represents historical vehicle listings.
- Prediction quality depends on the completeness and accuracy of vehicle information.
- The notebook uses ordinal encoding for categorical variables, which may introduce artificial numerical ordering.
- Hyperparameter selection was performed using the same test partition later used for evaluation, so the reported performance may be optimistic.
- A separate validation set or cross-validation procedure would strengthen model selection.
- Training and prediction times depend on the execution environment.
- Additional testing would be necessary before production deployment.

## 🚀 How to Run the Project

1. Clone or download the repository.
2. Install the required Python libraries.
3. Place `car_data.csv` inside a local `data/` directory.
4. Update the dataset path in the notebook.
5. Open the notebook in Jupyter Notebook or JupyterLab.
6. Execute the cells sequentially.

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm catboost notebook
```

## 🔮 Future Improvements

- Introduce separate training, validation, and test datasets.
- Implement cross-validation for model selection.
- Use a preprocessing pipeline to avoid data leakage.
- Experiment with alternative categorical encoding strategies.
- Apply systematic hyperparameter optimization.
- Evaluate feature importance and model interpretability.
- Investigate potential outliers and unrealistic vehicle records.
- Develop an interactive application for real-time price estimation.

## 📈 Conclusion

This project demonstrates how machine learning can be applied to predict used-car prices while considering both predictive accuracy and computational efficiency.

Five regression models were compared using RMSE, training time, and prediction time.

Random Forest achieved the lowest prediction error, while LightGBM provided an attractive balance between accuracy and execution speed.

The project showcases practical skills in **data preprocessing, regression modeling, ensemble learning, model evaluation, and performance benchmarking**.

## 👩‍💻 Author

**Andressa Miranda**

Computer Science | Data Analytics | Python | Machine Learning

[GitHub Profile](https://github.com/Andressa202)

*Developed as part of the TripleTen Data Science program.*
