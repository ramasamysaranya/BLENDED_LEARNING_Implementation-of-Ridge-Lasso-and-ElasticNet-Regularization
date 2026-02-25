<img width="1292" height="530" alt="image" src="https://github.com/user-attachments/assets/18a981bd-f109-4547-81e5-1607a80c496e" /># BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the car price dataset, select relevant numerical features (enginesize, horsepower, citympg, highwaympg) as input variables, and set price as the target variable. Split the data into training and testing sets.
2. Apply standardization to the training features using StandardScaler and transform the testing features using the same scaler to ensure consistent feature scaling.
3. Train a Linear Regression model using the scaled training data, predict prices for the test data, and evaluate model performance using MSE, RMSE, and R-squared metrics along with model coefficients.
4. Check linearity using actual vs predicted plots, test independence of errors using the Durbin–Watson statistic, assess homoscedasticity through residual plots, and verify normality of residuals using histogram and Q–Q plots.


## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("encoded_car_data (1).csv")
data.head()
df = pd.get_dummies(data, drop_first=True)

X = data.drop('price',axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results ={}

for name,model in models.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])

pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test,pred)

results[name] = {'MSE' : mse, 'MAE' : mae, 'Rscore': r2}

print("Name: SARANYA R")
print("Reg. No: 212225040384")
for model_name, metrics in results.items():
    print(f"{model_name} - \nMean Squared Error: {metrics['MSE']:.2f}, \nMean Absolute Error: {metrics['MAE']:.2f}, \nR Squared Score: {metrics['Rscore']:.2f}")

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'},inplace=True)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='Model',y='MSE',data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.barplot(x='Model',y='Rscore', data=results_df,palette='viridis')
plt.title('R Squared Score')
plt.ylabel('R Squared Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Output:
<img width="322" height="142" alt="image" src="https://github.com/user-attachments/assets/aef59b1e-015f-4777-af71-58b85ddc45ce" />
<img width="1292" height="530" alt="image" src="https://github.com/user-attachments/assets/f63b2b9f-d815-4d53-a796-7315d72f9f43" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
