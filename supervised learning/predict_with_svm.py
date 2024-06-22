import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data/BTC-USD.csv',parse_dates=['Date'],index_col='Date')

# Calculate daily logarithmic returns (replace with your chosen features)
data['Log_Return'] = np.log(data['Close']) - np.log(data['Open'])

# Drop rows with missing values (consider alternatives)
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low']][:-1]
# Define and fit the scaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

# Predict next day close price
target_variable = data['Close'][:-1]  # Shift target by 1 for prediction

X_train, X_test, y_train, y_test = train_test_split(scaled_features, target_variable, test_size=0.2, random_state=42)

# Define and train the SVR model
model = SVR(kernel='rbf', C=191, gamma=0.1)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the testing set
y_predicted = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print(f"Model1 Mean Squared Error: {mse:.2f}")
print(f"Model1 R-squared: {r2:.6f}")

# Prediction on new unseen data (replace with your new data)
X_projection = data[['Open', 'High', 'Low']][-1:]
scaled_X = scaler.transform(X_projection)
predict_price = model.predict(scaled_X)
print(f"Model1 predict BTC next day price: {predict_price}")

# Use GridSearchCV to grab best parameters
c_range = [i for i in range(1, 200, 10)]
gamma_range = [0.1, 0.3, 0.5, 0.7, 0.9]
epsilon_range = [0.01, 0.1, 1.0]
kernel = ['linear', 'rbf']
param_grid = dict(gamma=gamma_range, C=c_range, kernel=kernel,epsilon=epsilon_range)

grid = GridSearchCV(model, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("best parameters: ", grid.best_params_)
print("best accuracy: ", grid.best_score_)

# [Output]
# best parameters:  {'C': 191, 'epsilon': 1.0, 'gamma': 0.1, 'kernel': 'linear'}
# best accuracy:  0.9983971722785311

# Create model with given best parameters
model2 = SVR(kernel='linear', C=191, gamma=0.1, epsilon = 1.0)
model2.fit(X_train, y_train)

y_predicted = model2.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print(f"Model2 Mean Squared Error: {mse:.2f}")
print(f"Model2 R-squared: {r2:.6f}")

predict_price = model2.predict(scaled_X)
print(f"Model2 predicted BTC next day price: {predict_price}")
