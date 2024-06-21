import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objects as go

# Load Bitcoin csv data
df = pd.read_csv('data/BTC-USD.csv',parse_dates=['Date'],index_col='Date')

# Create variable N to predict future N days price.
N = 5
# Adj Close (Adjusted Close Price) can be used for long-term price analysis
df['Prediction'] = df['Adj Close'].shift(-N)

# Split data into features (X) and target (y)
# Feature: use Adjusted Closing Price as train feature
X = df[['Adj Close']][:-N]
# Target: Predict future prices
y = df['Prediction'][:-N]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Target predict on the test sets
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f'Model MSE: {mse}, RMSE: {rmse}, R²: {r2}')

X_projection = df[['Adj Close']][-N:]
# Make predictions on the projection set
y_projection = model.predict(X_projection)
print(f'Linear Regression Model predicted BTC next {N} days price are: {y_projection}')

# Append predict data to existing data sets
data_series = pd.Series(y_projection)
df = pd.concat([df, data_series.to_frame('Prediction')], ignore_index=True)

# Data Visualization
# Draw historical (include predict data) line
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index,y=df['Prediction'],
            mode='lines',
            fill='none',
            showlegend=False,
            line=dict(color='green',dash='solid')))

fig.update_layout(title="Bitcoin Price Data Prediction",
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            autosize=True,
            height=600)
        
fig.show()
