import pandas as pd
import numpy as np

from pandas import DataFrame
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import rgb2hex
import plotly.graph_objects as go

# Load Bitcoin csv data
filename = 'data/BTC-USD.csv'
df = pd.read_csv(filename,parse_dates=['Date'],index_col='Date')

display(Markdown(f"### DataFrame "))
display(df.head())
print(df.info())

def check_format_consistent(df: DataFrame) -> None:
    """
    Check and display format consistent for a given DataFrame.

    Parameters:
        df: A given DataFrame.

    """
    display(Markdown("### Data Format Consistent:"))
    # Check data types
    display(Markdown(f"- Data Types: {df.dtypes}"))
    # Check for missing values
    display(Markdown(f"- Null Values: {df.isna().sum()}"))
    # Check for unique values
    display(Markdown(f"- Unique Values: {df.nunique()}"))

# check_format_consistent(df)

def detect_outlier(df: DataFrame) -> dict:
    outlier_info = {}
    # Pick numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Initialize outlier data frames
    num_outliers = {}
    pct_outliers = {}

    # Data statistics summary
    desc = df.describe()
    for col in num_cols:
        # Get quartiles and IQR
        q1 = desc.loc['25%', col]
        q3 = desc.loc['75%', col]
        iqr = q3 - q1
        
        # Define the outlier range
        outlier_range = 1.5 * iqr

        # Count the number of outliers
        num_outliers[col] = ((df[col] > q3 + outlier_range) | (df[col] < q1 - outlier_range)).sum()

        # Compute the percentage of outliers
        percentage = round((num_outliers[col] / len(df[col])), 2) * 100
        pct_outliers[col] = f"{percentage}%"

    outlier_info = {
        'numerical_outliers': num_outliers,
        'percent_outliers': pct_outliers
    }
    return outlier_info

outlier_info = detect_outlier(df)
print(outlier_info)

import matplotlib.pyplot as plt
import seaborn as sns
def display_correlation_heatmap(df: DataFrame):
    # Pick numerical data
    numeric_data = df.select_dtypes(include=[np.number])
    # Get features correlation
    corr_matrix = numeric_data.corr()

    # Find the most correlated pair features
    corr_matrix_value = corr_matrix.mask(corr_matrix == 1.0).stack().idxmax()
    print(f'The most correlated feature pair is {corr_matrix_value}, with the value of {corr_matrix.loc[corr_matrix_value]} ')

    # Plot correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',fmt='.2f')
    plt.title('Bitcoin Features Correlation Heatmap')
    plt.show()

display_correlation_heatmap(df)

from matplotlib.colors import rgb2hex
import plotly.graph_objects as go
def plot_prices(features:list[str]):
    # Define a color palette for the features
    palette = sns.color_palette('Paired', n_colors=len(features))
    hex_palette = [rgb2hex(color) for color in palette]
    # Define line style for the features
    dashes = ['solid','dash', 'dot', 'dashdot']
    fig = go.Figure()

    # Function to create and show plots for a given feature
    for feature, color, dash in zip(features, hex_palette, dashes):
        fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature, line=dict(color=color,dash=dash)))
    
    fig.update_layout(
            title="Bitcoin OHLC Prices Over Time",
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            autosize=True,
            height=600,
        )
        
    fig.show()
# OHLC Features
features = ["Open","High","Low","Close"]
plot_prices(features)

from scipy.stats import linregress
import plotly.express as px

# Function to calculate regression statistics
def calculate_regression_stats(x, y):
    """
    Calculate regression statistics for the given data.

    This function computes the slope, intercept, coefficient of determination (R-squared),
    p-value, and standard error of the regression line for the provided x and y data.

    Parameters:
    x (array-like): The independent variable data.
    y (array-like): The dependent variable data.

    Returns:
    dict: A dictionary containing the following keys and their corresponding values:
        - 'slope': The slope of the regression line.
        - 'intercept': The intercept of the regression line.
        - 'r_squared': The coefficient of determination (R-squared) value.
        - 'p_value': The p-value for the slope.
        - 'std_err': The standard error of the regression line.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

# Define meaningful pairs of numerical features
feature_pairs = [('Open', 'Close')]

# Create scatter plots for each pair of numerical features for each selected coin
for feature_x, feature_y in feature_pairs:
    # Filter data for the specific coin
    coin_data = df

    # Create scatter plot
    fig = px.scatter(
        coin_data, x=feature_x, y=feature_y, title=f'{feature_x} vs {feature_y} (Bitcoin)',
        labels={feature_x: feature_x, feature_y: feature_y},
        template='plotly_dark', opacity=0.5
    )

    # Calculate regression statistics
    stats = calculate_regression_stats(coin_data[feature_x], coin_data[feature_y])

    # Conditionally add the regression line if R² is above a threshold and p-value is below a threshold
    if stats['r_squared'] > 0.5 and stats['p_value'] < 0.05:
        fig.add_trace(
            go.Scatter(
                x=coin_data[feature_x], y=stats['slope']*coin_data[feature_x] + stats['intercept'],
                mode='lines', name=f"y = {stats['slope']:.2f}x + {stats['intercept']:.2f}",
                line=dict(color='red')
            )
        )
    else:
        print(f"The relationship between {feature_x} and {feature_y} for Bitcoin is not significant.")

    fig.show()

import plotly.graph_objects as go
def plot_volume():
    feature_volume = "Volume"
    fig = go.Figure()
    # Function to create and show plots for a given feature
    fig.add_trace(go.Scatter(x=df.index, y=df[feature_volume], mode='lines', name=feature_volume, line=dict(color='blue')))
    
    fig.update_layout(
            title="Bitcoin Volume Movement Over Time",
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            autosize=True,
            height=600,
        )
        
    fig.show()
plot_volume()

# Define meaningful pairs of numerical features
feature_pairs = [
    ('Open', 'Volume'),
    ('Close', 'Volume')
]

# Create scatter plots for each pair of features
for feature_x, feature_y in feature_pairs:
    # Create scatter plot
    fig = px.scatter(
        df, x=feature_x, y=feature_y, title=f'{feature_x} vs {feature_y} (Bitcoin)',
        labels={feature_x: feature_x, feature_y: feature_y},
        template='plotly_dark', opacity=0.5
    )

    # Calculate regression statistics
    stats = calculate_regression_stats(df[feature_x], df[feature_y])

    # Conditionally add the regression line if R² is above a threshold and p-value is below a threshold
    if stats['r_squared'] > 0.5 and stats['p_value'] < 0.05:
        fig.add_trace(
            go.Scatter(
                x=df[feature_x], y=stats['slope']*df[feature_x] + stats['intercept'],
                mode='lines', name=f"y = {stats['slope']:.2f}x + {stats['intercept']:.2f}",
                line=dict(color='red')
            )
        )
    else:
        print(f"The relationship between {feature_x} and {feature_y} for Bitcoin is not significant.")

    # Show the plot
    fig.show()

# Calculate daily returns and cumulative returns
df['return'] = df['Close'].pct_change()
df['Cumulative_Return'] = (1 + df['return']).cumprod() - 1

# Create a Plotly figure
fig = go.Figure()
    
# Add trace for cumulative returns
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['Cumulative_Return'], 
    mode='lines', 
    name=f"{df['Cumulative_Return']}", 
    line=dict(color='green')
))

# Update layout
fig.update_layout(
    title='Cumulated Return for Bitcoin',
    xaxis_title='Date',
    yaxis_title='Cumulative Return',
    template='plotly_dark',  # Set the dark theme
    height=600,  # Adjust height as needed
)

# Show the plot
fig.show()
