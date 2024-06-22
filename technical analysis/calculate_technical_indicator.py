import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

"""
### Trend Following Indicators

**Moving Averages (MA)**: A simple yet effective way to smooth out price fluctuations and identify the overall trend. Common MAs include Simple Moving Average (SMA) and Exponential Moving Average (EMA), here we use SMA.

**Moving Average Convergence Divergence (MACD)**: This indicator combines two moving averages and a MACD histogram to identify potential trend reversals, trading signals, and momentum.

### Momentum Indicators

**Relative Strength Index (RSI)**: This indicator measures the magnitude of recent price movements to identify overbought or oversold conditions. A high RSI suggests the market might be overbought and due for a correction, while a low RSI indicates a potential oversold situation.
"""

# Load Bitcoin csv data
filename = 'data/BTC-USD.csv'
data = pd.read_csv(filename,parse_dates=['Date'],index_col='Date')

RSI_TIME_WINDOW = 14  # number of days
START_DATE = '2019-06-03'

# Function to compute SMA
def compute_sma(data, window):
    return data.rolling(window=window).mean()

# Function to add Golden Cross and Death Cross signals
def add_cross_signals(data, short_window=50, long_window=200):
    data[f'SMA_{short_window}'] = compute_sma(data['Close'], short_window)
    data[f'SMA_{long_window}'] = compute_sma(data['Close'], long_window)
    data['Golden_Cross'] = np.where(data[f'SMA_{short_window}'] > data[f'SMA_{long_window}'], 1, 0)
    data['Death_Cross'] = np.where(data[f'SMA_{short_window}'] < data[f'SMA_{long_window}'], 1, 0)
    return data

def data_preprocessing(start_date=START_DATE, rsi_time_window=14, overbought_level=70, oversold_level=30) -> list[pd.DataFrame]:
    """
    Preprocess multiple DataFrames into a single DataFrame.

    Parameters:
        start_date (str): Start date for filtering data.
        rsi_time_window (int): Time window for computing RSI.
        overbought_level (int): RSI level considered as overbought.
        oversold_level (int): RSI level considered as oversold.

    Returns:
        pd.DataFrame: Combined DataFrame with preprocessed data.
    """

    dataframe = data.query(f"Date > '{start_date}'")
    if dataframe.empty:
        print(f"No data available for Bitcoin after {start_date}")
        return
    dataframe = dataframe.loc[~dataframe.index.duplicated()]  # Remove duplicate indices
    dataframe["close_rsi"] = compute_rsi(dataframe['Close'], time_window=rsi_time_window)

    # Calculate SMA
    for window in [50, 100, 200]:
        dataframe[f'SMA_{window}'] = compute_sma(dataframe['Close'], window)

    # Calculate MACD
    dataframe['ema_12'] = dataframe['Close'].ewm(span=12, adjust=False).mean()
    dataframe['ema_26'] = dataframe['Close'].ewm(span=26, adjust=False).mean()
    dataframe['macd'] = dataframe['ema_12'] - dataframe['ema_26']
    dataframe['signal_line'] = dataframe['macd'].ewm(span=9, adjust=False).mean()
    dataframe['histogram'] = dataframe['macd'] - dataframe['signal_line']

    # Add Golden Cross and Death Cross signals
    dataframe = add_cross_signals(dataframe, 50, 200)

    # Add overbought and oversold signals
    dataframe['Overbought'] = np.where(dataframe['close_rsi'] >= overbought_level, 1, 0)
    dataframe['Oversold'] = np.where(dataframe['close_rsi'] <= oversold_level, 1, 0)
    # print(dataframe.info())
    return dataframe


def compute_rsi(data: pd.Series, time_window: int) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a given data series.

    Parameters:
        data (pd.Series): The data series to compute the RSI for.
        time_window (int): The window size to use for the RSI calculation.

    Returns:
        pd.Series: The computed RSI values.
    """
    # Ensure that there are enough data points for the RSI calculation.
    if len(data) <= time_window:
        return pd.Series([np.nan] * len(data), index=data.index)

    # Calculate price differences and positive/negative changes
    diff = data.diff(1).fillna(0)
    up_chg = np.where(diff > 0, diff, 0)
    down_chg = np.where(diff < 0, -diff, 0)

    # Compute exponential moving averages
    up_chg_avg = pd.Series(up_chg).ewm(span=time_window, adjust=False).mean()
    down_chg_avg = pd.Series(down_chg).ewm(span=time_window, adjust=False).mean()

    # Avoid division by zero and calculate RSI
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(down_chg_avg != 0, up_chg_avg / down_chg_avg, np.nan)

    rsi = 100 - 100 / (1 + rs)
    return rsi


def create_date_buttons():
    """
    Generates date range buttons and visibility buttons for a list of cryptocurrencies.

    Returns:
    tuple: A tuple containing two lists:
        - date_buttons (list): A list of dictionaries representing date range buttons.
    """
    date_buttons = [
        {'step': "all", 'label': "All time"},
        {'count': 1, 'step': "year", 'stepmode': "backward", 'label': "Last Year"},
        {'count': 2, 'step': "month", 'stepmode': "backward", 'label': "Last 2 Months"},
        {'count': 7, 'step': "day", 'stepmode': "backward", 'label': "Last Week"},
        {'count': 4, 'step': "day", 'stepmode': "backward", 'label': "Last 4 days"},
        {'count': 1, 'step': "day", 'stepmode': "backward", 'label': "Last day"},
    ]

    return date_buttons

# Find the minimum date among all cryptocurrencies
df = data_preprocessing(start_date=START_DATE, rsi_time_window=RSI_TIME_WINDOW)
# Plot
fig = make_subplots(
    rows=4,
    cols=2,
    shared_xaxes=True,
    specs=[
        [{"rowspan": 2}, {"rowspan": 2}],  # Subplot spanning 2 rows
        [None, None],                      # This row is spanned by the above subplots
        [{}, {}],                          # Regular subplots
        [{"colspan": 1}, {"colspan": 1}],  # Regular subplots
    ],
    subplot_titles=(
        "<b>Candlestick Chart</b>", "<b>Price Chart</b>",
        "<b>Volume</b>", "<b>Relative Strength Index (RSI)</b>",
        "<b>MACD</b>"
    )
)

# Define colors for SMA windows
sma_colors = ['blue', 'green', 'orange']
# for i, df in enumerate(btc_df):
    # Candlestick Chart
fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                showlegend=False),
                row=1, col=1)

# Trading Volume
fig.add_trace(go.Bar(x=df.index,
                        y=df["Volume"],
                        showlegend=False,
                        marker_color='aqua'),
                row=3, col=1)

# Price Chart
for trace in [
    (go.Scatter, 'Close', 'red', 'none', 4, 'solid'),
    (go.Scatter, 'Low', 'pink', 'tonexty', 2, 'dash'),
    (go.Scatter, 'High', 'pink', 'tonexty', 2, 'dash')
]:
    fig.add_trace(trace[0](x=df.index,
                            y=df[trace[1]],
                            mode='lines',
                            fill=trace[3],
                            showlegend=False,
                            line=dict(width=trace[4],
                                        color=trace[2],
                                        dash=trace[5])),
                    row=1, col=2)

# Relative Strength Index (RSI) (row 3, col 2)
fig.add_trace(go.Scatter(x=df.index,
                        y=df['close_rsi'],
                        mode='lines',
                        fill='none',
                        showlegend=False,
                        line=dict(width=4,
                                    color='aquamarine',
                                    dash='solid')),
                row=3, col=2)

# SMA for Candlestick Chart (row 1, col 1)
for idx, window in enumerate([50, 100, 200]):
    fig.add_trace(go.Scatter(x=df.index,
                                y=df[f'SMA_{window}'],
                                mode='lines',
                                name=f'SMA_{window}',
                                showlegend=False,
                                line=dict(width=1, color=sma_colors[idx])),
                    row=1, col=1)

# SMA for Price Chart (row 1, col 2)
for idx, window in enumerate([50, 100, 200]):
    fig.add_trace(go.Scatter(x=df.index,
                                y=df[f'SMA_{window}'],
                                mode='lines',
                                name=f'SMA_{window}',
                                showlegend=False,
                                line=dict(width=1, color=sma_colors[idx])),
                    row=1, col=2)

# MACD (row 4, col 1)
fig.add_trace(go.Scatter(x=df.index,
                            y=df['macd'],
                            mode='lines',
                            name='MACD',
                            showlegend=False,
                            line=dict(width=1,color='green')),
                row=4, col=1)

fig.add_trace(go.Scatter(x=df.index,
                            y=df['signal_line'],
                            mode='lines',
                            name='Signal Line',
                            showlegend=False,
                            line=dict(width=1,color='red')),
                        row=4, col=1)

fig.add_trace(go.Bar(x=df.index,
                        y=df['histogram'],
                        name='Histogram',
                        showlegend=False,
                        marker_color='orange'),
                row=4, col=1)

date_buttons = create_date_buttons()

# Update layout and add subplot titles
fig.update_layout(
    width=1300,
    height=1000,
    font_family='monospace',
    xaxis=dict(rangeselector=dict(buttons=date_buttons)),
    xaxis2=dict(rangeselector=dict(buttons=date_buttons)),
    title=dict(text='<b>Bitcoin Dashboard<b>', font=dict(color='#FFFFFF', size=22), x=0.50),
    font=dict(color="blue"),
    template="plotly_dark",
    spikedistance=100,
    xaxis_rangeslider_visible=False,
    hoverdistance=1000
)

# Update x-axis style
fig.update_xaxes(tickfont=dict(size=15, family='monospace', color='#B8B8B8'),
                 tickmode='array',
                 ticklen=6,
                 showline=False,
                 showgrid=True,
                 gridcolor='#595959',
                 ticks='outside',
                 showspikes=True,
                 spikesnap="cursor",
                 spikemode="across")

# Update y-axis settings for the volume trace
fig.update_yaxes(title_text="Volume",
                 tickfont=dict(size=15, family='monospace', color='#B8B8B8'),
                 showline=False,
                 showgrid=True,
                 gridcolor='#595959',
                 ticks='outside',
                 showspikes=True,
                 spikesnap='cursor',
                 spikemode="across",
                 row=4, col=2)

# Update the font color of the subplot titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color='#FFFFFF')  # Set the font color to white

# fig.show()

# New Created Features Correlation Analysis
df = df.dropna()

# Exclude non-numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Calculate correlation matrix
corr_matrix = numeric_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Bitcoin Indicators Correlation Heatmap')
plt.show()

# FIND THE MAXIMUM PAIR OF CORRELATION
corr_matrix_value = (corr_matrix.mask(corr_matrix == 1.0)
                   .stack()
                   .idxmax())

print(f'The most correlated pair is {corr_matrix_value} with a value of {corr_matrix.loc[corr_matrix_value]}')
