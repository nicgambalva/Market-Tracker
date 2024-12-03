# This is the main file of a project that tracks daily the performance of the tradable equities in the CAC All-Tradable index

# Importing libraries ---------------------------------------------------------

# Standard library imports
import os
import time
from datetime import datetime

# Third-party imports
import inflect
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# Local application/library-specific imports
import kaleido
import json

# Preparing environment -------------------------------------------------------

# Creating the containing folders if not already existing
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('hist'):
    os.makedirs('hist')
    os.makedirs('hist/general')
    os.makedirs('hist/returns')
    os.makedirs('hist/volume')
    os.makedirs('hist/returns/top')
    os.makedirs('hist/returns/bottom')
if not os.path.exists('images'):
    os.makedirs('images')
    
# Stopping scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Initializing the inflect engine
inflect_engine = inflect.engine()

# Getting the ticker list from CAC.xlsx
cac_all = pd.read_excel('CAC.xlsx')
all_tickers = cac_all['Yahoo Ticker'].tolist()

# Dictionary with the tickers and the corresponding names
ticker_names = dict(zip(cac_all['Yahoo Ticker'], cac_all['Entity Name']))

# Dictionary mapping the tickers to the corresponding industry (multiple tickers can be in the same industry)
sector_dict = dict(zip(cac_all['Yahoo Ticker'], cac_all['Primary Industry']))

# Dictionary with the subset of most common currencies and the symbol of the currency
currency_dict = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£',
    'JPY': '¥',
    'CHF': 'CHF',
    'CAD': 'C$',
    'AUD': 'A$',
    'CNY': '¥',
    'SEK': 'kr',
    'NZD': 'NZ$',
    'NOK': 'kr',
    'MXN': 'Mex$',
    'KRW': '₩',
    'COP': 'COL$',
    'BRL': 'R$'
}

# Dictionary of non-tradeable days (holidays)
holidays_france = {
    '01/01': 'New Year',
    '04/05': 'Easter Monday',
    '05/01': 'Labour Day',
    '05/08': 'Victory in Europe Day',
    '07/14': 'Bastille Day',
    '08/15': 'Assumption of Mary',
    '11/01': 'All Saints Day',
    '11/11': 'Armistice Day',
    '12/25': 'Christmas Day'
}

# Functions -------------------------------------------------------------------

# Function that goes through a dictionary of tickers, tries to download the data for each ticker
# Checks for errors and returns a dictionary with the tickers that have data
# And saves the tickers that caused an error in a file under data called 'delisted.csv'
def check_tickers(tickers):
    good_tickers = []
    delisted_tickers = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period='1mo')
            if data is None or data.empty:
                delisted_tickers.append(ticker)
            else:
                good_tickers.append(ticker)
        except:
            delisted_tickers.append(ticker)
    # Saving the delisted tickers to a CSV file
    delisted = pd.DataFrame(delisted_tickers, columns=['Ticker'])
    delisted.to_csv('data/delisted.csv')
    
    # Turning the good tickers into a list
    #good_tickers = list(good_tickers)
    return good_tickers

# Function that takes one date and checks if it is a holiday
def is_holiday(date, holidays_dict):
    date = date.strftime('%m/%d')
    if date in holidays_dict:
        return True
    else:
        return False
   
# Function that takes a date and returns the last trading day
def get_last_trading_day(holidays_dict):
    today = datetime.today()
    week_day = today.weekday()
    day = today.day
    month = today.month
    year = today.year
    day_ordinal = inflect_engine.ordinal(day)
    
    # If the time is before 6:00 PM, we consider the last market day to be the day before
    if today.hour < 18:
        today = today.replace(day=day-1)
        day = today.day
        month = today.month
        year = today.year
        day_ordinal = inflect_engine.ordinal(day)
    
    # If the day is a holiday, we consider the last market day to be the day before
    if is_holiday(today, holidays_dict):
        today = today.replace(day=day-1)
        day = today.day
        month = today.month
        year = today.year
        day_ordinal = inflect_engine.ordinal(day)
    
    # If the day is Saturday we consider the last market day to be the day before
    if week_day in [5]:
        today = today.replace(day=day-1)
        day = today.day
        month = today.month
        year = today.year
        day_ordinal = inflect_engine.ordinal(day)
    
    # If the day is Sunday we consider the last market day to two days before
    if week_day in [6]:
        today = today.replace(day=day-2)
        day = today.day
        month = today.month
        year = today.year
        day_ordinal = inflect_engine.ordinal(day)
        
    # Creating a date vector with
    # today, day, month, year, day_ordinal
    current_date = [today, day, month, year, day_ordinal]
    return current_date

# Function that checks if at the time the code is run, the market is open
def is_market_open():
    today = datetime.today()
    week_day = today.weekday()
    if week_day in [5, 6]:
        return False
    else:
        if today.hour < 18:
            if today.hour < 9:
                return False
            else:
                return True
        else:
            return False
    
# Function to get the 1 month of data for one ticker
def get_hist_data(ticker):
    data = yf.download(ticker, period='1mo')
    if data is None:
        print(f'No data for {ticker}')
        return None
    elif is_market_open():
        data = pd.DataFrame(data.head(len(data)-1))
        return data
    else:
        data = pd.DataFrame(data)
        return data   

# Function to calculate the average of the traded volume over a one month period
def get_average_volume(ticker, data=None):
    if data is None:
        data = yf.download(ticker, period='1mo')
        if is_market_open():
            data = pd.DataFrame(data.head(len(data)-1))
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        average_volume = data['Volume'].mean()
        return average_volume

# Function to calculate the volume index (ratio of the last trading session volume to the average volume over the last month)
def get_volume_index(ticker, data=None):
    if data is None:
        data = yf.download(ticker, period='1mo')
        if is_market_open():
            data = pd.DataFrame(data.head(len(data)-1))
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        average_volume = get_average_volume(ticker, data)
        try:
            volume_index = data['Volume'].iloc[-1] / average_volume
            return volume_index
        except:
            print(f'No historical data for {ticker}')
            return None
        
# Function that creates a dataframe with the calculated returns for each ticker and the volume index
def get_all_data(tickers):
    # Starting an empty dataframe with the following columns 'Ticker', 'Latest Return', 'Latest close', 'Previous close', 'Volume Index', 'Latest volume', 'Average volume'
    returns = pd.DataFrame(columns=['Ticker', 'Name', 'Latest Return', 'Latest Close', 'Previous Close', 'Volume Index', 'Latest Volume', 'Average Volume'])
    for ticker in tickers:
        name = ticker_names[ticker]
        data = get_hist_data(ticker)
        if data is None:
            print(f'No data for {ticker}')
            returns = pd.concat([returns, pd.DataFrame(
                {'Ticker': ticker,
                 'Name': name,
                 'Latest Return': np.nan,
                 'Latest Close': np.nan,
                 'Previous Close': np.nan,
                 'Volume Index': np.nan,
                 'Latest Volume': np.nan,
                 'Average Volume': np.nan},
                index=[0])], ignore_index=True)
            print (f'{name} has no data')
            continue
        else:
            latest_close = data['Adj Close'].iloc[-1]
            previous_close = data['Adj Close'].iloc[-2]
            latest_return = (latest_close - previous_close) / previous_close
            volume_index = get_volume_index(ticker, data)
            latest_volume = data['Volume'].iloc[-1]
            average_volume = get_average_volume(ticker, data)
            returns = pd.concat([returns, pd.DataFrame(
                {'Ticker': ticker,
                'Name': name,
                'Latest Return': latest_return,
                'Latest Close': latest_close,
                'Previous Close': previous_close,
                'Volume Index': volume_index,
                'Latest Volume': latest_volume,
                'Average Volume': average_volume},
                index=[0])], ignore_index=True)
            print(f'{name} done')
    return returns

# Function that creates 4 dataframes and saves them into 4 CSV files
# One with all the data, saved under the hist folder
# One with the tickers, latest returns, latest closes, and previous closes of the top 5 performers in returns
# One with the tickers, latest returns, latest closes, and previous closes of the bottom 5 performers in returns
# And one with the tickers, volume indexes, latest volumes, and average volumes of the top 5 performers in volume indexes

def save_all_data(tickers, holidays_dict):
    # Defining the date
    current_date = get_last_trading_day(holidays_dict)
    day = current_date[1]
    month = current_date[2]
    year = current_date[3]
    
    # Skipping all the tickers for which returns are equal to NaN or exactly 0
    
    all_data = get_all_data(tickers)
    all_data = all_data.dropna(subset=['Latest Return'])
    all_data = all_data[all_data['Latest Return'] != 0]
    
    # Formatting the numbers
    # Formatting numbers in the following format:
    # Percentages (returns): Percentage and 2 decimal points
    # Prices: 2 decimal points
    # Volumes: 0 decimal points (integers)
    # Volume ratios: 2 decimal points
    
    
    top_returns = all_data.sort_values(by='Latest Return', ascending=False).head(5)
    top_returns['Latest Return'] = top_returns['Latest Return'].apply(lambda x: "{:.2%}".format(x))
    top_returns['Latest Close'] = top_returns['Latest Close'].apply(lambda x: "{:.2f}".format(x))
    top_returns['Previous Close'] = top_returns['Previous Close'].apply(lambda x: "{:.2f}".format(x))
    top_returns['Volume Index'] = top_returns['Volume Index'].apply(lambda x: "{:.2f}".format(x))
    top_returns['Latest Volume'] = top_returns['Latest Volume'].apply(lambda x: "{:.0f}".format(x))
    top_returns['Average Volume'] = top_returns['Average Volume'].apply(lambda x: "{:.2f}".format(x))
    top_returns.to_csv(f'hist/returns/top/{day}_{month}_{year}.csv')
    top_returns.to_csv(f'data/top_returns.csv')
    
    bottom_returns = all_data.sort_values(by='Latest Return', ascending=True).head(5)
    bottom_returns['Latest Return'] = bottom_returns['Latest Return'].apply(lambda x: "{:.2%}".format(x))
    bottom_returns['Latest Close'] = bottom_returns['Latest Close'].apply(lambda x: "{:.2f}".format(x))
    bottom_returns['Previous Close'] = bottom_returns['Previous Close'].apply(lambda x: "{:.2f}".format(x))
    bottom_returns['Volume Index'] = bottom_returns['Volume Index'].apply(lambda x: "{:.2f}".format(x))
    bottom_returns['Latest Volume'] = bottom_returns['Latest Volume'].apply(lambda x: "{:.0f}".format(x))
    bottom_returns['Average Volume'] = bottom_returns['Average Volume'].apply(lambda x: "{:.2f}".format(x))
    bottom_returns.to_csv(f'hist/returns/bottom/{day}_{month}_{year}.csv')
    bottom_returns.to_csv(f'data/bottom_returns.csv')
    
    top_volume = all_data.sort_values(by='Volume Index', ascending=False).head(5)
    top_volume['Latest Return'] = top_volume['Latest Return'].apply(lambda x: "{:.2%}".format(x))
    top_volume['Latest Close'] = top_volume['Latest Close'].apply(lambda x: "{:.2f}".format(x))
    top_volume['Previous Close'] = top_volume['Previous Close'].apply(lambda x: "{:.2f}".format(x))
    top_volume['Volume Index'] = top_volume['Volume Index'].apply(lambda x: "{:.2f}".format(x))
    top_volume['Latest Volume'] = top_volume['Latest Volume'].apply(lambda x: "{:.0f}".format(x))
    top_volume['Average Volume'] = top_volume['Average Volume'].apply(lambda x: "{:.2f}".format(x))
    top_volume.to_csv(f'hist/volume/{day}_{month}_{year}.csv')
    top_volume.to_csv(f'data/top_volume.csv')
    
    all_data['Latest Return'] = all_data['Latest Return'].apply(lambda x: "{:.2%}".format(x))
    all_data['Latest Close'] = all_data['Latest Close'].apply(lambda x: "{:.2f}".format(x))
    all_data['Previous Close'] = all_data['Previous Close'].apply(lambda x: "{:.2f}".format(x))
    all_data['Volume Index'] = all_data['Volume Index'].apply(lambda x: "{:.2f}".format(x))
    all_data['Latest Volume'] = all_data['Latest Volume'].apply(lambda x: "{:.0f}".format(x))
    all_data['Average Volume'] = all_data['Average Volume'].apply(lambda x: "{:.2f}".format(x))
    all_data.to_csv(f'hist/general/{day}_{month}_{year}.csv')
    
# Function that takes a ticker, it downloads from Yahoo Fiannce the following data for valuation purposes:
# Company sector, Industry, Country, Price, EV, EBITDA, EBIT, Net Income, Revenue, Market Cap, Dividend Yield
# It then creates a dictionary with the data and returns it
def get_valuation_data(ticker):
    data = yf.Ticker(ticker)
    
    sector = sector_dict.get(ticker, 'N/A')
    industry = data.info.get('industry', 'N/A')
    country = data.info.get('country', 'N/A')
    price = data.info.get('regularMarketPrice', 'N/A')
    ev = data.info.get('enterpriseValue', 'N/A')
    ebitda = data.info.get('EBITDA', 'N/A')
    ebit = data.info.get('EBIT', 'N/A')
    net_income = data.info.get('netIncome', 'N/A')
    revenue = data.info.get('totalRevenue', 'N/A')
    market_cap = data.info.get('marketCap', 'N/A')
    dividend_yield = data.info.get('dividendYield', 'N/A')
        
    valuation_data = {
        'Sector': sector,
        'Industry': industry,
        'Country': country,
        'Price': price,
        'EV': ev,
        'EBITDA': ebitda,
        'EBIT': ebit,
        'Net Income': net_income,
        'Revenue': revenue,
        'Market Cap': market_cap,
        'Dividend Yield': dividend_yield
    }
    
    return valuation_data

# Function that reads the latest data from the CSV files and returns the list of the top 5 performers in returns (tickers)
def get_top_returns():
    top_returns = pd.read_csv('data/top_returns.csv')
    top_returns = top_returns['Ticker'].tolist()
    return top_returns

# Function that reads the latest data from the CSV files and returns the list of the bottom 5 performers in returns (tickers)
def get_bottom_returns():
    bottom_returns = pd.read_csv('data/bottom_returns.csv')
    bottom_returns = bottom_returns['Ticker'].tolist()
    return bottom_returns

# Function that reads the latest data from the CSV files and returns the list of the top 5 performers in volume indexes (tickers)
def get_top_volume():
    top_volume = pd.read_csv('data/top_volume.csv')
    top_volume = top_volume['Ticker'].tolist()
    return top_volume

# Function that takes a ticker and creates a candlestick chart with one year of data
def plot_candlestick(ticker, currency, data=None):
    # Defining the date
    current_date = get_last_trading_day(holidays_france)
    day = current_date[1]
    month = current_date[2]
    year = current_date[3]
    
    try:
        currency_symbol = currency_dict[currency]
    except:
        currency_symbol = ''
    
    if data is None:
        data = yf.download(ticker, period='1y')
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        # Setting the theme to plotly-white
        fig.update_layout(title=f'{ticker_names[ticker]} - {day}/{month}/{year}',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          font=dict(family='Geneva', size=18, color='black'),
                          xaxis_rangeslider_visible=False,
                          plot_bgcolor='whitesmoke',
                          paper_bgcolor='white')
        
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
                ]
            
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            tickprefix=currency_symbol
        )
        
        # Not graphing days with no data
        print(f'{ticker} done')
        return fig

# Function that takes a ticker and creates a bar chart with the volume data for one year
def plot_volume(ticker, holidays_dict, data=None):
    # Getting the date
    current_date = get_last_trading_day(holidays_dict)
    day = current_date[1]
    month = current_date[2]
    year = current_date[3]
    
    if data is None:
        data = yf.download(ticker, period='1y')
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        
        # Calculating and adding a 5 day moving average for the volume
        data['Volume MA'] = data['Volume'].rolling(window=5).mean()
        
        fig = go.Figure(data=[go.Bar(x=data.index, y=data['Volume'])])
        # Setting the theme to plotly-white
        fig.update_layout(title=f'{ticker_names[ticker]} - {day}/{month}/{year}',
                          xaxis_title='Date',
                          yaxis_title='Volume',
                          font=dict(family='Geneva', size=18, color='black'),
                          xaxis_rangeslider_visible=False,
                          plot_bgcolor='whitesmoke',
                          paper_bgcolor='white')
        
        fig.data[0].name = 'Volume'
        
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
                ]
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black'
        )
        
        # No border for the bars
        fig.update_traces(
            marker=dict(color='darkgray', line=dict(color='black', width=0))
        )
        
        # Adding the 5 day moving average to the chart as a line
        fig.add_trace(go.Scatter(x=data.index, y=data['Volume MA'], mode='lines', name='5 Day Moving Average', line=dict(color='darkorange', width=2)))
        
        # Updating legend
        fig.update_layout(legend=dict( orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        
        print(f'{ticker} done')
        return fig
    
# Function that takes a ticker and creates a stylized line graph of the adjusted closing price with the data from one year
def plot_adjusted_close(ticker, holidays_dict, data=None):
    # Getting the date
    current_date = get_last_trading_day(holidays_dict)
    day = current_date[1]
    month = current_date[2]
    year = current_date[3]
    
    if data is None:
        data = yf.download(ticker, period='1y')
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Adj Close'])])
        # Setting the theme to plotly-white
        fig.update_layout(title=f'{ticker_names[ticker]} - {day}/{month}/{year}',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          font=dict(family='Geneva', size=18, color='black'),
                          xaxis_rangeslider_visible=False,
                          plot_bgcolor='whitesmoke',
                          paper_bgcolor='white')
        
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            rangebreaks=[
                dict(bounds=["sat", "mon"])
                ]
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black'
        )
        
        # Adding the 5 day moving average to the chart as a line
        fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'].rolling(window=5).mean(), mode='lines', name='5 Day Moving Average', line=dict(color='darkorange', width=2)))
        
        # Updating legend
        fig.update_layout(legend=dict( orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        
        print(f'{ticker} done')
        return fig

# Function that creates the candlestick charts for the top 5 performers in returns
def plot_candlestick_top_returns(currency, width=800, height=500):
    top_returns = get_top_returns()
    path = 'images/Top_Returns'
    count = 0
    if not os.path.exists(path):
        os.makedirs(path)
    for ticker in top_returns:
        count += 1
        fig = plot_candlestick(ticker, currency=currency)
        fig.write_image(f'{path}/Top_{count}_candlestick.png', width=width, height=height, engine='kaleido')

# Function that creates the volume charts for the top 5 performers in returns
def plot_volume_top_returns(width=800, height=250):
    top_returns = get_top_returns()
    path = 'images/Top_Returns'
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for ticker in top_returns:
        count += 1
        fig = plot_volume(ticker, holidays_france)
        fig.write_image(f'{path}/Top_{count}_volume.png', width=width, height=height, engine='kaleido')

# Function that creates the candlestick charts for the bottom 5 performers in returns
def plot_candlestick_bottom_returns(currency, width=800, height=500):
    bottom_returns = get_bottom_returns()
    path = 'images/Bottom_Returns'
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for ticker in bottom_returns:
        count += 1
        fig = plot_candlestick(ticker, currency=currency)
        fig.write_image(f'{path}/Bottom_{count}_candlestick.png', width=width, height=height, engine='kaleido')
        
# Function that creates the volume charts for the bottom 5 performers in returns
def plot_volume_bottom_returns(width=800, height=250):
    bottom_returns = get_bottom_returns()
    path = 'images/Bottom_Returns'
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for ticker in bottom_returns:
        count += 1
        fig = plot_volume(ticker, holidays_france)
        fig.write_image(f'{path}/Bottom_{count}_volume.png', width=width, height=height, engine='kaleido')

# Function that creates the candlestick charts for the top 5 performers in volume indexes
def plot_candlestick_top_volume(currency, width=800, height=500):
    top_volume = get_top_volume()
    path = 'images/Top_Volume'
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for ticker in top_volume:
        count += 1
        fig = plot_candlestick(ticker, currency=currency)
        fig.write_image(f'{path}/Top_{count}_candlestick.png', width=width, height=height, engine='kaleido')

# Function that creates the volume charts for the top 5 performers in volume indexes
def plot_volume_top_volume(width=800, height=250):
    top_volume = get_top_volume()
    path = 'images/Top_Volume'
    if not os.path.exists(path):
        os.makedirs(path)
    count = 0
    for ticker in top_volume:
        count += 1
        fig = plot_volume(ticker, holidays_france)
        fig.write_image(f'{path}/Top_{count}_volume.png', width=width, height=height, engine='kaleido')

# Function that runs all the functions to create the candlestick and volume charts
def create_images(currency, width_candlestick=800, height_candlestick=500, width_volume=800, height_volume=250):
    # Candlestick charts 
    plot_candlestick_top_returns(currency, width=width_candlestick, height=height_candlestick)
    plot_candlestick_bottom_returns(currency, width=width_candlestick, height=height_candlestick)
    plot_candlestick_top_volume(currency, width=width_candlestick, height=height_candlestick)
    
    # Volume charts
    plot_volume_top_returns(width=width_volume, height=height_volume)
    plot_volume_bottom_returns(width=width_volume, height=height_volume)
    plot_volume_top_volume(width=width_volume, height=height_volume)

# Functions for ceating tables ------------------------------------------------

# Function that takes a ticker and creates a sumary table for the returns (adjusted close)
# For 1D, 1M, YTD, and 1Y

def create_summary_table_returns(ticker, currency, data=None):
    if data is None:
        data = yf.download(ticker, period='1y')
    if data is None:
        print(f'No data for {ticker}')
        return None
    else:
        # Getting the date
        current_date = get_last_trading_day(holidays_france)
        day = current_date[1]
        month = current_date[2]
        year = current_date[3]
        
        # Getting the currency symbol
        try:
            currency_symbol = currency_dict[currency]
        except:
            currency_symbol = ''
        
        # Creating the table
        summary_table = pd.DataFrame(columns=['Timeframe', 'Start Date', 'End Date', 'Start Price', 'End Price', 'Return'])
        
        # 1D
        one_day = data.tail(2)
        one_day_return = (one_day['Adj Close'].iloc[-1] - one_day['Adj Close'].iloc[0]) / one_day['Adj Close'].iloc[0]
        summary_table = pd.concat([summary_table, pd.DataFrame(
            {'Timeframe': '1D',
             'Start Date': one_day.index[0].strftime('%d/%m/%Y'),
             'End Date': one_day.index[-1].strftime('%d/%m/%Y'),
             'Start Price': one_day['Adj Close'].iloc[0],
             'End Price': one_day['Adj Close'].iloc[-1],
             'Return': one_day_return},
            index=[0])], ignore_index=True)
        
        # 1M
        one_month = data.tail(21)
        one_month_return = (one_month['Adj Close'].iloc[-1] - one_month['Adj Close'].iloc[0]) / one_month['Adj Close'].iloc[0]
        summary_table = pd.concat([summary_table, pd.DataFrame(
            {'Timeframe': '1M',
             'Start Date': one_month.index[0].strftime('%d/%m/%Y'),
             'End Date': one_month.index[-1].strftime('%d/%m/%Y'),
             'Start Price': one_month['Adj Close'].iloc[0],
             'End Price': one_month['Adj Close'].iloc[-1],
             'Return': one_month_return},
            index=[0])], ignore_index=True)
        
        # 1Y
        one_year = data.tail(252)
        one_year_return = (one_year['Adj Close'].iloc[-1] - one_year['Adj Close'].iloc[0]) / one_year['Adj Close'].iloc[0]
        summary_table = pd.concat([summary_table, pd.DataFrame(
            {'Timeframe': '1Y',
             'Start Date': one_year.index[0].strftime('%d/%m/%Y'),
             'End Date': one_year.index[-1].strftime('%d/%m/%Y'),
             'Start Price': one_year['Adj Close'].iloc[0],
             'End Price': one_year['Adj Close'].iloc[-1],
             'Return': one_year_return},
            index=[0])], ignore_index=True)
        
        # YTD
        # Calculating the number of trading days since the beginning of the year
        ytd = data[data.index.year == year]
        ytd_return = (ytd['Adj Close'].iloc[-1] - ytd['Adj Close'].iloc[0]) / ytd['Adj Close'].iloc[0]
        summary_table = pd.concat([summary_table, pd.DataFrame(
            {'Timeframe': 'YTD',
             'Start Date': ytd.index[0].strftime('%d/%m/%Y'),
             'End Date': ytd.index[-1].strftime('%d/%m/%Y'),
             'Start Price': ytd['Adj Close'].iloc[0],
             'End Price': ytd['Adj Close'].iloc[-1],
             'Return': ytd_return},
            index=[0])], ignore_index=True)
        
        # Formatting the numbers
        summary_table['Start Price'] = summary_table['Start Price'].apply(lambda x: "{:.2f}".format(x))
        summary_table['End Price'] = summary_table['End Price'].apply(lambda x: "{:.2f}".format(x))
        summary_table['Return'] = summary_table['Return'].apply(lambda x: "{:.2%}".format(x))
        
        return summary_table


# Testing ---------------------------------------------------------------------

#save_all_data(tickers, holidays_france)
#plot_candlestick_top_returns('EUR')
#plot_candlestick_bottom_returns('EUR')
#plot_candlestick_top_volume('EUR')
#print('Done')

tickers = check_tickers(all_tickers)
save_all_data(tickers, holidays_france)
create_images('EUR', width_candlestick=800, height_candlestick=500, width_volume=800, height_volume=400)

# Testing the valuation data function on Hermès
#hermes_data = get_valuation_data('RMS.PA')
#print(hermes_data)

# raw_hermes = yf.Ticker('RMS.PA')
# print(raw_hermes.balance_sheet)
# print(raw_hermes.balancesheet)
# print(raw_hermes.actions)
# print(raw_hermes.basic_info)
# print(raw_hermes.calendar)
# print(raw_hermes.dividends)
# print(raw_hermes.analyst_price_target)
# print(raw_hermes.earnings)
# print(raw_hermes.earnings_dates)
# print(raw_hermes.earnings_forecasts)
# print(raw_hermes.earnings_trend)
# print(raw_hermes.major_holders)
