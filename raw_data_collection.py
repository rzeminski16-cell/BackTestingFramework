import requests
import pandas as pd
import os
import time
from typing import List

# Alpha Vantage API configuration
API_KEY = "8N7K6A9VC2TIVNB6"  # Replace with your Alpha Vantage API key
BASE_URL = "https://www.alphavantage.co/query"

def fetch_stock_data(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Fetch historical daily stock data from Alpha Vantage API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
        
    Returns:
        DataFrame with date, OHLC, and volume data
    """
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",  # Get full historical data
        "apikey": api_key
    }
    
    try:
        print(f"Fetching data for {ticker}...")
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(data)
        
        # Check for error messages
        if "Error Message" in data:
            print(f"Error fetching {ticker}: {data['Error Message']}")
            return None
        
        if "Note" in data:
            print(f"API limit reached: {data['Note']}")
            return None
            
        # Extract time series data
        time_series = data.get("Time Series (Daily)", {})
        
        if not time_series:
            print(f"No data found for {ticker}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index.name = 'date'
        df.reset_index(inplace=True)
        
        # Rename columns to remove the number prefix
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)
        
        # Sort by date (oldest to newest)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"Successfully fetched {len(df)} records for {ticker}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Request error for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {ticker}: {e}")
        return None

def save_to_csv(df: pd.DataFrame, ticker: str, output_dir: str = "raw_data"):
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        ticker: Stock ticker symbol
        output_dir: Directory to save CSV files
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {ticker} data to {filepath}\n")

def collect_data(tickers: List[str], api_key: str, output_dir: str = "raw_data"):
    """
    Collect historical data for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        api_key: Alpha Vantage API key
        output_dir: Directory to save CSV files
    """
    print(f"Starting data collection for {len(tickers)} securities...\n")
    
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}")
        
        df = fetch_stock_data(ticker, api_key)
        
        if df is not None:
            save_to_csv(df, ticker, output_dir)
            successful += 1
        else:
            failed += 1
        
        # Alpha Vantage free tier: 5 API calls per minute, 500 per day
        # Add delay to avoid rate limiting
        if i < len(tickers):
            print("Waiting 12 seconds to avoid rate limiting...")
            time.sleep(12)
    
    print(f"\nData collection complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    # Define your list of tickers here
    tickers = [
        "AAPL"
        #"MSFT",
        #"GC=F",
        #"BTC-USD"
    ]
    
    # Make sure to set your API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your Alpha Vantage API key!")
        print("Get your free API key at: https://www.alphavantage.co/support/#api-key")
    else:
        collect_data(tickers, API_KEY)