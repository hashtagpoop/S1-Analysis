import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame

# We are currently using IPO scoop since it already categorizes,
# puts in the high level information of an IPO and gives us
# useful information like dates and initial share prices + volume.
# Later on, we may want to consider pulling data from Yahoo or Nasdaq,
# since those are more trustworthy sources.
# However, this is still a solid start. Just make sure data later on
# ends up having the schema that is mentioned in transform_ipo_data.
IPO_SCOOP_URL = "https://www.iposcoop.com/last-100-ipos/"
YAHOO_FINANCE_URL = "https://finance.yahoo.com/q?s="

def add_yahoo_finance_links(df):
    if df is not None and 'Symbol' in df.columns:
        df['yahoo_finance_link'] = df['Symbol'].apply(lambda x: f'{YAHOO_FINANCE_URL}{x}')
    return df

def get_latest_ipos():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(IPO_SCOOP_URL, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the specific table with the given class
        table = soup.find('table', class_='ipolist')
        
        if table:
            # Use pandas to read the HTML table
            df = pd.read_html(str(table))[0]
            
            # Clean column names (remove whitespace and special characters)
            df.columns = df.columns.str.strip().str.replace('\n', ' ')
            
            # Add Yahoo Finance links
            df = add_yahoo_finance_links(df)
            return transform_ipo_data(df)
        else:
            print("Table not found")
            return None
            
    except requests.RequestException as e:
        print(f"Error fetching IPO data: {e}")
        return None
    except Exception as e:
        print(f"Error processing IPO data: {e}")
        return None

def transform_ipo_data(df: DataFrame) -> DataFrame:
    # This is the schema definition for the IPO DataFrame.
    # This is what it gets marshalled into for CSV.
    # This renames all columns so they are compatible with
    # our csv format and later DB tables.
    df = df.rename(columns={
        'Symbol': 'ticker',
        'Company': 'company',
        'Industry': 'category',
        'Offer Date': 'ipo_date',
        'Shares (millions)': 'shares_in_millions',
        'Offer Price': 'offer_price',
    })
    # Drop unnecessary columns from the website table.
    df = df.drop(columns=['1st Day Close', 'Current Price', 'Return', 'SCOOP Rating'])
    
    # Perform conversions of data.
    df['ipo_date'] = pd.to_datetime(df['ipo_date'], format='%m/%d/%Y').apply(lambda x: int(x.timestamp()))
    return df

def save_to_csv(df: DataFrame, filename: str):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # This will get the latest IPOs from the webpage tables.
    # This will then transform the data to the proper format
    # where it can be used in downstream CSV files and databases.
    # The data is cleaned for the column names, data types and formats, etc.
    ipo_data = get_latest_ipos()
    if ipo_data is not None:
        print(ipo_data.head(50))
        save_to_csv(ipo_data, 'latest_ipos.csv')