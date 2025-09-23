import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import streamlit as st

# API
URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Function to fetch stock data
def get_stock_data(symbol):
    today = dt.datetime.now()

    params = {
        "symbol": symbol,
        "fromdate": "",  # no restriction
        "todate": "",    # always up to date
        "page": 1,
        "pagesize": 1000
    }
    resp = requests.get(URL, params=params, headers=HEADERS)
    res = resp.json()

    if not res.get("Data") or not res["Data"].get("Data"):
        return None

    df = pd.DataFrame(res["Data"]["Data"])
    df['Ngay'] = pd.to_datetime(df['Ngay'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Ngay'])

    df = df.rename(columns={
        'Ngay': 'Date',
        'GiaMoCua': 'Open',
        'GiaCaoNhat': 'High',
        'GiaThapNhat': 'Low',
        'GiaDongCua': 'Close',
        'KhoiLuongKhopLenh': 'Volume'
    })
    df = df[(df['Close'] > 0) & (df['Volume'] > 0)]
    df = df.set_index('Date').sort_index()
    return df


# Candlestick with matplotlib
def plot_candlestick(df, symbol):
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot candles manually
    for i, (date, row) in enumerate(df.iterrows()):
        color = "green" if row["Close"] >= row["Open"] else "red"
        ax.plot([i, i], [row["Low"], row["High"]], color=color)  # wick
        ax.add_patch(plt.Rectangle((i - 0.3, min(row["Open"], row["Close"])),
                                   0.6,
                                   abs(row["Close"] - row["Open"]),
                                   color=color))

    ax.set_xticks(range(0, len(df), max(len(df)//10, 1)))
    ax.set_xticklabels(df.index.strftime("%Y-%m-%d")[::max(len(df)//10, 1)], rotation=45)

    ax.set_title(f"{symbol} Candlestick Chart")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# Streamlit UI
st.title("📈 Vietnam Stock Candlestick Viewer")

symbol = st.text_input("Enter stock symbol (e.g., VNM, VIC, HPG):", "VIC")

if st.button("Show Chart"):
    df = get_stock_data(symbol)
    if df is None:
        st.warning(f"No data found for {symbol}")
    else:
        fig = plot_candlestick(df, symbol)
        st.pyplot(fig)
