import requests
import pandas as pd
import datetime as dt
import mplfinance as mpf



URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def plot_stock(symbol, interval="D"):
    today = dt.datetime.now()

    # --- Gọi API ---
    params = {
        "symbol": symbol,
        "StartDate": "",
        "EndDate": "",
        "PageIndex": 1,
        "PageSize": 1000
    }
    resp = requests.get(URL, params=params, headers=HEADERS)
    res = resp.json()

    if not res.get("Data") or not res["Data"].get("Data"):
        print(f"⚠️ Không có dữ liệu cho {symbol}")
        return

    # --- DataFrame ---
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

    # --- Resample theo interval ---
    if interval != "D":  # chỉ cần gộp khi không phải daily
        df = df.resample(interval).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

    # --- Bỏ nến hôm nay nếu chưa đóng cửa ---
    market_close = today.replace(hour=15, minute=0, second=0, microsecond=0)
    if today < market_close and df.index[-1].date() == today.date():
        df = df.iloc[:-1]

    # --- Vẽ chart ---

    custom_style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        rc={'font.size': 10}
    )

    mpf.plot(
        df,
        type='candle',  # candlestick
        volume=True,  # show volume
        style=custom_style,
        title=f"{symbol} Candlestick Chart",
        ylabel="Price",
        ylabel_lower="Volume",
        figsize=(12, 8)
    )


def ask_and_loop():
    while True:
        symbol = input("Nhập mã cổ phiếu (hoặc 'q' để thoát): ").upper()
        if symbol == "Q":
            break

        print("Chọn khung thời gian: ")
        print("1 = Daily, 2 = Weekly, 3 = Monthly")
        choice = input("👉 Nhập lựa chọn: ")

        if choice == "1":
            interval = "D"
        elif choice == "2":
            interval = "W"
        elif choice == "3":
            interval = "ME"
        else:
            interval = "D"

        plot_stock(symbol, interval)


if __name__ == "__main__":
    ask_and_loop()
