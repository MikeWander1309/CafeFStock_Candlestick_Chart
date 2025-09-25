import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np
from datetime import datetime, time
import pytz

st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")


class StockDashboard:
    def __init__(self):
        self.url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        self.vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')

    def is_market_open(self):
        """Check if Vietnam stock market is currently open"""
        now = datetime.now(self.vn_tz)
        current_time = now.time()
        current_weekday = now.weekday()

        # Market closed on weekends (Saturday=5, Sunday=6)
        if current_weekday >= 5:
            return False

        # Trading hours: 9:00-11:30 and 13:00-15:00 Vietnam time
        morning_start = time(9, 0)
        morning_end = time(11, 30)
        afternoon_start = time(13, 0)
        afternoon_end = time(15, 0)

        is_morning_session = morning_start <= current_time <= morning_end
        is_afternoon_session = afternoon_start <= current_time <= afternoon_end

        return is_morning_session or is_afternoon_session

    def filter_trading_days(self, df):
        """Remove weekends, invalid zero rows, and incomplete latest bar"""
        if df.empty:
            return df

        # Remove weekends
        df = df[df.index.weekday < 5].copy()

        # Remove any rows with zero/invalid prices (catches API quirks)
        df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]

        # Handle latest bar
        now = datetime.now(self.vn_tz)
        today = now.date()
        market_end = time(15, 0)

        if not df.empty:
            latest_date = df.index[-1].date()
            if latest_date == today:
                # If before end of trading day or market is open, drop latest if incomplete
                if now.time() < market_end or self.is_market_open():
                    df = df.iloc[:-1]
                # Extra check: if latest is still zero after above, drop it
                if not df.empty and df.iloc[-1]['Close'] == 0:
                    df = df.iloc[:-1]

        return df

    def get_data(self, symbol):
        params = {
            "symbol": symbol,
            "StartDate": "",
            "EndDate": "",
            "PageIndex": 1,
            "PageSize": 1000
        }

        try:
            response = requests.get(self.url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                records = data['Data']['Data']
                df = pd.DataFrame(records)

                df = df.rename(columns={
                    'Ngay': 'Date', 'GiaDongCua': 'Close', 'GiaMoCua': 'Open',
                    'GiaCaoNhat': 'High', 'GiaThapNhat': 'Low', 'KhoiLuongKhopLenh': 'Volume'
                })

                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna().sort_values('Date').set_index('Date')

                # Apply filters
                df = self.filter_trading_days(df)

                return df
        except Exception as e:
            st.warning(f"API error: {str(e)}. Using sample data.")
            pass

        # Fallback sample data (business days only, positive prices)
        dates = pd.date_range(end=datetime.now(), periods=200, freq='B')  # Business days only
        base = 50000
        data = []
        for date in dates:
            change = np.random.uniform(-0.03, 0.03)
            open_p = base * (1 + np.random.uniform(-0.01, 0.01))
            close_p = base * (1 + change)
            high_p = max(open_p, close_p) * (1 + abs(np.random.uniform(0, 0.015)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.uniform(0, 0.015)))
            vol = int(np.random.uniform(1000000, 5000000))

            data.append({
                'Date': date, 'Open': open_p, 'High': high_p,
                'Low': low_p, 'Close': close_p, 'Volume': vol
            })
            base = close_p

        df = pd.DataFrame(data).set_index('Date')
        return self.filter_trading_days(df)

    def calculate_indicators(self, df):
        # EMA
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()

        # SMA
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        return df

    def resample_data(self, df, timeframe):
        if timeframe == '1W':
            # Resample OHLCV data
            resampled = df.resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()

            # Recalculate indicators on resampled data
            resampled = self.calculate_indicators(resampled)
            return resampled

        elif timeframe == '1M':
            # Resample OHLCV data
            resampled = df.resample('M').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()

            # Recalculate indicators on resampled data
            resampled = self.calculate_indicators(resampled)
            return resampled

        # For daily timeframe, return as is
        return df

    def create_chart(self, df, symbol, toggles, timeframe):
        # Determine number of subplots
        subplot_count = 1
        subplot_titles = [f'{symbol} - {timeframe}']

        if toggles['volume']:
            subplot_count += 1
            subplot_titles.append('Volume')

        if toggles['rsi']:
            subplot_count += 1
            subplot_titles.append('RSI')

        if toggles['macd']:
            subplot_count += 1
            subplot_titles.append('MACD')

        # Create subplot heights
        if subplot_count == 1:
            heights = [1]
        elif subplot_count == 2:
            heights = [0.7, 0.3]
        elif subplot_count == 3:
            heights = [0.6, 0.2, 0.2]
        else:
            heights = [0.5, 0.2, 0.15, 0.15]

        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=heights
        )

        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price',
                increasing_line_color='#26C281', decreasing_line_color='#E74C3C'
            ), row=1, col=1
        )

        # Moving averages (only add if they exist in the dataframe)
        if toggles['sma20'] and 'SMA20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA20'], name='SMA20',
                           line=dict(color='#3498DB', width=2)), row=1, col=1
            )

        if toggles['sma50'] and 'SMA50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                           line=dict(color='#9B59B6', width=2)), row=1, col=1
            )

        if toggles['ema12'] and 'EMA12' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['EMA12'], name='EMA12',
                           line=dict(color='#F39C12', width=1)), row=1, col=1
            )

        if toggles['ema26'] and 'EMA26' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['EMA26'], name='EMA26',
                           line=dict(color='#E67E22', width=1)), row=1, col=1
            )

        # Bollinger Bands
        if toggles['bollinger'] and 'BB_Upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                           line=dict(color='gray', width=1, dash='dot'),
                           showlegend=False), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                           line=dict(color='gray', width=1, dash='dot'),
                           fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                           showlegend=False), row=1, col=1
            )

        current_row = 2

        # Volume
        if toggles['volume']:
            colors = ['#E74C3C' if close < open else '#26C281'
                      for open, close in zip(df['Open'], df['Close'])]
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                       marker_color=colors, showlegend=False),
                row=current_row, col=1
            )
            current_row += 1

        # RSI
        if toggles['rsi'] and 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                           line=dict(color='#F1C40F', width=2)),
                row=current_row, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.update_yaxes(range=[0, 100], row=current_row, col=1)
            current_row += 1

        # MACD
        if toggles['macd'] and 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                           line=dict(color='#3498DB', width=2)),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                           line=dict(color='#E74C3C', width=2)),
                row=current_row, col=1
            )

            colors = ['#26C281' if x > 0 else '#E74C3C' for x in df['MACD_Hist']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                       marker_color=colors, showlegend=False),
                row=current_row, col=1
            )

        fig.update_layout(
            template='plotly_dark',
            height=200 + (subplot_count * 150),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def get_market_status(self):
        """Get current market status for display"""
        now = datetime.now(self.vn_tz)
        is_open = self.is_market_open()

        if is_open:
            return "🟢 Market Open", "success"
        else:
            current_time = now.time()
            if current_time < time(9, 0):
                return "🔴 Pre-Market", "warning"
            elif time(11, 30) < current_time < time(13, 0):
                return "🟡 Lunch Break", "warning"
            elif current_time > time(15, 0):
                return "🔴 Market Closed", "error"
            else:
                return "🔴 Market Closed", "error"


def main():
    st.title("📊 Interactive Stock Dashboard")
    dashboard = StockDashboard()

    # New UX: Use tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Chart", "📊 Metrics", "⚙️ Settings"])

    with tab3:  # Settings tab
        st.header("Settings")

        # Stock input and timeframe in a row
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Stock Symbol", value="VCB", placeholder="Enter stock code")
        with col2:
            timeframe = st.selectbox(
                "Time Frame",
                options=['1D', '1W', '1M'],
                index=0
            )

        st.divider()

        # Indicators in expander
        with st.expander("📈 Indicators", expanded=True):
            col_ind1, col_ind2 = st.columns(2)
            with col_ind1:
                sma20 = st.checkbox("SMA20", value=True)
                sma50 = st.checkbox("SMA50", value=False)
                ema12 = st.checkbox("EMA12", value=False)
                ema26 = st.checkbox("EMA26", value=False)
            with col_ind2:
                bollinger = st.checkbox("Bollinger", value=False)
                volume = st.checkbox("Volume", value=True)
                rsi = st.checkbox("RSI", value=True)
                macd = st.checkbox("MACD", value=False)

        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Load data (done outside tabs to avoid reloading)
    if symbol:
        with st.spinner(f"Loading {symbol}..."):
            df = dashboard.get_data(symbol)
            df = dashboard.calculate_indicators(df)
            df_resampled = dashboard.resample_data(df, timeframe)

        # Toggles dictionary
        toggles = {
            'sma20': sma20, 'sma50': sma50, 'ema12': ema12, 'ema26': ema26,
            'bollinger': bollinger, 'volume': volume, 'rsi': rsi, 'macd': macd
        }

        with tab1:  # Chart tab
            # Market status at top
            market_status, status_type = dashboard.get_market_status()
            st.markdown(f"**{market_status}** - Vietnam Time: {datetime.now(dashboard.vn_tz).strftime('%H:%M:%S')}")

            # Create and display chart
            if not df_resampled.empty:
                fig = dashboard.create_chart(df_resampled, symbol, toggles, timeframe)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No data available for chart.")

        with tab2:  # Metrics tab
            # Market status again for context
            st.markdown(f"**{market_status}** - Vietnam Time: {datetime.now(dashboard.vn_tz).strftime('%H:%M:%S')}")

            if not df.empty:
                # Ensure latest is valid
                if df.iloc[-1]['Close'] == 0:
                    st.warning("Latest data invalid (zero price). Using previous day.")
                    df = df.iloc[:-1]

                if len(df) >= 1:
                    current = df.iloc[-1]
                    previous = df.iloc[-2] if len(df) > 1 else current
                    change = current['Close'] - previous['Close']
                    change_pct = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0

                    # Display metrics in cards
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Price", f"{current['Close']:,.0f}", f"{change:+,.0f} ({change_pct:+.2f}%)")
                        st.metric("High", f"{current['High']:,.0f}")
                    with col_m2:
                        st.metric("Volume", f"{current['Volume']:,.0f}")
                        st.metric("Low", f"{current['Low']:,.0f}")

                    # Show data info
                    st.caption(f"Latest data: {df.index[-1].strftime('%Y-%m-%d')} | Total bars: {len(df)}")
                else:
                    st.error("No valid data available after filtering.")
            else:
                st.error("No data available.")

            # Optional: Add a small table of recent data
            st.subheader("Recent Data")
            if not df.empty:
                st.dataframe(df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']])
    else:
        st.info("Enter a stock symbol in the Settings tab to load data.")

if __name__ == "__main__":
    main()
