# app.py (v3.0 - Professional Dashboard)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from fear_greed_index.main import get_fear_greed_index
import matplotlib.pyplot as plt

# --- 1. 全局設定與數據快取 ---

# 預設市場組合
# 每個組合都包含：主要ETF、等權重ETF、波動率指數、以及用來判斷市場政權嘅大盤指數
MARKET_PRESETS = {
    "美國科技股 (QQQ)": {
        "etf": "QQQ", "equal_weight": "QQQE", "volatility": "^VXN", "market_index": "^GSPC"
    },
    "美國大盤 (SPY)": {
        "etf": "SPY", "equal_weight": "RSP", "volatility": "^VIX", "market_index": "^GSPC"
    },
    "黃金 (GLD)": {
        "etf": "GLD", "equal_weight": "GDX", "volatility": "^GVZ", "market_index": "^GSPC"
    },
    "新興市場 (EEM)": {
        "etf": "EEM", "equal_weight": "EWS", "volatility": "^VXEEM", "market_index": "^GSPC"
    }
}

# 動態權重設定
# 根據市場處於牛市或熊市，給予不同指標不同的重要性
WEIGHTS_CONFIG = {
    "bull": {'etf_pos': 0.25, 'breadth': 0.35, 'volatility': 0.15, 'trend': 0.25}, # 牛市重寬度同趨勢
    "bear": {'etf_pos': 0.30, 'breadth': 0.15, 'volatility': 0.40, 'trend': 0.15}, # 熊市重波幅同超賣
}

@st.cache_data(ttl=3600) # 快取數據，每小時自動更新一次，避免重複下載
def load_data(tickers, period="3y"):
    """從 Yahoo Finance 下載所有需要的數據"""
    try:
        data = yf.download(list(tickers.values()), period=period, progress=False)
        if data.empty or data['Close'].isnull().all().any(): return None
        return data
    except Exception:
        return None

# --- 2. 核心計算引擎 ---

def calculate_fear_greed_engine(data, tickers, ranking_window=252):
    """執行所有指標計算、政權判斷與動態加權的核心函數"""
    # 準備一個乾淨的 DataFrame
    df = pd.DataFrame(index=data.index)
    df['etf_close'] = data['Close'][tickers['etf']]
    df['equal_weight_close'] = data['Close'][tickers['equal_weight']]
    df['vol_close'] = data['Close'][tickers['volatility']]
    df['market_close'] = data['Close'][tickers['market_index']]
    df = df.dropna()

    # --- a. 判斷市場政權 (Regime) ---
    # 使用大盤指數與其200日均線的關係來定義牛熊
    df['market_ma200'] = df['market_close'].ewm(span=200, adjust=False).mean()
    df['regime'] = np.where(df['market_close'] > df['market_ma200'], 'bull', 'bear')

    # --- b. 計算四大指標原始值 ---
    df['etf_ema200'] = df['etf_close'].ewm(span=200, adjust=False).mean()
    df['etf_pos_raw'] = (df['etf_close'] - df['etf_ema200']) / df['etf_ema200'] * 100
    df['breadth_raw'] = df['equal_weight_close'] / df['etf_close']
    df['vol_raw'] = df['vol_close']
    delta = df['etf_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs_gain_loss = gain / loss
    df['etf_trend_raw'] = 100 - (100 / (1 + rs_gain_loss))
    
    df_clean = df.dropna().copy()

    # --- c. 使用滾動百分位排名進行正規化 ---
    def percentile_rank(series):
        # min_periods 參數防止在數據初期因窗口不足而產生過多 NaN
        return series.rolling(window=ranking_window, min_periods=ranking_window//2).apply(lambda x: percentileofscore(x, x[-1]), raw=False)
    
    df_clean['etf_pos_score'] = percentile_rank(df_clean['etf_pos_raw'])
    df_clean['breadth_score'] = percentile_rank(df_clean['breadth_raw'])
    df_clean['vol_score'] = 100 - percentile_rank(df_clean['vol_raw']) # 反轉波動率分數
    df_clean['trend_score'] = percentile_rank(df_clean['etf_trend_raw'])
    
    df_final = df_clean.dropna().copy()

    # --- d. 應用動態權重 ---
    bull_weights = WEIGHTS_CONFIG['bull']
    bear_weights = WEIGHTS_CONFIG['bear']
    
    # 使用 np.where 根據每日的政權狀態，應用不同的權重組合
    df_final['fear_greed_index'] = np.where(
        df_final['regime'] == 'bull',
        # 牛市權重計算
        (df_final['etf_pos_score'] * bull_weights['etf_pos'] + 
         df_final['breadth_score'] * bull_weights['breadth'] + 
         df_final['vol_score'] * bull_weights['volatility'] + 
         df_final['trend_score'] * bull_weights['trend']),
        # 熊市權重計算
        (df_final['etf_pos_score'] * bear_weights['etf_pos'] + 
         df_final['breadth_score'] * bear_weights['breadth'] + 
         df_final['vol_score'] * bear_weights['volatility'] + 
         df_final['trend_score'] * bear_weights['trend'])
    )

    # --- e. 計算子分數和10日移動平均線 ---
    df_final['health_sub_score'] = (df_final['breadth_score'] + df_final['trend_score']) / 2
    # 風險度反向計算，分數越低風險越高
    df_final['risk_sub_score'] = (df_final['vol_score'] * 0.6 + df_final['etf_pos_score'] * 0.4)
    df_final['index_ma10'] = df_final['fear_greed_index'].rolling(window=10).mean()
    
    return df_final

# --- 3. 繪圖與界面顯示函數 ---

def plot_chart(df, etf_ticker):
    """繪製歷史走勢圖"""
    plt.style.use('seaborn-v0_8-darkgrid')
    # 處理中文字體顯示問題
    try:
        # 優先嘗試蘋果儷黑體
        plt.rcParams['font.sans-serif'] = ['LiHei Pro']
        plt.rcParams['axes.unicode_minus'] = False 
    except:
        try:
            # 兼容 Windows 的微軟正黑體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            # 如果都没有，使用默认字体
            pass

    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    ax1.set_xlabel('日期')
    ax1.set_ylabel(f'{etf_ticker} 收市價', color='blue')
    ax1.plot(df.index, df['etf_close'], color='blue', alpha=0.6, label=f'{etf_ticker} 價格')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('貪婪與恐懼指數 (0-100)', color='red')
    ax2.plot(df.index, df['fear_greed_index'], color='red', label='自訂貪恐指數')
    ax2.plot(df.index, df['index_ma10'], color='orange', linestyle=':', label='10日均線')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax2.axhline(75, color='darkred', linestyle='--', alpha=0.5)
    ax2.axhline(25, color='darkgreen', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 75, 100, color='darkred', alpha=0.1, label='極度貪婪區')
    ax2.fill_between(df.index, 0, 25, color='darkgreen', alpha=0.1, label='極度恐懼區')

    fig.tight_layout()
    plt.title(f'自訂貪婪與恐懼指數 vs. {etf_ticker} 價格 (已啟用動態權重)')
    fig.legend(loc='upper left', bbox_to_anchor=(0.01, 0.95))
    return fig

# --- 4. Streamlit App 主體 ---

st.set_page_config(page_title="專業市場情緒儀表板", page_icon="⚡", layout="wide")
st.title("⚡ 專業市場情緒儀表板 v3.0")
st.caption("引入市場政權模型，實現動態權重調整，並提供更深層次嘅結構分析。")

# --- 側邊欄參數設定 ---
st.sidebar.header("🎯 選擇分析目標")
selected_market = st.sidebar.selectbox("從預設市場中選擇", list(MARKET_PRESETS.keys()), help="選擇一個你感興趣的市場組合，模型會自動加載對應的代碼進行分析。")
tickers = MARKET_PRESETS[selected_market]

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ 查看當前市場嘅代碼"):
    st.json(tickers)

st.sidebar.markdown("---")
st.sidebar.info("模型介紹:\n1. **動態權重**: 自動判斷牛熊市，調整指標重要性。\n2. **子分數**: 分析升勢健康度與潛在風險。")

# --- 主畫面顯示 ---
with st.spinner(f"正在為【{selected_market}】進行深度分析..."):
    raw_data = load_data(tickers)

if raw_data is None:
    st.error(f"無法獲取【{selected_market}】嘅數據，可能部分Ticker已失效或網絡問題。請稍後再試。")
else:
    processed_df = calculate_fear_greed_engine(raw_data, tickers)
    
    if processed_df is None or processed_df.empty:
        st.error("計算過程中出現錯誤，可能是數據不足或格式不符。")
    else:
        latest = processed_df.iloc[-1]
        
        # --- a. 頂部核心指標 ---
        st.header(f"📊 {selected_market} 最新情緒快照")
        
        # 判斷政權與狀態
        current_regime = latest['regime']
        regime_icon = "🐮" if current_regime == 'bull' else "🐻"
        regime_color = "green" if current_regime == 'bull' else "red"
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("自訂貪恐指數", f"{latest['fear_greed_index']:.2f}")
        col2.metric("升勢健康度", f"{latest['health_sub_score']:.2f}", help="由市場寬度與趨勢動能組成，反映升勢質量。")
        col3.metric("市場風險度", f"{latest['risk_sub_score']:.2f}", help="由波幅與超賣程度組成，分數越高風險越低。")
        with col4:
            st.markdown(f"#### <span style='color:{regime_color};'>{regime_icon} 當前政權: {current_regime.capitalize()}</span>", unsafe_allow_html=True)
            st.caption("已啟用對應政權之動態權重")

        # --- b. 深入分析與歷史圖表 (用Tabs分頁) ---
        tab1, tab2, tab3 = st.tabs(["📈 歷史走勢圖", "🔬 指標構成細節", "📖 模型說明"])
        
        with tab1:
            st.pyplot(plot_chart(processed_df, tickers['etf']))

        with tab2:
            st.subheader("🔬 指標構成細節")
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### 原始指標分數 (百分位排名)")
                st.metric("ETF點位分數 (倉位動能)", f"{latest['etf_pos_score']:.1f}")
                st.metric("市場寬度分數 (健康度)", f"{latest['breadth_score']:.1f}")
                st.metric("引伸波幅分數 (恐慌度)", f"{latest['vol_score']:.1f}")
                st.metric("ETF趨勢分數 (RSI動能)", f"{latest['trend_score']:.1f}")
            with c2:
                st.write(f"#### 當前應用權重 ({current_regime.capitalize()} Regime)")
                st.json(WEIGHTS_CONFIG[current_regime])

        with tab3:
            st.markdown("""
            ### 模型設計理念
            這個儀表板旨在提供一個比傳統貪婪恐懼指數更具深度和適應性的市場情緒視角。
            - **市場政權 (Regime Model)**: 我們認為市場在牛市和熊市的行為模式截然不同。因此，模型首先判斷大盤（S&P 500）處於其200日均線之上（牛市）還是之下（熊市），然後應用不同的權重策略。
              - **牛市**: 我們更關注升勢是否健康，因此會加重「市場寬度」和「趨勢」指標的權重。
              - **熊市**: 我們更關注市場的恐慌程度和超賣機會，因此會大幅加重「引伸波幅」和「點位偏離」指標的權重。
            - **子分數 (Sub-Scores)**:
              - **健康度**: 衡量市場上升的質量。如果指數上漲但健康度低，說明升勢可能由少數股票帶動，根基不穩。
              - **風險度**: 衡量市場的潛在波動和拋售壓力。分數越高，代表市場越穩定，風險越低。
            - **數據處理**: 所有指標均採用過去一年（252個交易日）的滾動百分位排名進行正規化，以確保分數的歷史可比性和穩健性。
            """)
st.sidebar.markdown("---")
st.sidebar.markdown("**免責聲明:** 本工具僅為學術研究與概念演示，不構成任何投資建議。")
