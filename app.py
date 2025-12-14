
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顶级量化私募智能投研终端 V3.0 - 技术指标完整版
核心升级：MACD/KDJ/EXPMA/W&R/RSI全技术指标筛选 + 形态识别 + 市场关注度
作者：首席量化工程师
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import pytz
import json
import time as time_module
from openai import OpenAI
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
st.set_page_config(
    page_title="DeepSeek量化终端V3.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {font-family: 'Arial', 'Microsoft YaHei', sans-serif;}
    .stock-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .limit-down {border-color: #ff4444 !important; background: #ffe0e0 !important;}
    .g-signal-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .g1-badge {background: linear-gradient(135deg, #ff6b6b, #ee5a6f);}
    .g2-badge {background: linear-gradient(135deg, #4ecdc4, #44a08d);}
    .g3-badge {background: linear-gradient(135deg, #45b7d1, #3498db);}
    .signal-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
        margin: 0 2px;
    }
    @media (max-width: 768px) {
        .stock-card {padding: 10px; font-size: 14px;}
        h1 {font-size: 22px;}
        h3 {font-size: 16px;}
    }
</style>
""", unsafe_allow_html=True)

TZ = pytz.timezone('Asia/Shanghai')

# ============================================================
# 装饰器：重试机制
# ============================================================
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, pd.DataFrame):
                        if not result.empty:
                            return result
                    else:
                        return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        time_module.sleep(delay * (2 ** attempt))
            return pd.DataFrame()
        return wrapper
    return decorator

# ============================================================
# DeepSeek API
# ============================================================
def get_deepseek_client():
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    except:
        return None

DEEPSEEK_CLIENT = get_deepseek_client()

# ============================================================
# 交易日历
# ============================================================
@st.cache_data(ttl=86400)
@retry_on_failure(max_retries=2)
def get_trade_calendar():
    try:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
        return df['trade_date'].tolist()
    except:
        dates = []
        for i in range(60):
            d = datetime.now(TZ) - timedelta(days=i)
            if d.weekday() < 5:
                dates.append(d.strftime('%Y%m%d'))
        return dates[:30]

def is_trading_time():
    now = datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return (time(9, 30) <= current_time <= time(11, 30)) or \
           (time(13, 0) <= current_time <= time(15, 0))

def get_latest_trade_date():
    calendar = get_trade_calendar()
    today = datetime.now(TZ).strftime('%Y%m%d')
    
    # 如果正在交易时间内，且今天是交易日，直接返回今天
    if is_trading_time() and today in calendar:
        return today
    
    # 否则找出 calendar 中 <= today 的最大（即最新）交易日
    valid_dates = [date for date in calendar if date <= today]
    if valid_dates:
        return max(valid_dates)
    
    # 如果都没找到（极少见），返回日历最后一个日期
    return calendar[-1] if calendar else today

# ============================================================
# 数据获取层
# ============================================================
@st.cache_data(ttl=300)
@retry_on_failure(max_retries=3)
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=300)
@retry_on_failure(max_retries=5, delay=2)  # 增加重试次数，更稳健
def get_all_stocks_realtime():
    """
    获取全A股实时数据（多数据源容错版）
    优先尝试东方财富接口 → 备用新浪接口
    自动映射列名 + 补全缺失列（确保与原代码完全兼容）
    """
    import time as time_module  # 确保已导入
    
    # 数据源顺序：优先东方财富（字段最全），备用新浪
    data_sources = [
        {
            "name": "东方财富",
            "func": lambda: ak.stock_zh_a_spot_em()
        },
        {
            "name": "新浪",
            "func": lambda: ak.stock_zh_a_spot()
        }
    ]
    
    for source in data_sources:
        try:
            df = source["func"]()
            
            if df.empty:
                continue  # 直接尝试下一个源
            
            # ===== 列名映射（根据实际接口返回的中文字段）=====
            if source["name"] == "东方财富":
                column_mapping = {
                    '代码': 'code',
                    '名称': 'name',
                    '最新价': 'price',
                    '涨跌幅': 'pct_chg',
                    '换手率': 'turnover',
                    '量比': 'volume_ratio',
                    '流通市值': 'float_mv',
                    '总市值': 'total_mv',
                    '市盈率-动态': 'pe_ttm',
                    '市净率': 'pb',
                    '今开': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨速': 'speed',
                    '5分钟涨跌': 'pct_5min',
                    '60日涨跌幅': 'pct_60d'
                }
            else:  # 新浪接口（字段较少）
                column_mapping = {
                    '代码': 'code',
                    '名称': 'name',
                    '最新价': 'price',
                    '涨跌幅': 'pct_chg',
                    # 新浪缺少的字段后续统一补全
                }
            
            df = df.rename(columns=column_mapping)
            
            # ===== 确保所有原代码需要的列都存在（缺失补默认值）=====
            required_columns = {
                'code': '',
                'name': 'Unknown',
                'price': 0.0,
                'pct_chg': 0.0,
                'turnover': 0.0,
                'volume_ratio': 1.0,
                'float_mv': 0.0,
                'total_mv': 0.0,
                'pe_ttm': 0.0,
                'pb': 0.0,
                'open': 0.0,
                'high': 0.0,
                'low': 0.0,
                'volume': 0.0,
                'amount': 0.0,
                'amplitude': 0.0,
                'speed': 0.0,
                'pct_5min': 0.0,
                'pct_60d': 0.0,
                # 原代码中后续会模拟的字段
                'pct_5d': 0.0
            }
            
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    df[col] = default_val
            
            # 数值列强制转类型（防止后续计算报错）
            numeric_cols = ['price', 'pct_chg', 'turnover', 'volume_ratio', 
                            'float_mv', 'total_mv', 'pe_ttm', 'pb', 'open', 
                            'high', 'low', 'volume', 'amount', 'pct_5d']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            # 代码清洗（确保6位数字字符串）
            if 'code' in df.columns:
                df['code'] = df['code'].astype(str).str.zfill(6)
            
            return df
            
        except Exception as e:
            # 静默重试（不干扰缓存）
            time_module.sleep(1)
            continue
    
    # 所有源都失败 → 返回空DataFrame（触发原代码的“数据加载失败”提示）
    safety_columns = ['code', 'name', 'price', 'pct_chg', 'turnover', 
                      'volume_ratio', 'float_mv', 'total_mv', 'pe_ttm', 'pb', 
                      'open', 'high', 'low', 'volume', 'amount', 'pct_5d']
    return pd.DataFrame(columns=safety_columns)

@st.cache_data(ttl=14400)
@retry_on_failure(max_retries=3)
def get_stock_history(symbol, period='daily', days=120):
    """
    获取个股历史数据（用于技术指标计算）
    days=120确保有足够数据计算长周期指标（如MACD的26日EMA）
    """
    end_date = datetime.now(TZ).strftime('%Y%m%d')
    start_date = (datetime.now(TZ) - timedelta(days=days)).strftime('%Y%m%d')
    
    df = ak.stock_zh_a_hist(
        symbol=symbol, period=period,
        start_date=start_date, end_date=end_date, adjust="qfq"
    )
    
    if df.empty:
        return pd.DataFrame()
    
    df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 
                  'amount', 'amplitude', 'pct_chg', 'chg', 'turnover']
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=600)
@retry_on_failure(max_retries=2)
def get_north_flow():
    df = ak.stock_hsgt_board_rank_em(symbol="北向资金增持市值", indicator="今日排行")
    return df

@st.cache_data(ttl=3600)
@retry_on_failure(max_retries=2)
def get_stock_hot_rank():
    """
    获取市场关注度（热度排名）
    数据源：东方财富人气榜
    用途：识别市场热点股票
    """
    try:
        df = ak.stock_hot_rank_em()
        return df
    except:
        return pd.DataFrame()

# ============================================================
# 技术指标计算模块（核心）
# 
# 实现的指标：
# 1. MACD：金叉、二次金叉、低位金叉、MACD拐头向上
# 2. KDJ：金叉、二次金叉、低位金叉、拐头向上
# 3. EXPMA：金叉（快线上穿慢线）
# 4. W&R（威廉指标）：超卖反弹
# 5. RSI（相对强弱指标）：超卖反弹
# 6. K线形态：一阳穿三线（一根阳线突破MA5/MA10/MA20）
# 
# 算法说明：
# - 金叉：快线上穿慢线（当日快>慢 且 前日快<慢）
# - 二次金叉：最近N日内出现两次金叉
# - 低位金叉：金叉时指标值在低位区间（MACD<0, KDJ<30）
# - 拐头向上：指标连续3日上升
# ============================================================

def calculate_macd(df, short=12, long=26, signal=9):
    """
    计算MACD指标
    参数：短期EMA=12, 长期EMA=26, 信号线=9（标准参数）
    返回：包含DIF、DEA、MACD柱的DataFrame
    
    计算公式：
    EMA(n) = (2/(n+1)) * 今日收盘价 + (n-1)/(n+1) * 昨日EMA
    DIF = EMA(12) - EMA(26)
    DEA = EMA(9, DIF)
    MACD柱 = 2 * (DIF - DEA)
    """
    if df.empty or len(df) < long + signal:
        return df
    
    df = df.copy()
    
    # 计算短期和长期EMA
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    
    # DIF线
    df['dif'] = df['ema_short'] - df['ema_long']
    
    # DEA线（DIF的9日EMA）
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    
    # MACD柱
    df['macd'] = 2 * (df['dif'] - df['dea'])
    
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    参数：N=9, M1=3, M2=3（标准参数）
    
    计算公式：
    RSV = (收盘价 - N日最低价) / (N日最高价 - N日最低价) * 100
    K = (2/3) * 前日K + (1/3) * 当日RSV
    D = (2/3) * 前日D + (1/3) * 当日K
    J = 3K - 2D
    """
    if df.empty or len(df) < n:
        return df
    
    df = df.copy()
    
    # 计算RSV
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    
    df['rsv'] = (df['close'] - low_list) / (high_list - low_list) * 100
    df['rsv'].fillna(50, inplace=True)
    
    # 计算K、D、J
    df['k'] = df['rsv'].ewm(com=m1-1, adjust=False).mean()
    df['d'] = df['k'].ewm(com=m2-1, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    return df

def calculate_expma(df, short=12, long=50):
    """
    计算EXPMA指标（指数移动平均线）
    参数：短期12日，长期50日
    用途：判断趋势，金叉买入，死叉卖出
    """
    if df.empty or len(df) < long:
        return df
    
    df = df.copy()
    df['expma_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['expma_long'] = df['close'].ewm(span=long, adjust=False).mean()
    
    return df

def calculate_wr(df, n=14):
    """
    计算W&R威廉指标
    参数：N=14（标准参数）
    
    计算公式：
    W&R = (N日最高价 - 当日收盘价) / (N日最高价 - N日最低价) * -100
    
    判断标准：
    W&R < -80：超卖，考虑买入
    W&R > -20：超买，考虑卖出
    """
    if df.empty or len(df) < n:
        return df
    
    df = df.copy()
    
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    
    df['wr'] = (high_list - df['close']) / (high_list - low_list) * -100
    
    return df

def calculate_rsi(df, n=14):
    """
    计算RSI相对强弱指标
    参数：N=14（标准参数）
    
    计算公式：
    RS = N日平均涨幅 / N日平均跌幅
    RSI = 100 - 100/(1+RS)
    
    判断标准：
    RSI < 30：超卖
    RSI > 70：超买
    """
    if df.empty or len(df) < n:
        return df
    
    df = df.copy()
    
    # 计算价格变化
    df['price_change'] = df['close'].diff()
    
    # 分离涨跌
    df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['price_change'].apply(lambda x: -x if x < 0 else 0)
    
    # 计算平均涨跌幅
    avg_gain = df['gain'].rolling(window=n, min_periods=1).mean()
    avg_loss = df['loss'].rolling(window=n, min_periods=1).mean()
    
    # 避免除零错误
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_ma(df, periods=[5, 10, 20, 60]):
    """
    计算多周期移动平均线
    用途：判断趋势和支撑/压力位
    """
    if df.empty:
        return df
    
    df = df.copy()
    for period in periods:
        df[f'ma{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
    
    return df

# ============================================================
# 技术信号识别模块（核心）
# 
# 识别逻辑：
# 1. 金叉：今日快线>慢线 且 昨日快线<慢线
# 2. 二次金叉：最近20日内出现2次金叉
# 3. 低位金叉：金叉时指标处于低位区间
# 4. 拐头向上：连续3日指标上升
# 5. 一阳穿三线：今日阳线且收盘价突破MA5/MA10/MA20
# ============================================================

def detect_macd_signals(df):
    """
    检测MACD信号
    返回：{'golden_cross': bool, 'double_golden': bool, 'low_golden': bool, 'turn_up': bool}
    """
    signals = {
        'macd_golden': False,
        'macd_double_golden': False,
        'macd_low_golden': False,
        'macd_turn_up': False
    }
    
    if df.empty or len(df) < 30:
        return signals
    
    df = calculate_macd(df)
    
    # 金叉：DIF上穿DEA
    if len(df) >= 2:
        today_dif = df['dif'].iloc[-1]
        today_dea = df['dea'].iloc[-1]
        yesterday_dif = df['dif'].iloc[-2]
        yesterday_dea = df['dea'].iloc[-2]
        
        if today_dif > today_dea and yesterday_dif <= yesterday_dea:
            signals['macd_golden'] = True
            
            # 低位金叉：金叉时MACD柱<0
            if df['macd'].iloc[-1] < 0:
                signals['macd_low_golden'] = True
    
    # 二次金叉：最近20日内出现2次金叉
    if len(df) >= 20:
        recent_df = df.tail(20)
        golden_count = 0
        for i in range(1, len(recent_df)):
            if recent_df['dif'].iloc[i] > recent_df['dea'].iloc[i] and \
               recent_df['dif'].iloc[i-1] <= recent_df['dea'].iloc[i-1]:
                golden_count += 1
        
        if golden_count >= 2:
            signals['macd_double_golden'] = True
    
    # 拐头向上：连续3日DIF上升
    if len(df) >= 3:
        if df['dif'].iloc[-1] > df['dif'].iloc[-2] > df['dif'].iloc[-3]:
            signals['macd_turn_up'] = True
    
    return signals

def detect_kdj_signals(df):
    """检测KDJ信号"""
    signals = {
        'kdj_golden': False,
        'kdj_double_golden': False,
        'kdj_low_golden': False,
        'kdj_turn_up': False
    }
    
    if df.empty or len(df) < 15:
        return signals
    
    df = calculate_kdj(df)
    
    # 金叉：K线上穿D线
    if len(df) >= 2:
        today_k = df['k'].iloc[-1]
        today_d = df['d'].iloc[-1]
        yesterday_k = df['k'].iloc[-2]
        yesterday_d = df['d'].iloc[-2]
        
        if today_k > today_d and yesterday_k <= yesterday_d:
            signals['kdj_golden'] = True
            
            # 低位金叉：K<30
            if today_k < 30:
                signals['kdj_low_golden'] = True
    
    # 二次金叉
    if len(df) >= 20:
        recent_df = df.tail(20)
        golden_count = 0
        for i in range(1, len(recent_df)):
            if recent_df['k'].iloc[i] > recent_df['d'].iloc[i] and \
               recent_df['k'].iloc[i-1] <= recent_df['d'].iloc[i-1]:
                golden_count += 1
        
        if golden_count >= 2:
            signals['kdj_double_golden'] = True
    
    # 拐头向上：K线连续3日上升
    if len(df) >= 3:
        if df['k'].iloc[-1] > df['k'].iloc[-2] > df['k'].iloc[-3]:
            signals['kdj_turn_up'] = True
    
    return signals

def detect_expma_golden(df):
    """检测EXPMA金叉"""
    if df.empty or len(df) < 52:
        return False
    
    df = calculate_expma(df)
    
    if len(df) >= 2:
        today_short = df['expma_short'].iloc[-1]
        today_long = df['expma_long'].iloc[-1]
        yesterday_short = df['expma_short'].iloc[-2]
        yesterday_long = df['expma_long'].iloc[-2]
        
        if today_short > today_long and yesterday_short <= yesterday_long:
            return True
    
    return False

def detect_wr_oversold(df, threshold=-80):
    """检测W&R超卖反弹"""
    if df.empty or len(df) < 15:
        return False
    
    df = calculate_wr(df)
    
    # W&R < -80 视为超卖
    if df['wr'].iloc[-1] < threshold:
        return True
    
    return False

def detect_rsi_oversold(df, threshold=30):
    """检测RSI超卖反弹"""
    if df.empty or len(df) < 15:
        return False
    
    df = calculate_rsi(df)
    
    # RSI < 30 视为超卖
    if df['rsi'].iloc[-1] < threshold:
        return True
    
    return False

def detect_one_yang_three_lines(df):
    """
    检测K线形态：一阳穿三线
    定义：今日为阳线（收盘>开盘）且收盘价同时突破MA5、MA10、MA20
    
    判断逻辑：
    1. 今日收盘价 > 开盘价（阳线）
    2. 今日收盘价 > MA5 且 昨日收盘价 <= MA5
    3. 今日收盘价 > MA10 且 昨日收盘价 <= MA10
    4. 今日收盘价 > MA20 且 昨日收盘价 <= MA20
    """
    if df.empty or len(df) < 22:
        return False
    
    df = calculate_ma(df, periods=[5, 10, 20])
    
    if len(df) >= 2:
        # 今日数据
        today_close = df['close'].iloc[-1]
        today_open = df['open'].iloc[-1]
        today_ma5 = df['ma5'].iloc[-1]
        today_ma10 = df['ma10'].iloc[-1]
        today_ma20 = df['ma20'].iloc[-1]
        
        # 昨日数据
        yesterday_close = df['close'].iloc[-2]
        yesterday_ma5 = df['ma5'].iloc[-2]
        yesterday_ma10 = df['ma10'].iloc[-2]
        yesterday_ma20 = df['ma20'].iloc[-2]
        
        # 判断阳线
        is_yang = today_close > today_open
        
        # 判断突破三线
        break_ma5 = today_close > today_ma5 and yesterday_close <= yesterday_ma5
        break_ma10 = today_close > today_ma10 and yesterday_close <= yesterday_ma10
        break_ma20 = today_close > today_ma20 and yesterday_close <= yesterday_ma20
        
        if is_yang and break_ma5 and break_ma10 and break_ma20:
            return True
    
    return False

def calculate_market_attention(code, hot_df):
    """
    计算市场关注度
    基于人气排名、成交额、换手率综合评分
    返回：0-100分
    """
    score = 0
    
    # 人气排名加分
    if not hot_df.empty and '代码' in hot_df.columns:
        if code in hot_df['代码'].values:
            rank = hot_df[hot_df['代码'] == code].index[0] + 1
            # 排名越前分数越高
            score += max(0, 100 - rank)
    
    return min(score, 100)

# ============================================================
# G信号系统
# ============================================================
def init_g_signals():
    if 'g_signals' not in st.session_state:
        st.session_state.g_signals = {
            'G1': {
                'name': 'V型反转',
                'enabled': True,
                'period': 10,
                'stages': [
                    {'type': 'fall', 'pct': -10.0, 'days': 5},
                    {'type': 'rise', 'pct': 8.0, 'days': 2}
                ],
                'logic': 'and'
            }
        }

def detect_g_signal(symbol, g_config, hist_df=None):
    if not g_config['enabled']:
        return False
    
    if hist_df is None or hist_df.empty:
        hist_df = get_stock_history(symbol, period='daily', days=g_config['period'] + 10)
    
    if hist_df.empty or len(hist_df) < g_config['period']:
        return False
    
    recent_df = hist_df.tail(g_config['period']).copy()
    stages = g_config['stages']
    stage_results = []
    
    for stage in stages:
        stage_days = stage.get('days', 1)
        stage_pct = stage.get('pct', 0)
        stage_type = stage.get('type', 'rise')
        
        if len(recent_df) < stage_days:
            stage_results.append(False)
            continue
        
        stage_data = recent_df.tail(stage_days)
        cumulative_pct = ((stage_data['close'].iloc[-1] / stage_data['close'].iloc[0]) - 1) * 100
        
        if stage_type == 'rise':
            pct_match = cumulative_pct >= stage_pct
        else:
            pct_match = cumulative_pct <= stage_pct
        
        stage_results.append(pct_match)
    
    if g_config['logic'] == 'and':
        return all(stage_results)
    else:
        return any(stage_results)

def scan_g_signals_optimized(df_stocks, limit=200):
    results = {}
    g_configs = st.session_state.get('g_signals', {})
    enabled_signals = {k: v for k, v in g_configs.items() if v['enabled']}
    
    if not enabled_signals:
        return results
    
    candidates = df_stocks.head(limit)
    
    for idx, row in candidates.iterrows():
        symbol = row['code']
        max_period = max([g['period'] for g in enabled_signals.values()])
        hist_df = get_stock_history(symbol, period='daily', days=max_period + 10)
        
        matched_signals = []
        for g_id, g_config in enabled_signals.items():
            if detect_g_signal(symbol, g_config, hist_df):
                matched_signals.append(g_id)
        
        if matched_signals:
            results[symbol] = matched_signals
    
    return results

# ============================================================
# 多因子打分（增强版：加入技术指标权重）
# 
# 新的打分逻辑：
# 基础分60分（原有的涨势+量能+估值+资金流）
# 技术指标加分40分：
# - MACD金叉 +5分
# - KDJ金叉 +5分
# - EXPMA金叉 +5分
# - W&R超卖 +5分
# - RSI超卖 +5分
# - 一阳穿三线 +10分
# - 市场关注度 +5分
# ============================================================
def calculate_score_with_technicals(row, north_symbols, tech_signals, hot_df):
    """增强版打分：基础分60 + 技术指标40 + 关注度"""
    score = 0.0  # 强制float，避免类型问题
    
    # 基础分（60分）
    pct_5d = row.get('pct_5d', 0)
    if 3 <= pct_5d <= 15:
        score += 30 * (pct_5d / 15)
    
    volume_ratio = row.get('volume_ratio', 0)
    if volume_ratio > 1.5:
        score += 15
    
    pe = row.get('pe_ttm', 0)
    if 10 <= pe <= 30:
        score += 10
    
    if row.get('code', '') in north_symbols:
        score += 5
    
    # 技术指标加分（40分）
    if tech_signals:  # tech_signals是dict
        if tech_signals.get('macd_golden'):
            score += 5
        if tech_signals.get('macd_low_golden'):
            score += 3
        if tech_signals.get('kdj_golden'):
            score += 5
        if tech_signals.get('kdj_low_golden'):
            score += 3
        if tech_signals.get('expma_golden'):
            score += 5
        if tech_signals.get('wr_oversold'):
            score += 5
        if tech_signals.get('rsi_oversold'):
            score += 5
        if tech_signals.get('one_yang_three_lines'):
            score += 10
    
    # 市场关注度加分（最多5分）
    attention_score = calculate_market_attention(row.get('code', ''), hot_df)
    score += attention_score * 0.05
    
    # 确保返回单个数字！
    return float(min(score, 100))
    
def filter_and_score_with_technicals(df, filters, north_symbols, hot_df, g_results=None):
    """
    筛选、打分、排序（技术指标增强版）
    
    核心优化：
    1. 先应用基础筛选条件（市值、价格等）
    2. 仅对筛选后的股票计算技术指标（减少计算量）
    3. 批量计算技术指标，避免重复获取历史数据
    4. 综合打分排序
    """
    df = df.copy()
    
    # 数值字段清洗
    numeric_cols = ['price', 'pct_chg', 'turnover', 'volume_ratio', 'float_mv', 'pe_ttm', 'pb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 风控：剔除ST
    if filters.get('exclude_st', True):
        df = df[~df['name'].str.contains('ST|退', na=False)]
    
    # 应用基础筛选条件
    min_mv = filters.get('mv_range', [0, 2000])[0]
    max_mv = filters.get('mv_range', [0, 2000])[1]
    min_price = filters.get('price_range', [1, 500])[0]
    max_price = filters.get('price_range', [1, 500])[1]
    min_pct = filters.get('pct_range', [-10, 10])[0]
    max_pct = filters.get('pct_range', [-10, 10])[1]
    
    mask = (
        (df['float_mv'] >= min_mv) &
        (df['float_mv'] <= max_mv) &
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['pct_chg'] >= min_pct) &
        (df['pct_chg'] <= max_pct)
    )
    df = df[mask].copy()
    
    # 模拟近5日涨幅
    df['pct_5d'] = df['pct_chg'] * np.random.uniform(1.2, 2.5, len(df))
    
    # 批量计算技术指标（仅对前500只，避免超时）
    tech_signals_map = {}
    
    for idx, row in df.head(500).iterrows():
        symbol = row['code']
        hist_df = get_stock_history(symbol, period='daily', days=60)
        
        if not hist_df.empty:
            # 计算所有技术信号
            macd_signals = detect_macd_signals(hist_df)
            kdj_signals = detect_kdj_signals(hist_df)
            
            tech_signals_map[symbol] = {
                **macd_signals,
                **kdj_signals,
                'expma_golden': detect_expma_golden(hist_df),
                'wr_oversold': detect_wr_oversold(hist_df),
                'rsi_oversold': detect_rsi_oversold(hist_df),
                'one_yang_three_lines': detect_one_yang_three_lines(hist_df)
            }
        
        # 应用技术指标筛选条件
        signals = tech_signals_map.get(symbol, {})
        
        # MACD筛选
        if filters.get('macd_filter'):
            macd_type = filters.get('macd_type', 'golden')
            if macd_type == 'golden' and not signals.get('macd_golden'):
                df = df[df['code'] != symbol]
                continue
            elif macd_type == 'double_golden' and not signals.get('macd_double_golden'):
                df = df[df['code'] != symbol]
                continue
            elif macd_type == 'low_golden' and not signals.get('macd_low_golden'):
                df = df[df['code'] != symbol]
                continue
            elif macd_type == 'turn_up' and not signals.get('macd_turn_up'):
                df = df[df['code'] != symbol]
                continue
        
        # KDJ筛选
        if filters.get('kdj_filter'):
            kdj_type = filters.get('kdj_type', 'golden')
            if kdj_type == 'golden' and not signals.get('kdj_golden'):
                df = df[df['code'] != symbol]
                continue
            elif kdj_type == 'double_golden' and not signals.get('kdj_double_golden'):
                df = df[df['code'] != symbol]
                continue
            elif kdj_type == 'low_golden' and not signals.get('kdj_low_golden'):
                df = df[df['code'] != symbol]
                continue
            elif kdj_type == 'turn_up' and not signals.get('kdj_turn_up'):
                df = df[df['code'] != symbol]
                continue
        
        # EXPMA筛选
        if filters.get('expma_filter') and not signals.get('expma_golden'):
            df = df[df['code'] != symbol]
            continue
        
        # W&R筛选
        if filters.get('wr_filter') and not signals.get('wr_oversold'):
            df = df[df['code'] != symbol]
            continue
        
        # RSI筛选
        if filters.get('rsi_filter') and not signals.get('rsi_oversold'):
            df = df[df['code'] != symbol]
            continue
        
        # 一阳穿三线筛选
        if filters.get('one_yang_filter') and not signals.get('one_yang_three_lines'):
            df = df[df['code'] != symbol]
            continue
    
    # 综合打分
    df['tech_signals'] = df['code'].map(lambda x: tech_signals_map.get(x, {}))
   # 替换原来的 df['score'] = df.apply(...) 这行
    scores = []
    for _, row in df.iterrows():
        s = calculate_score_with_technicals(row, north_symbols, row.get('tech_signals', {}), hot_df)
        scores.append(float(s))  # 强制float
    df['score'] = scores
    
    # 标注G信号
    if g_results:
        df['g_signals'] = df['code'].map(lambda x: g_results.get(x, []))
    else:
        df['g_signals'] = [[] for _ in range(len(df))]
    
    # 排序
    df = df.sort_values('score', ascending=False)
    
    return df

# ============================================================
# K线图绘制（增强版：显示技术指标）
# ============================================================
def plot_kline_with_indicators(symbol, name, period='daily'):
    """
    绘制K线图 + 技术指标叠加
    包含：K线、MA均线、MACD、KDJ
    """
    if period == '1min':
        df = pd.DataFrame()  # 分时暂不支持指标
        title_suffix = "分时"
    else:
        period_map = {'daily': '日K', 'weekly': '周K', 'monthly': '月K'}
        df = get_stock_history(symbol, period=period, days=120)
        title_suffix = period_map.get(period, '日K')
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    # 计算技术指标
    df = calculate_ma(df, periods=[5, 10, 20, 60])
    df = calculate_macd(df)
    df = calculate_kdj(df)
    
    # 创建子图
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=(f'{name}({symbol}) - {title_suffix}', 'MACD', 'KDJ')
    )
    
    # K线主图
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='red',
        decreasing_line_color='green',
        name='K线'
    ), row=1, col=1)
    
    # MA均线
    colors = ['orange', 'blue', 'purple', 'brown']
    for i, period in enumerate([5, 10, 20, 60]):
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[f'ma{period}'],
            mode='lines',
            name=f'MA{period}',
            line=dict(color=colors[i], width=1)
        ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['dif'],
        mode='lines', name='DIF',
        line=dict(color='blue', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['dea'],
        mode='lines', name='DEA',
        line=dict(color='orange', width=1)
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=df['date'], y=df['macd'],
        name='MACD',
        marker_color=['red' if x > 0 else 'green' for x in df['macd']]
    ), row=2, col=1)
    
    # KDJ
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['k'],
        mode='lines', name='K',
        line=dict(color='blue', width=1)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['d'],
        mode='lines', name='D',
        line=dict(color='orange', width=1)
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['j'],
        mode='lines', name='J',
        line=dict(color='purple', width=1)
    ), row=3, col=1)
    
    # 布局
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig

# ============================================================
# 侧边栏筛选器（完整版：包含所有技术指标）
# ============================================================
def render_sidebar_with_technicals(top10_data, filters):
    """渲染完整的技术指标筛选器"""
    st.sidebar.title("🎯 智能选股筛选器")
    
    # 初始化默认值
    if 'mv_range' not in st.session_state:
        st.session_state.mv_range = [10.0, 1000.0]
    if 'price_range' not in st.session_state:
        st.session_state.price_range = [1.0, 300.0]
    if 'pct_range' not in st.session_state:
        st.session_state.pct_range = [-10.0, 10.0]
    if 'exclude_st' not in st.session_state:
        st.session_state.exclude_st = True
    
    # 基础筛选
    st.sidebar.markdown("### 📊 基础指标")
    mv_range = st.sidebar.slider("流通市值（亿）", 0.0, 2000.0, st.session_state.mv_range)
    price_range = st.sidebar.slider("股价区间（元）", 1.0, 500.0, st.session_state.price_range)
    pct_range = st.sidebar.slider("今日涨跌幅（%）", -10.0, 10.0, st.session_state.pct_range)
    exclude_st = st.sidebar.checkbox("自动剔除ST股", st.session_state.exclude_st)
    
    st.sidebar.markdown("---")
    
    # 技术指标筛选
    st.sidebar.markdown("### 📈 技术指标筛选")
    
    # MACD筛选
    macd_filter = st.sidebar.checkbox("启用MACD筛选", key="macd_filter")
    macd_type = None
    if macd_filter:
        macd_type = st.sidebar.selectbox(
            "MACD类型",
            ["golden", "double_golden", "low_golden", "turn_up"],
            format_func=lambda x: {
                "golden": "金叉",
                "double_golden": "二次金叉",
                "low_golden": "低位金叉",
                "turn_up": "拐头向上"
            }[x]
        )
    
    # KDJ筛选
    kdj_filter = st.sidebar.checkbox("启用KDJ筛选", key="kdj_filter")
    kdj_type = None
    if kdj_filter:
        kdj_type = st.sidebar.selectbox(
            "KDJ类型",
            ["golden", "double_golden", "low_golden", "turn_up"],
            format_func=lambda x: {
                "golden": "金叉",
                "double_golden": "二次金叉",
                "low_golden": "低位金叉",
                "turn_up": "拐头向上"
            }[x]
        )
    
    # EXPMA筛选
    expma_filter = st.sidebar.checkbox("EXPMA金叉", key="expma_filter")
    
    # W&R筛选
    wr_filter = st.sidebar.checkbox("W&R超卖", key="wr_filter")
    
    # RSI筛选
    rsi_filter = st.sidebar.checkbox("RSI超卖", key="rsi_filter")
    
    # 一阳穿三线
    one_yang_filter = st.sidebar.checkbox("一阳穿三线", key="one_yang_filter")
    
    # 市场关注度筛选
    attention_filter = st.sidebar.checkbox("高关注度", key="attention_filter")
    
    # 更新筛选条件
    filters = {
        'mv_range': mv_range,
        'price_range': price_range,
        'pct_range': pct_range,
        'exclude_st': exclude_st,
        'macd_filter': macd_filter,
        'macd_type': macd_type,
        'kdj_filter': kdj_filter,
        'kdj_type': kdj_type,
        'expma_filter': expma_filter,
        'wr_filter': wr_filter,
        'rsi_filter': rsi_filter,
        'one_yang_filter': one_yang_filter,
        'attention_filter': attention_filter
    }
    
    # 更新session_state
    st.session_state.mv_range = mv_range
    st.session_state.price_range = price_range
    st.session_state.pct_range = pct_range
    st.session_state.exclude_st = exclude_st
    
    st.sidebar.markdown("---")
    
    # AI助手（简化版）
    with st.sidebar.expander("🤖 AI助手", expanded=False):
        st.caption("技术指标解读 + G信号生成")
        user_input = st.text_input("输入问题", key="ai_input")
        if st.button("发送", key="ai_send"):
            if user_input and DEEPSEEK_CLIENT:
                st.info("AI功能开发中...")
    
    return filters

# ============================================================
# 主程序
# ============================================================
def main():
    """主程序入口"""
    
    init_g_signals()
    
    st.title("📈 DeepSeek量化投研终端 V3.0")
    st.caption("🚀 技术指标完整版 | MACD/KDJ/EXPMA/W&R/RSI/形态识别")
    
    # 获取交易日期
    target_date = get_latest_trade_date()
    is_trading = is_trading_time()
    
    if is_trading:
        st.success(f"🟢 实时更新中 | {datetime.now(TZ).strftime('%H:%M:%S')}")
    else:
        st.info(f"📅 闭市复盘 | 数据：{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}")
    
    # 加载数据
    with st.spinner("🔄 加载市场数据..."):
        all_stocks = get_all_stocks_realtime()
    
    if all_stocks.empty:
        st.error("❌ 数据加载失败")
        return
    
    # 加载辅助数据
    north_df = get_north_flow()
    north_symbols = set(north_df['代码'].tolist()) if not north_df.empty else set()
    
    hot_df = get_stock_hot_rank()
    
    # 临时筛选条件
    temp_filters = {
        'mv_range': st.session_state.get('mv_range', [10, 1000]),
        'price_range': st.session_state.get('price_range', [1, 300]),
        'pct_range': st.session_state.get('pct_range', [-10, 10]),
        'exclude_st': st.session_state.get('exclude_st', True)
    }
    
    # 扫描G信号
    g_results = {}
    if st.session_state.get('g_signals'):
        with st.spinner("🔍 扫描G信号..."):
            # 先简单筛选，再扫描G信号
            from pandas import DataFrame
            simple_filtered = all_stocks[
                (all_stocks['float_mv'] / 100000000 >= temp_filters['mv_range'][0]) &
                (all_stocks['float_mv'] / 100000000 <= temp_filters['mv_range'][1])
            ]
            g_results = scan_g_signals_optimized(simple_filtered, limit=200)
    
    # 筛选打分（技术指标增强版）
    with st.spinner("📊 计算技术指标..."):
        filtered_df = filter_and_score_with_technicals(
            all_stocks, temp_filters, north_symbols, hot_df, g_results
        )
    
    # Top10
    top10 = filtered_df.head(10).copy()
    top10_data = top10[['code', 'name', 'price', 'pct_chg', 'score']].to_dict('records')
    
    # 渲染侧边栏
    filters = render_sidebar_with_technicals(top10_data, temp_filters)
    
    # 创建Tab
    tab1, tab2 = st.tabs(["🎯 智能选股", "🧪 G信号实验室"])
    
    # ========== Tab1: 智能选股 ==========
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("全市场", f"{len(all_stocks)}")
        with col2:
            st.metric("筛选后", f"{len(filtered_df)}")
        with col3:
            st.metric("命中G信号", f"{len(g_results)}")
        with col4:
            st.metric("更新频率", "10秒" if is_trading else "手动")
        
        if len(filtered_df) == 0:
            st.warning("⚠️ 无符合条件的股票，请调整筛选器")
            return
        
        st.subheader("🏆 今日潜力Top10")
        
        for _, row in top10.iterrows():
            # G信号标签
            g_badges = ""
            for g_id in row.get('g_signals', []):
                g_badges += f'<span class="g-signal-badge">{g_id}</span>'
            
            # 技术信号标签
            tech_signals = row.get('tech_signals', {})
            signal_badges = ""
            if tech_signals.get('macd_golden'):
                signal_badges += '<span class="signal-badge">MACD金叉</span>'
            if tech_signals.get('kdj_golden'):
                signal_badges += '<span class="signal-badge">KDJ金叉</span>'
            if tech_signals.get('expma_golden'):
                signal_badges += '<span class="signal-badge">EXPMA金叉</span>'
            if tech_signals.get('one_yang_three_lines'):
                signal_badges += '<span class="signal-badge">一阳穿三线</span>'
            
            card_class = "stock-card limit-down" if row['pct_chg'] < -9.5 else "stock-card"
            
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col_info, col_chart = st.columns([1, 2])
            
            with col_info:
                st.markdown(
                    f"### {row['name']} ({row['code']}) {g_badges} {signal_badges}",
                    unsafe_allow_html=True
                )
                
                pct_color = "🔴" if row['pct_chg'] < 0 else "🟢"
                st.metric("最新价", f"¥{row['price']:.2f}", f"{row['pct_chg']:.2f}% {pct_color}")
                st.metric("综合评分", f"{row['score']:.1f}分")
                
                st.text(f"换手率: {row['turnover']:.2f}% | 量比: {row['volume_ratio']:.2f}")
                st.text(f"PE: {row['pe_ttm']:.2f} | PB: {row['pb']:.2f}")
                st.text(f"流通市值: {row['float_mv']/100000000:.2f}亿")
            
            with col_chart:
                period_tab = st.radio(
                    "K线周期",
                    ["日K", "周K"],
                    horizontal=True,
                    key=f"period_{row['code']}"
                )
                period_map = {"日K": "daily", "周K": "weekly"}
                fig = plot_kline_with_indicators(
                    row['code'], row['name'], period_map[period_tab]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== Tab2: G信号实验室 ==========
    with tab2:
        st.subheader("🧪 G信号实验室")
        st.info("💡 G信号功能保留，具体实现参考V2.1版本")
    
    # 自动刷新
    if is_trading:
        st.markdown("---")
        st.caption("🔄 自动刷新：10秒")
        time_module.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()





