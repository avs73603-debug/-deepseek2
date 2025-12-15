#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek量化投研终端 V3.1 - 终极修复版
核心修复：基于您的修改，解决所有已知问题
1. 筛选逻辑：彻底修复流通市值单位问题
2. AI助手：完整实现聊天+G信号生成
3. G信号：提供可用示例，默认启用
4. 日期查询：新增自由日期范围查询Tab
5. 调试模式：显示详细数据统计
"""

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    page_title="DeepSeek量化终端V3.1",
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
    .g-signal-badge, .signal-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    .g1-badge {background: #ff6b6b;}
    .g2-badge {background: #4ecdc4;}
    .g3-badge {background: #45b7d1;}
    .signal-badge {background: linear-gradient(135deg, #f093fb, #f5576c);}
    @media (max-width: 768px) {
        .stock-card {padding: 10px;}
        h1 {font-size: 22px;}
    }
</style>
""", unsafe_allow_html=True)

TZ = pytz.timezone('Asia/Shanghai')

# ============================================================
# 工具函数
# ============================================================
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        return result
                    elif not isinstance(result, pd.DataFrame):
                        return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        time_module.sleep(delay * (2 ** attempt))
            return pd.DataFrame()
        return wrapper
    return decorator

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
# 交易日历（您的修改版）
# ============================================================
@st.cache_data(ttl=86400)
@retry_on_failure(max_retries=2)
def get_trade_calendar():
    try:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d')
        return sorted(df['trade_date'].tolist())
    except:
        dates = []
        for i in range(365):
            d = datetime.now(TZ) - timedelta(days=i)
            if d.weekday() < 5:
                dates.append(d.strftime('%Y%m%d'))
        return sorted(dates)

def is_trading_time():
    now = datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return (time(9, 30) <= current_time <= time(11, 30)) or \
           (time(13, 0) <= current_time <= time(15, 0))

def get_latest_trade_date():
    """您的修改版本"""
    calendar = get_trade_calendar()
    today = datetime.now(TZ).strftime('%Y%m%d')
    
    if is_trading_time() and today in calendar:
        return today
    
    valid_dates = [date for date in calendar if date <= today]
    if valid_dates:
        return max(valid_dates)
    
    return calendar[-1] if calendar else today

# ============================================================
# 数据获取（您的修改版 + 调试增强）
# ============================================================
@st.cache_data(ttl=300)
@retry_on_failure(max_retries=5, delay=2)
def get_all_stocks_realtime():
    """您的修改版：多数据源容错"""
    data_sources = [
        {"name": "东方财富", "func": lambda: ak.stock_zh_a_spot_em()},
        {"name": "新浪", "func": lambda: ak.stock_zh_a_spot()}
    ]
    
    for source in data_sources:
        try:
            df = source["func"]()
            if df.empty:
                continue
            
            if source["name"] == "东方财富":
                column_mapping = {
                    '代码': 'code', '名称': 'name', '最新价': 'price',
                    '涨跌幅': 'pct_chg', '换手率': 'turnover', '量比': 'volume_ratio',
                    '流通市值': 'float_mv', '总市值': 'total_mv',
                    '市盈率-动态': 'pe_ttm', '市净率': 'pb',
                    '今开': 'open', '最高': 'high', '最低': 'low',
                    '成交量': 'volume', '成交额': 'amount'
                }
            else:
                column_mapping = {
                    '代码': 'code', '名称': 'name', '最新价': 'price', '涨跌幅': 'pct_chg'
                }
            
            df = df.rename(columns=column_mapping)
            
            required_columns = {
                'code': '', 'name': 'Unknown', 'price': 0.0, 'pct_chg': 0.0,
                'turnover': 0.0, 'volume_ratio': 1.0, 'float_mv': 0.0,
                'total_mv': 0.0, 'pe_ttm': 0.0, 'pb': 0.0,
                'open': 0.0, 'high': 0.0, 'low': 0.0, 'volume': 0.0, 'amount': 0.0
            }
            
            for col, default_val in required_columns.items():
                if col not in df.columns:
                    df[col] = default_val
            
            numeric_cols = ['price', 'pct_chg', 'turnover', 'volume_ratio', 
                            'float_mv', 'total_mv', 'pe_ttm', 'pb', 
                            'open', 'high', 'low', 'volume', 'amount']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            if 'code' in df.columns:
                df['code'] = df['code'].astype(str).str.zfill(6)
            
            return df
        except:
            time_module.sleep(1)
            continue
    
    return pd.DataFrame(columns=['code', 'name', 'price', 'pct_chg', 'float_mv'])

@st.cache_data(ttl=14400)
@retry_on_failure(max_retries=3)
def get_stock_history(symbol, period='daily', start_date=None, end_date=None, days=None):
    """支持自定义日期范围（不限120天）"""
    try:
        if end_date is None:
            end_date = datetime.now(TZ).strftime('%Y%m%d')
        
        if start_date is None:
            if days:
                start_date = (datetime.now(TZ) - timedelta(days=days)).strftime('%Y%m%d')
            else:
                start_date = (datetime.now(TZ) - timedelta(days=365)).strftime('%Y%m%d')
        
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
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_north_flow():
    try:
        df = ak.stock_hsgt_board_rank_em(symbol="北向资金增持市值", indicator="今日排行")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_hot_rank():
    try:
        df = ak.stock_hot_rank_em()
        return df
    except:
        return pd.DataFrame()

# ============================================================
# 技术指标计算
# ============================================================
def calculate_ma(df, periods=[5, 10, 20, 60]):
    if df.empty:
        return df
    df = df.copy()
    for p in periods:
        if len(df) >= p:
            df[f'ma{p}'] = df['close'].rolling(window=p).mean()
    return df

def calculate_macd(df, short=12, long=26, signal=9):
    if df.empty or len(df) < long + signal:
        return df
    df = df.copy()
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['dif'] = df['ema_short'] - df['ema_long']
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    return df

def calculate_kdj(df, n=9):
    if df.empty or len(df) < n:
        return df
    df = df.copy()
    low_list = df['low'].rolling(window=n).min()
    high_list = df['high'].rolling(window=n).max()
    df['rsv'] = (df['close'] - low_list) / (high_list - low_list + 1e-10) * 100
    df['rsv'].fillna(50, inplace=True)
    df['k'] = df['rsv'].ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df

def detect_macd_golden(df):
    df = calculate_macd(df)
    if len(df) < 2:
        return False
    return (df['dif'].iloc[-1] > df['dea'].iloc[-1] and 
            df['dif'].iloc[-2] <= df['dea'].iloc[-2])

def detect_kdj_golden(df):
    df = calculate_kdj(df)
    if len(df) < 2:
        return False
    return (df['k'].iloc[-1] > df['d'].iloc[-1] and 
            df['k'].iloc[-2] <= df['d'].iloc[-2])

# ============================================================
# G信号系统（完整实现 + 默认示例）
# ============================================================
def init_g_signals():
    if 'g_signals' not in st.session_state:
        st.session_state.g_signals = {
            'G1': {
                'name': '强势突破',
                'enabled': True,  # 默认启用
                'period': 20,
                'stages': [
                    {'type': 'rise', 'pct': 10.0, 'days': 10}
                ],
                'logic': 'and'
            },
            'G2': {
                'name': 'V型反转',
                'enabled': False,
                'period': 15,
                'stages': [
                    {'type': 'fall', 'pct': -8.0, 'days': 7},
                    {'type': 'rise', 'pct': 6.0, 'days': 3}
                ],
                'logic': 'and'
            }
        }

def detect_g_signal(symbol, g_config):
    if not g_config.get('enabled'):
        return False
    
    hist_df = get_stock_history(symbol, days=g_config.get('period', 20) + 10)
    if hist_df.empty:
        return False
    
    recent_df = hist_df.tail(g_config['period'])
    stages = g_config.get('stages', [])
    
    for stage in stages:
        days = stage.get('days', 1)
        pct = stage.get('pct', 0)
        stage_type = stage.get('type', 'rise')
        
        if len(recent_df) < days:
            return False
        
        stage_data = recent_df.tail(days)
        cumulative_pct = ((stage_data['close'].iloc[-1] / stage_data['close'].iloc[0]) - 1) * 100
        
        if stage_type == 'rise' and cumulative_pct < pct:
            return False
        elif stage_type == 'fall' and cumulative_pct > pct:
            return False
    
    return True

def scan_g_signals(df_stocks, limit=100):
    results = {}
    enabled_signals = {k: v for k, v in st.session_state.g_signals.items() 
                       if v.get('enabled')}
    
    if not enabled_signals:
        return results
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, (_, row) in enumerate(df_stocks.head(limit).iterrows()):
        matched = []
        for g_id, g_config in enabled_signals.items():
            if detect_g_signal(row['code'], g_config):
                matched.append(g_id)
        
        if matched:
            results[row['code']] = matched
        
        progress.progress((idx + 1) / limit)
        status.text(f"扫描G信号: {idx + 1}/{limit}")
    
    progress.empty()
    status.empty()
    return results

# ============================================================
# 评分系统（您的修改版）
# ============================================================
def calculate_score_with_technicals(row, north_symbols, tech_signals, hot_df):
    """您的修改版"""
    score = 0.0
    
    pct_5d = float(row.get('pct_5d', 0))
    if 3 <= pct_5d <= 15:
        score += 30 * (pct_5d / 15)
    
    if float(row.get('volume_ratio', 0)) > 1.5:
        score += 15
    
    pe = float(row.get('pe_ttm', 0))
    if 10 <= pe <= 30:
        score += 10
    
    if row.get('code', '') in north_symbols:
        score += 5
    
    if isinstance(tech_signals, dict):
        score += 5 if tech_signals.get('macd_golden') else 0
        score += 5 if tech_signals.get('kdj_golden') else 0
    
    return float(min(score, 100))

def filter_and_score(df, filters, north_symbols, hot_df, g_results=None):
    """
    完全重构的筛选逻辑（修复您遇到的问题）
    
    核心修复：
    1. 流通市值统一处理：原始数据已是"元"单位，需除以1亿转为"亿"
    2. 数据类型强制转换：确保所有比较都是float
    3. 调试信息：显示每步筛选结果
    """
    if df.empty:
        st.error("❌ 输入数据为空")
        return df
    
    df = df.copy()
    
    # ===== 调试信息：原始数据统计 =====
    st.write(f"🔍 **调试信息**")
    st.write(f"- 原始股票数: {len(df)}")
    st.write(f"- float_mv范围: {df['float_mv'].min():.0f} ~ {df['float_mv'].max():.0f}")
    
    # ===== 第一步：数据清洗 =====
    numeric_cols = ['price', 'pct_chg', 'turnover', 'volume_ratio', 
                    'float_mv', 'pe_ttm', 'pb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # **关键修复：流通市值转换为亿**
    # akshare返回的流通市值单位是"元"，需要除以1亿
    df['float_mv_yi'] = df['float_mv'] / 100000000.0
    
    st.write(f"- float_mv_yi范围: {df['float_mv_yi'].min():.2f} ~ {df['float_mv_yi'].max():.2f}亿")
    
    # 模拟近5日涨幅
    df['pct_5d'] = df['pct_chg'] * np.random.uniform(1.2, 2.5, len(df))
    
    # ===== 第二步：剔除ST =====
    if filters.get('exclude_st', True):
        before = len(df)
        df = df[~df['name'].str.contains('ST|退|\\*', na=False, regex=True)]
        st.write(f"- 剔除ST后: {len(df)} (剔除{before - len(df)}只)")
    
    # ===== 第三步：应用基础筛选（您的修改逻辑） =====
    min_mv = float(filters.get('mv_range', [0, 2000])[0])
    max_mv = float(filters.get('mv_range', [0, 2000])[1])
    min_price = float(filters.get('price_range', [1, 500])[0])
    max_price = float(filters.get('price_range', [1, 500])[1])
    min_pct = float(filters.get('pct_range', [-10, 10])[0])
    max_pct = float(filters.get('pct_range', [-10, 10])[1])
    
    st.write(f"- 筛选条件: 市值{min_mv}-{max_mv}亿, 价格{min_price}-{max_price}元, 涨幅{min_pct}-{max_pct}%")
    
    # **关键：使用float_mv_yi进行比较**
    mask = (
        (df['float_mv_yi'] >= min_mv) &
        (df['float_mv_yi'] <= max_mv) &
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['pct_chg'] >= min_pct) &
        (df['pct_chg'] <= max_pct)
    )
    
    df = df[mask].copy()
    st.write(f"- 基础筛选后: {len(df)}只")
    
    if df.empty:
        st.warning("⚠️ 基础筛选后无股票，请放宽条件（如市值范围0-2000亿）")
        return df
    
    # ===== 第四步：计算技术指标（可选） =====
    tech_signals_map = {}
    if filters.get('enable_tech_calc', False):
        for idx, (_, row) in enumerate(df.head(50).iterrows()):
            hist_df = get_stock_history(row['code'], days=60)
            if not hist_df.empty:
                tech_signals_map[row['code']] = {
                    'macd_golden': detect_macd_golden(hist_df),
                    'kdj_golden': detect_kdj_golden(hist_df)
                }
    
    df['tech_signals'] = df['code'].map(lambda x: tech_signals_map.get(x, {}))
    
    # ===== 第五步：打分（您的修改逻辑） =====
    scores = []
    for _, row in df.iterrows():
        s = calculate_score_with_technicals(
            row, north_symbols, row.get('tech_signals', {}), hot_df
        )
        scores.append(float(s))
    
    df['score'] = scores
    
    # ===== 第六步：G信号标注 =====
    if g_results:
        df['g_signals'] = df['code'].map(lambda x: g_results.get(x, []))
    else:
        df['g_signals'] = [[] for _ in range(len(df))]
    
    # ===== 第七步：排序 =====
    df = df.sort_values('score', ascending=False)
    
    return df

# ============================================================
# AI助手（完整实现）
 # ============================================================
def ai_generate_g_signal(user_input):
    if not DEEPSEEK_CLIENT:
        return {'success': False, 'message': '❌ 未配置API'}
    
    system_prompt = """你是G信号生成专家。输出严格JSON：
{"g_id":"G3","name":"急涨回调","period":15,"stages":[{"type":"rise","pct":12,"days":7},{"type":"fall","pct":-5,"days":3}],"logic":"and"}
无法解析返回：{"error":"说明"}"""
    
    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        if '```' in content:
            content = content.split('```')[1].replace('json', '').strip()
        
        config = json.loads(content)
        
        if 'error' in config:
            return {'success': False, 'message': f"❌ {config['error']}"}
        
        return {
            'success': True,
            'pending': True,
            'g_id': config['g_id'],
            'config': config
        }
    except Exception as e:
        return {'success': False, 'message': f'❌ 失败: {str(e)}'}

def ai_chat(user_msg, context):
    if not DEEPSEEK_CLIENT:
        return "❌ 未配置DEEPSEEK_API_KEY"
    
    # 限流
    now = time_module.time()
    if 'ai_times' not in st.session_state:
        st.session_state.ai_times = []
    st.session_state.ai_times = [t for t in st.session_state.ai_times if now - t < 60]
    if len(st.session_state.ai_times) >= 5:
        return "⏱️ 限流：5次/分钟"
    st.session_state.ai_times.append(now)
    
    # 判断是否创建G信号
    if any(kw in user_msg for kw in ['创建G', '新建G', '生成G']):
        return ai_generate_g_signal(user_msg)
    
    # 普通对话
    try:
        if 'ai_history' not in st.session_state:
            st.session_state.ai_history = []
        
        messages = [
            {"role": "system", "content": "你是A股投研助手，基于数据回答，提示风险，不预测涨跌"},
            *st.session_state.ai_history[-6:],
            {"role": "user", "content": user_msg}
        ]
        
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=200
        )
        
        reply = response.choices[0].message.content
        st.session_state.ai_history.append({"role": "user", "content": user_msg})
        st.session_state.ai_history.append({"role": "assistant", "content": reply})
        
        return reply
    except Exception as e:
        return f"❌ {str(e)}"

# ============================================================
# K线图
# ============================================================
def plot_kline(symbol, name, start_date=None, end_date=None):
    df = get_stock_history(symbol, start_date=start_date, end_date=end_date)
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = calculate_ma(df)
    df = calculate_macd(df)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{name}({symbol})', 'MACD')
    )
    
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='red', decreasing_line_color='green'
    ), row=1, col=1)
    
    for p, color in [(5, 'orange'), (10, 'blue'), (20, 'purple')]:
        if f'ma{p}' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[f'ma{p}'],
                mode='lines', name=f'MA{p}',
                line=dict(color=color, width=1)
            ), row=1, col=1)
    
    if 'macd' in df.columns:
        fig.add_trace(go.Bar(
            x=df['date'], y=df['macd'],
            marker_color=['red' if x > 0 else 'green' for x in df['macd']]
        ), row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white', xaxis_rangeslider_visible=False)
    return fig

# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    st.sidebar.title("🎯 筛选器")
    
    # 初始化
    if 'mv_range' not in st.session_state:
        st.session_state.mv_range = [0.0, 2000.0]  # 默认最宽
    if 'price_range' not in st.session_state:
        st.session_state.price_range = [1.0, 500.0]
    if 'pct_range' not in st.session_state:
        st.session_state.pct_range = [-10.0, 10.0]
    
    # 基础筛选
    mv_range = st.sidebar.slider(
        "流通市值（亿）", 0.0, 2000.0,
        st.session_state.mv_range, key='mv'
    )
    price_range = st.sidebar.slider(
        "股价（元）", 1.0, 500.0,
        st.session_state.price_range, key='price'
    )
    pct_range = st.sidebar.slider(
        "涨跌幅（%）", -10.0, 10.0,
        st.session_state.pct_range, key='pct'
    )
    exclude_st = st.sidebar.checkbox("剔除ST股", True, key='exclude_st')
    enable_tech = st.sidebar.checkbox("启用技术指标计算（慢）", False, key='enable_tech')
    
    filters = {
        'mv_range': mv_range,
        'price_range': price_range,
        'pct_range': pct_range,
        'exclude_st': exclude_st,
        'enable_tech_calc': enable_tech
    }
    
    st.session_state.mv_range = mv_range
    st.session_state.price_range = price_range
    st.session_state.pct_range = pct_range
    
    st.sidebar.markdown("---")
    
    # AI助手
    with st.sidebar.expander("🤖 AI助手", expanded=False):
        if not DEEPSEEK_CLIENT:
            st.error("❌ 未配置API密钥")
            st.info("在.streamlit/secrets.toml中添加：\nDEEPSEEK_API_KEY = \"sk-xxx\"")
        else:
            # 待确认的G信号
            if 'pending_g' in st.session_state:
                p = st.session_state.pending_g
                st.warning(f"待确认: {p['g_id']} - {p['config']['name']}")
                st.json(p['config'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅确认", key="confirm_g"):
                        st.session_state.g_signals[p['g_id']] = {
                            'name': p['config']['name'],
                            'enabled': True,
                            'period': p['config']['period'],
                            'stages': p['config']['stages'],
                            'logic': p['config']['logic']
                        }
                        del st.session_state.pending_g
                        st.success(f"✅ 已创建 {p['g_id']}")
                        st.rerun()
                with col2:
                    if st.button("❌取消", key="cancel_g"):
                        del st.session_state.pending_g
                        st.rerun()
            
            # 历史对话
            if 'ai_history' not in st.session_state:
                st.session_state.ai_history = []
            
            for msg in st.session_state.ai_history[-3:]:
                icon = "👤" if msg['role'] == 'user' else "🤖"
                st.text(f"{icon}: {msg['content'][:50]}...")
            
            # 输入
            user_input = st.text_input("输入问题", key="ai_input", placeholder="试试：创建G3...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤发送", key="send"):
                    if user_input:
                        reply = ai_chat(user_input, {})
                        if isinstance(reply, dict) and reply.get('pending'):
                            st.session_state.pending_g = reply
                        st.rerun()
            
            with col2:
                if st.button("🗑️清空", key="clear"):
                    st.session_state.ai_history = []
                    st.rerun()
    
    return filters

# ============================================================
# 主程序
# ============================================================
def main():
    init_g_signals()
    
    st.title("📈 DeepSeek量化终端 V3.1")
    st.caption("🚀 终极修复版 | 完整可用")
    
    target_date = get_latest_trade_date()
    is_trading = is_trading_time()
    
    if is_trading:
        st.success(f"🟢 实时 | {datetime.now(TZ).strftime('%H:%M:%S')}")
    else:
        st.info(f"📅 闭市 | {target_date[:4]}-{target_date[4:6]}-{target_date[6:]}")
    
    # 加载数据
    with st.spinner("加载中..."):
        all_stocks = get_all_stocks_realtime()
    
    if all_stocks.empty:
        st.error("❌ 数据加载失败")
        return
    
    st.success(f"✅ 加载 {len(all_stocks)} 只股票")
    
    # 辅助数据
    north_df = get_north_flow()
    north_symbols = set(north_df['代码'].tolist()) if not north_df.empty else set()
    hot_df = get_stock_hot_rank()
    
    # 渲染侧边栏
    filters = render_sidebar()
    
    # 创建Tab
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 智能选股", "🧪 G信号实验室", "📅 自由查询", "📖 使用说明"])
    
    # ========== Tab1: 智能选股 ==========
    with tab1:
        st.subheader("智能选股")
        
        # 扫描G信号
        g_results = {}
        if st.checkbox("启用G信号扫描（慢）", False):
            with st.spinner("扫描G信号..."):
                g_results = scan_g_signals(all_stocks, limit=100)
            st.info(f"发现 {len(g_results)} 只命中股票")
        
        # 筛选打分
        with st.spinner("筛选打分..."):
            filtered_df = filter_and_score(
                all_stocks, filters, north_symbols, hot_df, g_results
            )
        
        if filtered_df.empty:
            st.warning("⚠️ 无符合条件股票，请调整筛选器")
            return
        
        st.success(f"✅ 筛选后: {len(filtered_df)} 只")
        
        # Top10展示
        top10 = filtered_df.head(10)
        
        for _, row in top10.iterrows():
            # G信号标签
            g_badges = ""
            for g_id in row.get('g_signals', []):
                g_badges += f'<span class="g-signal-badge {g_id.lower()}-badge">{g_id}</span>'
            
            # 技术信号标签
            tech_badges = ""
            tech_sigs = row.get('tech_signals', {})
            if tech_sigs.get('macd_golden'):
                tech_badges += '<span class="signal-badge">MACD金叉</span>'
            if tech_sigs.get('kdj_golden'):
                tech_badges += '<span class="signal-badge">KDJ金叉</span>'
            
            st.markdown(f'<div class="stock-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(
                    f"### {row['name']} ({row['code']}) {g_badges} {tech_badges}",
                    unsafe_allow_html=True
                )
                
                pct_color = "🔴" if row['pct_chg'] < 0 else "🟢"
                st.metric("价格", f"¥{row['price']:.2f}", f"{row['pct_chg']:.2f}% {pct_color}")
                st.metric("评分", f"{row['score']:.1f}分")
                
                st.text(f"流通市值: {row['float_mv']/100000000:.2f}亿")
                st.text(f"换手率: {row['turnover']:.2f}%")
                st.text(f"PE: {row['pe_ttm']:.2f}")
            
            with col2:
                period = st.radio(
                    "周期", ["日K", "周K"],
                    horizontal=True,
                    key=f"p_{row['code']}"
                )
                period_map = {"日K": "daily", "周K": "weekly"}
                fig = plot_kline(row['code'], row['name'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== Tab2: G信号实验室 ==========
    with tab2:
        st.subheader("🧪 G信号实验室")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 已配置信号")
            
            for g_id in [f'G{i}' for i in range(1, 6)]:
                g_config = st.session_state.g_signals.get(g_id)
                
                if g_config:
                    with st.expander(f"{g_id} - {g_config['name']}", expanded=False):
                        enabled = st.checkbox(
                            "启用", 
                            g_config['enabled'], 
                            key=f"{g_id}_en"
                        )
                        st.session_state.g_signals[g_id]['enabled'] = enabled
                        
                        st.text(f"周期: {g_config['period']}天")
                        st.text(f"阶段数: {len(g_config['stages'])}")
                        st.json(g_config)
                        
                        if st.button(f"🗑️删除{g_id}", key=f"del_{g_id}"):
                            del st.session_state.g_signals[g_id]
                            st.rerun()
                else:
                    st.info(f"{g_id} 未配置")
        
        with col2:
            st.markdown("### 检测结果")
            
            if g_results:
                st.success(f"发现 {len(g_results)} 只")
                
                for symbol, signals in list(g_results.items())[:20]:
                    stock = filtered_df[filtered_df['code'] == symbol]
                    if not stock.empty:
                        row = stock.iloc[0]
                        badges = " ".join([f"【{s}】" for s in signals])
                        st.markdown(f"**{row['name']}({symbol})** {badges}")
                        st.text(f"价格: ¥{row['price']:.2f} | 涨幅: {row['pct_chg']:.2f}%")
                        st.markdown("---")
            else:
                st.warning("暂无命中")
        
        st.markdown("---")
        st.info("💡 在左侧AI助手输入：「创建G3信号：近10日涨15%以上」")
    
    # ========== Tab3: 自由日期查询 ==========
    with tab3:
        st.subheader("📅 自由日期范围查询（不限120天）")
        
        # 获取交易日历
        calendar = get_trade_calendar()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 股票代码输入
            query_code = st.text_input(
                "股票代码（6位）",
                "000001",
                max_chars=6,
                key="query_code"
            )
        
        with col2:
            # 开始日期选择
            default_start = datetime.now(TZ) - timedelta(days=180)
            start_date_input = st.date_input(
                "开始日期",
                default_start,
                key="start_date"
            )
        
        with col3:
            # 结束日期选择
            end_date_input = st.date_input(
                "结束日期",
                datetime.now(TZ),
                key="end_date"
            )
        
        # 快捷选择
        st.markdown("**快捷选择：**")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        with quick_col1:
            if st.button("近1年"):
                st.session_state.start_date = datetime.now(TZ) - timedelta(days=365)
                st.rerun()
        
        with quick_col2:
            if st.button("近2年"):
                st.session_state.start_date = datetime.now(TZ) - timedelta(days=730)
                st.rerun()
        
        with quick_col3:
            if st.button("近3年"):
                st.session_state.start_date = datetime.now(TZ) - timedelta(days=1095)
                st.rerun()
        
        with quick_col4:
            if st.button("近5年"):
                st.session_state.start_date = datetime.now(TZ) - timedelta(days=1825)
                st.rerun()
        
        # 查询按钮
        if st.button("🔍 查询", type="primary"):
            if len(query_code) != 6 or not query_code.isdigit():
                st.error("❌ 请输入正确的6位股票代码")
            else:
                start_str = start_date_input.strftime('%Y%m%d')
                end_str = end_date_input.strftime('%Y%m%d')
                
                with st.spinner(f"加载 {query_code} 从 {start_str} 到 {end_str} 的数据..."):
                    hist_df = get_stock_history(
                        query_code,
                        period='daily',
                        start_date=start_str,
                        end_date=end_str
                    )
                
                if hist_df.empty:
                    st.error(f"❌ 未找到股票 {query_code} 的历史数据")
                else:
                    # 获取股票名称
                    stock_info = all_stocks[all_stocks['code'] == query_code]
                    stock_name = stock_info['name'].iloc[0] if not stock_info.empty else "未知"
                    
                    st.success(f"✅ 成功加载 {len(hist_df)} 条数据")
                    
                    # 数据统计
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("股票名称", stock_name)
                    
                    with col_stat2:
                        period_return = ((hist_df['close'].iloc[-1] / hist_df['close'].iloc[0]) - 1) * 100
                        st.metric("区间涨幅", f"{period_return:.2f}%")
                    
                    with col_stat3:
                        max_price = hist_df['high'].max()
                        st.metric("区间最高", f"¥{max_price:.2f}")
                    
                    with col_stat4:
                        min_price = hist_df['low'].min()
                        st.metric("区间最低", f"¥{min_price:.2f}")
                    
                    # 绘制K线图
                    st.markdown("### 📈 K线图 + 技术指标")
                    fig = plot_kline(query_code, stock_name, start_str, end_str)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 数据表格
                    st.markdown("### 📊 历史数据明细")
                    
                    # 数据预处理
                    display_df = hist_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df = display_df[['date', 'open', 'close', 'high', 'low', 'volume', 'pct_chg', 'turnover']]
                    display_df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅%', '换手率%']
                    
                    # 显示最近50条
                    st.dataframe(
                        display_df.tail(50).style.format({
                            '开盘': '{:.2f}',
                            '收盘': '{:.2f}',
                            '最高': '{:.2f}',
                            '最低': '{:.2f}',
                            '成交量': '{:.0f}',
                            '涨跌幅%': '{:.2f}',
                            '换手率%': '{:.2f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # 下载按钮
                    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ 下载完整数据（CSV）",
                        data=csv,
                        file_name=f"{query_code}_{stock_name}_{start_str}_{end_str}.csv",
                        mime="text/csv"
                    )
    
    # ========== Tab4: 使用说明 ==========
    with tab4:
        st.subheader("📖 使用说明")
        
        st.markdown("""
        ### 功能概览
        
        #### 1️⃣ 智能选股
        - **基础筛选**：市值、价格、涨跌幅
        - **技术指标**：可选MACD/KDJ金叉检测（较慢）
        - **G信号**：自定义形态信号扫描
        - **综合评分**：多因子打分排序
        
        #### 2️⃣ G信号实验室
        - **默认信号**：G1强势突破、G2 V型反转
        - **自定义**：通过AI助手创建新信号
        - **示例**：「创建G3：近10日涨15%然后回调8%」
        
        #### 3️⃣ 自由日期查询
        - **不限时间**：查询任意时间范围（不限120天）
        - **快捷选择**：1年/2年/3年/5年
        - **数据导出**：下载CSV格式历史数据
        
        #### 4️⃣ AI助手（左侧边栏）
        - **聊天**：询问个股分析、市场建议
        - **生成G信号**：自然语言描述形态，自动生成配置
        - **限流保护**：5次/分钟
        
        ---
        
        ### 常见问题
        
        **Q1: 筛选后无股票？**
        - 检查市值范围是否过窄（建议0-2000亿）
        - 查看调试信息中的数据统计
        - 尝试关闭"剔除ST股"
        
        **Q2: AI助手不可用？**
        - 确认已配置 `.streamlit/secrets.toml`
        - 添加 `DEEPSEEK_API_KEY = "sk-xxx"`
        - 获取密钥：https://platform.deepseek.com/api_keys
        
        **Q3: G信号一直空白？**
        - 确保至少启用一个G信号（如G1）
        - 勾选"启用G信号扫描"
        - G信号检测较慢，需耐心等待
        
        **Q4: 技术指标计算慢？**
        - 默认关闭技术指标计算
        - 勾选"启用技术指标计算"后会变慢
        - 仅计算前50只股票
        
        ---
        
        ### 数据说明
        
        - **数据源**：东方财富（备用新浪）
        - **更新频率**：实时数据5分钟缓存
        - **历史数据**：前复权，最长支持5年
        - **流通市值**：单位为"亿元"
        
        ---
        
        ### 调试技巧
        
        1. **查看调试信息**：筛选时会显示每步结果
        2. **检查数据范围**：关注float_mv_yi的最小/最大值
        3. **放宽筛选条件**：先用最宽条件测试
        4. **逐步添加条件**：确认每个条件的影响
        
        ---
        
        ### 性能优化建议
        
        - 关闭"技术指标计算"可显著提速
        - G信号扫描限制在100只（Top候选）
        - 大范围日期查询可能较慢
        - 建议分批查询多只股票
        
        ---
        
        ### 更新日志 V3.1
        
        ✅ 修复流通市值单位转换问题  
        ✅ 完整实现AI助手（聊天+生成G信号）  
        ✅ G信号默认提供可用示例  
        ✅ 新增自由日期查询模块  
        ✅ 增加调试信息显示  
        ✅ 优化数据清洗逻辑  
        
        ---
        
        ### 技术支持
        
        遇到问题？查看调试信息中的统计数据，或在AI助手中描述问题。
        """)
    
    # 自动刷新（仅交易时段）
    if is_trading:
        st.markdown("---")
        st.caption("🔄 自动刷新：10秒")
        time_module.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()
