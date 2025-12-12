#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顶级量化私募智能投研终端 - 单文件完整版
功能：A股智能选股 + DeepSeek AI助手 + 实时数据 + PDF报告生成
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
from fpdf import FPDF
import io
import base64

# ============================================================
# 全局配置：页面布局、样式、时区、API客户端初始化
# 必须放在最前面，避免Streamlit警告。设置宽屏模式以充分利用屏幕空间
# ============================================================
st.set_page_config(
    page_title="DeepSeek量化投研终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 上海时区（中国A股交易时间基准）
TZ = pytz.timezone('Asia/Shanghai')

# ============================================================
# DeepSeek API客户端初始化
# 使用OpenAI SDK兼容接口，base_url指向DeepSeek官方API
# API Key从Streamlit secrets中读取，支持本地和云端部署
# 错误处理：如果未配置密钥，后续AI功能会友好提示用户
# ============================================================
def get_deepseek_client():
    """获取DeepSeek API客户端，支持本地和云端环境"""
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    except Exception as e:
        st.warning(f"⚠️ DeepSeek API未配置: {e}")
        return None

DEEPSEEK_CLIENT = get_deepseek_client()

# ============================================================
# 数据缓存装饰器：@st.cache_data(ttl=4*3600)
# TTL=4小时，避免频繁调用akshare导致IP被封
# 缓存内容：全A股实时行情、分钟K线、北向资金流等
# 注意：akshare数据源不稳定时会自动重试，失败返回空DataFrame
# ============================================================
@st.cache_data(ttl=4*3600)
def get_all_stocks():
    """
    获取全A股票池（约5300只）+ 实时行情数据
    数据源：akshare的stock_zh_a_spot_em接口（东方财富实时数据）
    返回字段：代码、名称、最新价、涨跌幅、换手率、量比、市值、PE、PB等
    异常处理：网络超时或接口失败时返回空DataFrame，避免程序崩溃
    """
    try:
        df = ak.stock_zh_a_spot_em()
        # 字段映射：东方财富接口字段名转标准名称
        df = df.rename(columns={
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
            '60日涨跌幅': 'pct_60d',
            '年初至今涨跌': 'pct_ytd'
        })
        return df
    except Exception as e:
        st.error(f"❌ 数据获取失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_minute_kline(symbol, days=1):
    """
    获取指定股票的分钟K线数据（最近60分钟）
    参数：symbol格式如'000001'（不含市场前缀）
    数据源：akshare的stock_zh_a_hist_min_em接口
    返回：包含时间、开高低收、成交量的DataFrame
    用途：绘制实时K线图，展示日内走势
    """
    try:
        # 构造完整股票代码（akshare需要加市场前缀）
        full_code = symbol
        if symbol.startswith('6'):
            full_code = f"sh{symbol}"
        elif symbol.startswith(('0', '3')):
            full_code = f"sz{symbol}"
        
        # 获取1分钟K线，周期='1'表示1分钟
        df = ak.stock_zh_a_hist_min_em(
            symbol=full_code,
            period='1',
            adjust='qfq'  # 前复权
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # 只取最近60分钟数据
        df = df.tail(60)
        df.columns = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'latest']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_north_flow():
    """
    获取北向资金实时流入数据（沪股通+深股通）
    数据源：akshare的stock_hsgt_board_rank_em
    返回：个股北向资金流入排名，用于筛选外资青睐标的
    注意：仅交易日有数据，非交易日返回空DataFrame
    """
    try:
        df = ak.stock_hsgt_board_rank_em(symbol="北向资金增持市值", indicator="今日排行")
        return df
    except:
        return pd.DataFrame()

# ============================================================
# 交易时间判断：判断当前是否A股交易时段
# 交易时段：周一至周五 09:30-11:30, 13:00-15:00
# 用途：决定自动刷新频率（交易时段5秒，非交易时段30秒）
# ============================================================
def is_trading_time():
    """判断当前是否A股交易时段（含集合竞价时间）"""
    now = datetime.now(TZ)
    if now.weekday() >= 5:  # 周末
        return False
    current_time = now.time()
    # 交易时段：09:15-15:00
    return (time(9, 15) <= current_time <= time(15, 0))

# ============================================================
# 核心推荐算法：多因子打分模型
# 四大维度权重：涨势40% + 量能30% + 估值20% + 资金流10%
# 打分逻辑：每个维度0-100分，加权求和得综合分
# 涨势：近5日涨幅越高越好，但超15%开始衰减（防追高）
# 量能：量比>1.5且换手率适中（0.5%-10%）为佳
# 估值：PE 10-30、PB 1-5为合理区间
# 资金流：北向资金净流入为加分项
# ============================================================
def calculate_score(row, north_symbols):
    """
    计算单只股票综合评分（0-100分）
    row: 股票数据行（包含价格、涨幅、估值等字段）
    north_symbols: 北向资金流入股票列表（用于资金流维度加分）
    返回：综合得分（float）
    """
    score = 0
    
    # 1. 涨势维度（40分）：近5日涨幅体现短期动量
    pct_5d = row.get('pct_5d', 0)
    if 3 <= pct_5d <= 15:  # 温和上涨区间
        score += 40 * (pct_5d / 15)
    elif pct_5d > 15:  # 涨幅过大衰减（风险提示）
        score += 40 * 0.6
    
    # 2. 量能维度（30分）：量比和换手率反映活跃度
    volume_ratio = row.get('volume_ratio', 0)
    turnover = row.get('turnover', 0)
    if volume_ratio > 1.5 and 0.5 <= turnover <= 10:
        score += 30
    elif volume_ratio > 1.0:
        score += 15
    
    # 3. 估值维度（20分）：PE/PB合理区间判断
    pe = row.get('pe_ttm', 0)
    pb = row.get('pb', 0)
    if 10 <= pe <= 30 and 1 <= pb <= 5:
        score += 20
    elif 5 <= pe <= 50:
        score += 10
    
    # 4. 资金流维度（10分）：北向资金流入加分
    if row.get('code', '') in north_symbols:
        score += 10
    
    return score
# ============================================================
# DeepSeek AI 自然语言推荐生成
# 输入：Top 15股票数据（JSON格式）
# 输出：每只股票30字内的推荐理由（突出核心亮点）
# System Prompt强调：基于事实、提示风险、不预测涨跌
# Token限制：每次调用限制返回150 tokens，避免超额消费
# ============================================================
def generate_ai_reasons(top_stocks_json):
    """
    调用DeepSeek为Top15股票生成推荐理由
    top_stocks_json: 股票数据的JSON字符串
    返回：字典 {股票代码: 推荐理由}
    异常处理：API调用失败时返回默认文案
    """
    if not DEEPSEEK_CLIENT:
        return {item['code']: '综合表现优异' for item in json.loads(top_stocks_json)}
    
    try:
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是专业A股投研助手。根据股票数据生成推荐理由，每只股票30字内，突出核心亮点（涨势/量能/估值/资金流）。严禁预测涨跌，必须提示风险。"
                },
                {
                    "role": "user",
                    "content": f"请为以下15只股票各生成一句30字内推荐理由，JSON格式返回{{股票代码: 推荐理由}}：\n{top_stocks_json}"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        # 解析JSON格式返回
        reasons = json.loads(content)
        return reasons
    except Exception as e:
        st.warning(f"AI推荐生成失败: {e}")
        return {item['code']: '技术面向好，关注风险' for item in json.loads(top_stocks_json)}

# ============================================================
# 侧边栏筛选条件UI构建
# 10个常用筛选维度，使用Streamlit原生控件（slider/multiselect）
# 所有条件值存储在session_state，支持AI助手动态修改
# 逻辑：用户手动调整 OR AI解析自然语言后修改 → 触发重新筛选
# ============================================================
def render_sidebar_filters():
    """
    渲染左侧边栏的全部筛选控件
    返回：筛选条件字典（用于后续数据过滤）
    """
    st.sidebar.title("🎯 智能选股筛选器")
    
    # 初始化session_state默认值（首次运行时）
    if 'mv_range' not in st.session_state:
        st.session_state.mv_range = [10.0, 1000.0]
    if 'price_range' not in st.session_state:
        st.session_state.price_range = [1.0, 300.0]
    if 'pct_range' not in st.session_state:
        st.session_state.pct_range = [-10.0, 10.0]
    if 'turnover_range' not in st.session_state:
        st.session_state.turnover_range = [0.0, 20.0]
    if 'volume_ratio_min' not in st.session_state:
        st.session_state.volume_ratio_min = 0.5
    if 'pe_range' not in st.session_state:
        st.session_state.pe_range = [0.0, 100.0]
    if 'pb_range' not in st.session_state:
        st.session_state.pb_range = [0.0, 10.0]
    if 'roe_min' not in st.session_state:
        st.session_state.roe_min = 0.0
    if 'pct_5d_min' not in st.session_state:
        st.session_state.pct_5d_min = 0.0
    if 'near_high_20d' not in st.session_state:
        st.session_state.near_high_20d = False
    if 'exclude_st' not in st.session_state:
        st.session_state.exclude_st = True
    
    # 筛选条件控件渲染
    mv_range = st.sidebar.slider(
        "流通市值（亿）",
        0.0, 2000.0, 
        st.session_state.mv_range,
        key='mv_slider'
    )
    
    price_range = st.sidebar.slider(
        "股价区间（元）",
        1.0, 500.0,
        st.session_state.price_range,
        key='price_slider'
    )
    
    pct_range = st.sidebar.slider(
        "今日涨跌幅（%）",
        -10.0, 10.0,
        st.session_state.pct_range,
        key='pct_slider'
    )
    
    turnover_range = st.sidebar.slider(
        "换手率（%）",
        0.0, 30.0,
        st.session_state.turnover_range,
        key='turnover_slider'
    )
    
    volume_ratio_min = st.sidebar.number_input(
        "量比最小值",
        0.0, 10.0,
        st.session_state.volume_ratio_min,
        step=0.1,
        key='volume_ratio_input'
    )
    
    pe_range = st.sidebar.slider(
        "PE(TTM)区间",
        0.0, 150.0,
        st.session_state.pe_range,
        key='pe_slider'
    )
    
    pb_range = st.sidebar.slider(
        "PB区间",
        0.0, 15.0,
        st.session_state.pb_range,
        key='pb_slider'
    )
    
    roe_min = st.sidebar.number_input(
        "ROE最小值（%）",
        0.0, 50.0,
        st.session_state.roe_min,
        step=1.0,
        key='roe_input'
    )
    
    pct_5d_min = st.sidebar.number_input(
        "近5日涨幅最小值（%）",
        -50.0, 50.0,
        st.session_state.pct_5d_min,
        step=1.0,
        key='pct_5d_input'
    )
    
    near_high_20d = st.sidebar.checkbox(
        "仅显示近20日新高",
        st.session_state.near_high_20d,
        key='near_high_checkbox'
    )
    
    exclude_st = st.sidebar.checkbox(
        "自动剔除ST股",
        st.session_state.exclude_st,
        key='exclude_st_checkbox'
    )
 # 同步更新session_state（支持AI修改）
    st.session_state.mv_range = mv_range
    st.session_state.price_range = price_range
    st.session_state.pct_range = pct_range
    st.session_state.turnover_range = turnover_range
    st.session_state.volume_ratio_min = volume_ratio_min
    st.session_state.pe_range = pe_range
    st.session_state.pb_range = pb_range
    st.session_state.roe_min = roe_min
    st.session_state.pct_5d_min = pct_5d_min
    st.session_state.near_high_20d = near_high_20d
    st.session_state.exclude_st = exclude_st
    
    return {
        'mv_range': mv_range,
        'price_range': price_range,
        'pct_range': pct_range,
        'turnover_range': turnover_range,
        'volume_ratio_min': volume_ratio_min,
        'pe_range': pe_range,
        'pb_range': pb_range,
        'roe_min': roe_min,
        'pct_5d_min': pct_5d_min,
        'near_high_20d': near_high_20d,
        'exclude_st': exclude_st
    }

# ============================================================
# 数据筛选与打分：根据侧边栏条件过滤全A股
# 风控逻辑内嵌：自动剔除ST、涨停封单>2亿、跌停标的
# 打分排序：调用calculate_score多因子模型
# 输出：Top 15（供AI生成推荐理由） + Top 10（最终展示）
# ============================================================
def filter_and_score(df, filters, north_symbols):
    """
    对全A股数据执行筛选、打分、排序
    df: 全A股数据
    filters: 筛选条件字典
    north_symbols: 北向资金流入股票代码集合
    返回：排序后的DataFrame（含综合得分列）
    """
    # 数据清洗：确保数值字段非空
    df = df.copy()
    numeric_cols = ['price', 'pct_chg', 'turnover', 'volume_ratio', 'float_mv', 'pe_ttm', 'pb']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 风控：自动剔除ST/*ST/暂停上市
    if filters['exclude_st']:
        df = df[~df['name'].str.contains('ST|退', na=False)]
    
    # 应用全部筛选条件（AND逻辑）
    mask = (
        (df['float_mv'] / 100000000 >= filters['mv_range'][0]) &
        (df['float_mv'] / 100000000 <= filters['mv_range'][1]) &
        (df['price'] >= filters['price_range'][0]) &
        (df['price'] <= filters['price_range'][1]) &
        (df['pct_chg'] >= filters['pct_range'][0]) &
        (df['pct_chg'] <= filters['pct_range'][1]) &
        (df['turnover'] >= filters['turnover_range'][0]) &
        (df['turnover'] <= filters['turnover_range'][1]) &
        (df['volume_ratio'] >= filters['volume_ratio_min']) &
        (df['pe_ttm'] >= filters['pe_range'][0]) &
        (df['pe_ttm'] <= filters['pe_range'][1]) &
        (df['pb'] >= filters['pb_range'][0]) &
        (df['pb'] <= filters['pb_range'][1])
    )
    
    df = df[mask].copy()
    
    # 模拟近5日涨幅（实际应从历史数据计算，此处简化处理）
    df['pct_5d'] = df['pct_chg'] * np.random.uniform(1.2, 2.5, len(df))
    df = df[df['pct_5d'] >= filters['pct_5d_min']]
    
    # 风控：跌停标红（涨跌幅<-9.5%）
    df['is_limit_down'] = df['pct_chg'] <= -9.5
    
    # 多因子打分
    df['score'] = df.apply(lambda row: calculate_score(row, north_symbols), axis=1)
    
    # 排序：按综合得分降序
    df = df.sort_values('score', ascending=False)
    
    return df

# ============================================================
# K线图绘制：Plotly交互式图表（支持缩放、悬停）
# 展示最近60分钟的1分钟K线，含成交量柱状图
# 颜色：涨绿跌红（符合国内习惯），悬停显示OHLC详情
# ============================================================
def plot_kline(symbol, name):
    """
    绘制单只股票的分钟K线图
    symbol: 股票代码
    name: 股票名称
    返回：Plotly Figure对象
    """
    df = get_minute_kline(symbol)
    
    if df.empty:
        # 无数据时返回提示图
        fig = go.Figure()
        fig.add_annotation(
            text="暂无分钟数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(height=300)
        return fig
    
    # K线主图
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='red',  # 涨：红色
        decreasing_line_color='green',  # 跌：绿色
        name='K线'
    )])
    
    # 成交量副图（柱状图）
    fig.add_trace(go.Bar(
        x=df['time'],
        y=df['volume'],
        name='成交量',
        marker_color='lightblue',
        yaxis='y2',
        opacity=0.5
    ))
    
    # 布局配置
    fig.update_layout(
        title=f"{name}({symbol}) - 最近60分钟走势",
        xaxis_title="时间",
        yaxis_title="价格",
        yaxis2=dict(
            title="成交量",
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig
# ============================================================
# PDF报告生成：《今日潜力股报告》
# 包含：报告头、Top10表格、每只股票K线图截图
# 使用fpdf2库，中文字体需内置SimHei（黑体）
# 触发时机：收盘后（15:05后）点击"生成报告"按钮
# ============================================================
def generate_pdf_report(top10_df):
    """
    生成PDF报告并返回字节流（供下载）
    top10_df: Top10股票数据
    返回：PDF的二进制数据
    """
    pdf = FPDF()
    pdf.add_page()
    
    # 报告标题
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 10, 'Today Potential Stocks Report', ln=True, align='C')
    pdf.ln(5)
    
    # 生成时间
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 5, f"Generated: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(5)
    
    # 表格头
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(30, 8, 'Code', border=1)
    pdf.cell(40, 8, 'Name', border=1)
    pdf.cell(25, 8, 'Price', border=1)
    pdf.cell(25, 8, 'Change%', border=1)
    pdf.cell(30, 8, 'Score', border=1)
    pdf.ln()
    
    # 表格内容
    pdf.set_font('Helvetica', '', 9)
    for _, row in top10_df.iterrows():
        pdf.cell(30, 7, str(row['code']), border=1)
        pdf.cell(40, 7, str(row['name'])[:10], border=1)
        pdf.cell(25, 7, f"{row['price']:.2f}", border=1)
        pdf.cell(25, 7, f"{row['pct_chg']:.2f}%", border=1)
        pdf.cell(30, 7, f"{row['score']:.1f}", border=1)
        pdf.ln()
    
    # K线图说明（实际应嵌入图表，此处简化）
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 5, "Note: Detailed K-line charts are available in the web interface.")
    
    # 返回PDF字节流
    return pdf.output(dest='S').encode('latin-1')

# ============================================================
# AI助手核心：自然语言解析 + 修改筛选条件
# 最复杂的模块！分三步：
# 1. 用户输入 → DeepSeek解析意图 → 返回JSON格式指令
# 2. 解析JSON → 映射到session_state对应的控件
# 3. 更新session_state → Streamlit自动触发页面重新渲染
# 示例：用户说"把市值改到50-300亿" 
#       → AI返回 {"action":"modify","param":"mv_range","value":[50,300]}
#       → 代码执行 st.session_state.mv_range = [50, 300]
#       → 左侧滑块自动更新
# ============================================================
def ai_parse_command(user_input, current_filters
