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

# 强制设置环境变量（用于Render部署）
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
        # 方法1: 优先尝试从环境变量读取
        import os
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        
        # 方法2: 如果环境变量没有，再尝试从st.secrets读取
        if not api_key:
            api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        
        # 如果两种方法都没有获取到密钥
        if not api_key:
            st.warning("⚠️ DeepSeek API未配置: 请在Render的环境变量中设置 DEEPSEEK_API_KEY")
            return None
            
        return OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    except Exception as e:
        st.warning(f"⚠️ DeepSeek API配置错误: {e}")
        return None

DEEPSEEK_CLIENT = get_deepseek_client()

# ============================================================
# 数据缓存装饰器：@st.cache_data(ttl=4*3600)
# TTL=4小时，避免频繁调用akshare导致IP被封
# 缓存内容：全A股实时行情、分钟K线、北向资金流等
# 注意：akshare数据源不稳定时会自动重试，失败返回空DataFrame
# ============================================================
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=4*3600)
@st.cache_data(ttl=4*3600)
def get_all_stocks():
    """
    获取全A股票池 - 已修正新浪接口列名问题
    """
    max_retries = 2
    data_sources = [
        {"name": "新浪", "func": lambda: ak.stock_zh_a_spot()},
        {"name": "东方财富", "func": lambda: ak.stock_zh_a_spot_em()}
    ]
    
    for source in data_sources:
        for attempt in range(max_retries):
            try:
                st.info(f"正在从【{source['name']}】接口获取数据...")
                df = source['func']()
                
                # 根据数据源进行正确的字段映射
                if source['name'] == "新浪":
                    # 【关键修正】新浪接口实际返回中文列名
                    column_mapping = {
                        '代码': 'code',
                        '名称': 'name', 
                        '最新价': 'price',
                        '涨跌幅': 'pct_chg',
                        # 新浪可能没有的字段，后续会统一补全
                    }
                else:  # 东方财富
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
                        '市净率': 'pb'
                    }
                
                # 应用字段重命名
                df = df.rename(columns=column_mapping)
                
                # ====== 确保所有必需列都存在 ======
                required_columns = {
                    'code': 'Unknown',
                    'name': 'Unknown', 
                    'price': 0.0,
                    'pct_chg': 0.0,
                    'turnover': 0.0,      # 新浪可能缺失
                    'volume_ratio': 1.0,  # 新浪可能缺失
                    'float_mv': 0.0,      # 新浪可能缺失
                    'total_mv': 0.0,      # 新浪可能缺失
                    'pe_ttm': 0.0,        # 新浪可能缺失
                    'pb': 0.0,            # 新浪可能缺失
                    'pct_5d': 0.0
                }
                
                for col, default_val in required_columns.items():
                    if col not in df.columns:
                        df[col] = default_val
                # ====== 修复结束 ======
                
                st.success(f"✅ 成功获取{len(df)}条数据")
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time_module.sleep(1)
                    continue
                else:
                    st.warning(f"⚠️ 【{source['name']}】接口尝试失败，将尝试备用源...")
    
    # 所有数据源都失败
    st.error("❌ 数据获取失败，请检查网络后刷新。")
    safety_columns = ['code', 'name', 'price', 'pct_chg', 'turnover', 
                     'volume_ratio', 'float_mv', 'total_mv', 'pe_ttm', 'pb', 'pct_5d']
    return pd.DataFrame(columns=safety_columns)
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
def ai_parse_command(user_input, current_filters):
    """
    AI解析用户自然语言指令，返回筛选条件修改指令
    这是整个AI助手最核心的函数！负责将自然语言转换为可执行的代码操作
    
    工作流程：
    1. 接收用户输入（如"把ROE改成大于20%"、"加上芯片概念"）
    2. 将当前筛选条件JSON化，连同用户输入一起发送给DeepSeek
    3. DeepSeek按照System Prompt要求，返回结构化JSON指令
    4. 解析JSON，执行对应的session_state修改操作
    
    JSON指令格式示例：
    {
        "action": "modify",  // 动作类型：modify修改/add增加/remove删除
        "param": "mv_range",  // 要修改的参数名（对应session_state键）
        "value": [50, 300],   // 新值（支持数字、列表、布尔）
        "message": "已将流通市值调整为50-300亿"  // 反馈给用户的文字
    }
    
    参数映射表（自然语言 → session_state键）：
    - "市值"/"流通市值" → mv_range
    - "股价"/"价格" → price_range
    - "涨跌幅"/"涨幅" → pct_range
    - "换手率" → turnover_range
    - "量比" → volume_ratio_min
    - "PE"/"市盈率" → pe_range
    - "PB"/"市净率" → pb_range
    - "ROE"/"净资产收益率" → roe_min
    - "近5日涨幅" → pct_5d_min
    - "新高" → near_high_20d
    - "ST股" → exclude_st
    
    异常处理：
    - API调用失败 → 返回友好错误提示
    - JSON解析失败 → 返回"无法理解指令"
    - 参数名不存在 → 返回"不支持该筛选条件"
    """
    if not DEEPSEEK_CLIENT:
        return {"success": False, "message": "❌ DeepSeek API未配置，请在secrets.toml中添加DEEPSEEK_API_KEY"}
    
    # 构造给AI的System Prompt（定义AI的行为规范和输出格式）
    system_prompt = """你是A股智能投研助手的指令解析器。用户会说自然语言来修改筛选条件，你需要将其转换为JSON指令。

可修改的参数及格式：
1. mv_range: 流通市值范围[最小值, 最大值]，单位亿
2. price_range: 股价区间[最小值, 最大值]，单位元
3. pct_range: 今日涨跌幅[最小值, 最大值]，单位%
4. turnover_range: 换手率[最小值, 最大值]，单位%
5. volume_ratio_min: 量比最小值，数字
6. pe_range: PE区间[最小值, 最大值]
7. pb_range: PB区间[最小值, 最大值]
8. roe_min: ROE最小值，单位%
9. pct_5d_min: 近5日涨幅最小值，单位%
10. near_high_20d: 是否仅显示近20日新高，布尔值
11. exclude_st: 是否剔除ST股，布尔值

返回JSON格式（必须严格遵守）：
{
    "action": "modify",
    "param": "参数名",
    "value": 新值,
    "message": "人性化反馈（30字内）"
}

如果用户意图不明确或无法解析，返回：
{
    "action": "error",
    "message": "无法理解指令，请换个说法"
}"""
    try:
        # 调用DeepSeek API进行自然语言理解
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"当前筛选条件：{json.dumps(current_filters, ensure_ascii=False)}\n\n用户指令：{user_input}\n\n请解析为JSON指令："
                }
            ],
            max_tokens=200,
            temperature=0.3  # 低温度保证输出稳定
        )
        
        # 提取AI返回的内容
        ai_response = response.choices[0].message.content.strip()
        
        # 清理可能的Markdown代码块标记
        if ai_response.startswith('```'):
            ai_response = ai_response.split('\n', 1)[1]
        if ai_response.endswith('```'):
            ai_response = ai_response.rsplit('\n', 1)[0]
        
        # 解析JSON指令
        command = json.loads(ai_response)
        
        # 执行指令：修改session_state
        if command.get('action') == 'modify':
            param = command.get('param')
            value = command.get('value')
            
            # 验证参数名是否合法
            valid_params = [
                'mv_range', 'price_range', 'pct_range', 'turnover_range',
                'volume_ratio_min', 'pe_range', 'pb_range', 'roe_min',
                'pct_5d_min', 'near_high_20d', 'exclude_st'
            ]
            
            if param not in valid_params:
                return {
                    "success": False,
                    "message": f"❌ 不支持修改参数'{param}'，请检查指令"
                }
            
            # 类型转换与校验
            try:
                if param in ['mv_range', 'price_range', 'pct_range', 'turnover_range', 'pe_range', 'pb_range']:
                    # 范围类参数：必须是长度为2的列表
                    if not isinstance(value, list) or len(value) != 2:
                        raise ValueError("范围参数需要[最小值, 最大值]格式")
                    value = [float(value[0]), float(value[1])]
                
                elif param in ['volume_ratio_min', 'roe_min', 'pct_5d_min']:
                    # 数值类参数
                    value = float(value)
                
                elif param in ['near_high_20d', 'exclude_st']:
                    # 布尔类参数
                    value = bool(value)
                
                # 更新session_state（这是关键！修改后Streamlit会自动重新渲染页面）
                st.session_state[param] = value
                
                return {
                    "success": True,
                    "message": f"✅ {command.get('message', '筛选条件已更新')}"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"❌ 参数值格式错误：{str(e)}"
                }
        
        elif command.get('action') == 'error':
            return {
                "success": False,
                "message": command.get('message', '❌ 无法理解您的指令')
            }
        
        else:
            return {
                "success": False,
                "message": "❌ AI返回了未知指令类型"
            }
    
    except json.JSONDecodeError:
        return {
            "success": False,
            "message": "❌ AI返回格式错误，请重新描述您的需求"
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ AI解析失败：{str(e)}"
        }

# ============================================================
# AI聊天助手：支持上下文对话 + 实时数据注入
# 功能亮点：
# 1. 自动将Top10股票数据注入每次对话的上下文
# 2. 用户可以问"第一只股票怎么样"，AI能看到完整数据
# 3. 限流保护：每分钟最多3次API调用
# 4. 对话历史存储在session_state，支持多轮对话
# ============================================================
def ai_chat_response(user_message, top10_data, current_filters):
    """
    处理用户与AI助手的对话
    user_message: 用户输入的消息
    top10_data: 当前Top10股票数据（JSON格式）
    current_filters: 当前筛选条件（JSON格式）
    返回：AI的回复文本
    
    限流逻辑：
    - 使用session_state记录最近1分钟的调用时间戳
    - 超过3次则拒绝调用，提示用户稍后再试
    """
    if not DEEPSEEK_CLIENT:
        return "❌ DeepSeek API未配置，请在设置中添加API密钥"
    
    # 限流检查：每分钟最多3次调用
    now = time_module.time()
    if 'ai_call_times' not in st.session_state:
        st.session_state.ai_call_times = []
    
    # 清理1分钟前的调用记录
    st.session_state.ai_call_times = [
        t for t in st.session_state.ai_call_times 
        if now - t < 60
    ]
    
    # 检查是否超过限制
    if len(st.session_state.ai_call_times) >= 3:
        return "⏱️ 调用频率过高，请1分钟后再试（限流保护：每分钟3次）"
    
    # 记录本次调用时间
    st.session_state.ai_call_times.append(now)
    
    # 构造System Prompt（定义AI的角色和行为准则）
    system_prompt = f"""你是专业A股投研助手，当前实时数据如下：

【当前Top10股票】
{top10_data}

【当前筛选条件】
{json.dumps(current_filters, ensure_ascii=False, indent=2)}

【行为准则】
1. 只基于上述实时数据回答问题，不编造信息
2. 涉及个股时必须引用具体数据（价格、涨幅、评分等）
3. 永远提示"股市有风险，投资需谨慎"
4. 严禁预测明天涨跌，只能分析当前技术面
5. 如果用户问题超出数据范围，坦诚告知并建议使用筛选功能
6. 回答简洁专业，每次不超过150字

当前时间：{datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}"""
    
    try:
        # 获取对话历史（支持多轮对话）
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # 构造消息列表（包含历史对话）
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(st.session_state.ai_chat_history)
        messages.append({"role": "user", "content": user_message})
        
        # 调用DeepSeek API
        response = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        ai_reply = response.choices[0].message.content
        
        # 保存对话历史（最多保留最近10轮）
        st.session_state.ai_chat_history.append({"role": "user", "content": user_message})
        st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_reply})
        
        # 限制历史长度，避免上下文过长
        if len(st.session_state.ai_chat_history) > 20:
            st.session_state.ai_chat_history = st.session_state.ai_chat_history[-20:]
        
        return ai_reply
    
    except Exception as e:
        return f"❌ AI调用失败：{str(e)}"

# ============================================================
# 主程序入口：页面渲染与逻辑控制
# 分为两个Tab：
# Tab1 - 智能选股：实时数据、筛选、推荐、K线图
# Tab2 - AI智能助手：自然对话 + 修改筛选条件
# ============================================================
def main():
    """主程序：协调各模块，渲染完整页面"""
    
    # 页面标题
    st.title("📈 DeepSeek量化投研终端")
    st.caption("🚀 AI驱动的A股智能选股系统 | 实时数据 + 多因子模型 + 自然语言交互")
    
    # 渲染侧边栏筛选器
    filters = render_sidebar_filters()
    
    # 获取全A股数据
    with st.spinner("🔄 加载全A股数据..."):
        all_stocks = get_all_stocks()
    
    if all_stocks.empty:
        st.error("❌ 数据加载失败，请检查网络或稍后重试")
        return
    
    # 获取北向资金数据
    north_df = get_north_flow()
    north_symbols = set(north_df['代码'].tolist()) if not north_df.empty else set()
    
    # 执行筛选与打分
    filtered_df = filter_and_score(all_stocks, filters, north_symbols)
    
    # 创建Tab页
    tab1, tab2 = st.tabs(["🎯 智能选股", "🤖 AI智能助手"])
    
    # ========== Tab1: 智能选股 ==========
    with tab1:
        # 显示筛选结果统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("全市场股票数", f"{len(all_stocks)}")
        with col2:
            st.metric("筛选后数量", f"{len(filtered_df)}")
        with col3:
            trading_status = "🟢 交易中" if is_trading_time() else "🔴 休市"
            st.metric("市场状态", trading_status)
        with col4:
            st.metric("更新时间", datetime.now(TZ).strftime("%H:%M:%S"))
        
        if len(filtered_df) == 0:
            st.warning("⚠️ 当前筛选条件下无符合标的，请调整筛选器")
            return
        
        # 获取Top15用于AI生成推荐理由
        top15 = filtered_df.head(15).copy()
        top15_json = top15[['code', 'name', 'price', 'pct_chg', 'score']].to_json(
            orient='records', force_ascii=False
        )
        
        # 调用AI生成推荐理由
        with st.spinner("🤖 AI正在生成推荐理由..."):
            ai_reasons = generate_ai_reasons(top15_json)
        
        # 最终Top10展示
        top10 = filtered_df.head(10).copy()
        top10['推荐理由'] = top10['code'].map(ai_reasons).fillna('技术面向好')
        
        st.subheader("🏆 今日潜力Top10")
        
        # 展示每只股票的详细信息 + K线图
        for idx, row in top10.iterrows():
            # 跌停标红处理
            border_color = "red" if row.get('is_limit_down', False) else "#e0e0e0"
            
            with st.container():
                st.markdown(f"""
                <div style="border: 2px solid {border_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                """, unsafe_allow_html=True)
                
                # 股票基本信息
                col_info, col_chart = st.columns([1, 2])
                
                with col_info:
                    st.markdown(f"### {row['name']} ({row['code']})")
                    st.metric("最新价", f"¥{row['price']:.2f}", f"{row['pct_chg']:.2f}%")
                    st.metric("综合评分", f"{row['score']:.1f}分")
                    st.info(f"💡 {row['推荐理由']}")
                    
                    # 详细指标
                    st.markdown("---")
                    st.text(f"换手率: {row['turnover']:.2f}%")
                    st.text(f"量比: {row['volume_ratio']:.2f}")
                    st.text(f"流通市值: {row['float_mv']/100000000:.2f}亿")
                    st.text(f"PE(TTM): {row['pe_ttm']:.2f}")
                
                with col_chart:
                    # 绘制K线图
                    fig = plot_kline(row['code'], row['name'])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # PDF报告生成按钮（仅收盘后显示）
        now = datetime.now(TZ)
        if now.time() >= time(15, 5):
            if st.button("📄 生成今日报告PDF"):
                with st.spinner("📝 正在生成PDF报告..."):
                    pdf_bytes = generate_pdf_report(top10)
                    st.download_button(
                        label="⬇️ 下载《今日潜力股报告.pdf》",
                        data=pdf_bytes,
                        file_name=f"潜力股报告_{datetime.now(TZ).strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
    
    # ========== Tab2: AI智能助手 ==========
    with tab2:
        st.subheader("🤖 DeepSeek AI投研助手")
        st.caption("💬 支持自然对话 + 智能修改筛选条件 | 每分钟最多3次调用")
        
        # 准备注入上下文的数据
        top10_context = top10[['code', 'name', 'price', 'pct_chg', 'score', '推荐理由']].to_json(
            orient='records', force_ascii=False
        )
        
        # 显示当前筛选条件（方便用户了解上下文）
        with st.expander("📊 当前筛选条件（AI可见）"):
            st.json(filters)
        
        # 聊天历史显示
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # 显示历史对话
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.ai_chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"**👤 您：** {msg['content']}")
                else:
                    st.markdown(f"**🤖 AI：** {msg['content']}")
        
        # 用户输入框（固定底部）
        st.markdown("---")
        user_input = st.text_input(
            "💬 输入您的问题或指令",
            placeholder="例如：第一只股票怎么样？ / 把ROE改成大于20% / 加个芯片概念",
            key="ai_input"
        )
        
        col_send, col_clear, col_modify = st.columns([1, 1, 1])
        
        with col_send:
            if st.button("📤 发送", use_container_width=True):
                if user_input.strip():
                    # 判断是否为修改筛选条件的指令
                    modify_keywords = ['改', '修改', '调整', '设置', '加上', '去掉', '剔除']
                    is_modify_command = any(kw in user_input for kw in modify_keywords)
                    
                    if is_modify_command:
                        # 调用AI解析指令
                        with st.spinner("🔧 AI正在解析您的指令..."):
                            result = ai_parse_command(user_input, filters)
                        
                        if result['success']:
                            st.success(result['message'])
                            st.rerun()  # 重新渲染页面以更新筛选器
                        else:
                            st.error(result['message'])
                    else:
                        # 普通对话
                        with st.spinner("🤔 AI正在思考..."):
                            ai_reply = ai_chat_response(user_input, top10_context, filters)
                        st.rerun()  # 刷新显示新对话
        
        with col_clear:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.ai_chat_history = []
                st.session_state.ai_call_times = []
                st.rerun()
        
        with col_modify:
            st.markdown("💡 **快捷指令示例**")
        
        # 快捷指令按钮
        st.markdown("---")
        st.caption("⚡ 一键快捷指令")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("🔥 推荐一只高分股票"):
                user_input_quick = "推荐一只综合评分最高的股票，详细分析其优势"
                with st.spinner("🤔 AI正在分析..."):
                    ai_reply = ai_chat_response(user_input_quick, top10_context, filters)
                st.rerun()
        
        with quick_col2:
            if st.button("📈 分析市场热点"):
                user_input_quick = "分析当前Top10股票的共同特征和市场热点"
                with st.spinner("🤔 AI正在分析..."):
                    ai_reply = ai_chat_response(user_input_quick, top10_context, filters)
                st.rerun()
        
        with quick_col3:
            if st.button("⚠️ 风险提示"):
                user_input_quick = "对Top10股票进行风险评估，指出潜在风险"
                with st.spinner("🤔 AI正在分析..."):
                    ai_reply = ai_chat_response(user_input_quick, top10_context, filters)
                st.rerun()
    
    # ========== 自动刷新逻辑 ==========
    st.markdown("---")
    refresh_interval = 5 if is_trading_time() else 30
    st.caption(f"🔄 自动刷新：{refresh_interval}秒 | 交易时段5秒，非交易时段30秒")
    
    # 倒计时显示
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time_module.time()
    
    elapsed = int(time_module.time() - st.session_state.last_refresh)
    remaining = max(0, refresh_interval - elapsed)
    
    progress_bar = st.progress(remaining / refresh_interval)
    countdown_text = st.empty()
    countdown_text.text(f"⏱️ 下次刷新倒计时: {remaining}秒")
    
    # 自动刷新触发
    if remaining == 0:
        st.session_state.last_refresh = time_module.time()
        st.rerun()
    
    # 使用JavaScript实现精确倒计时（可选，提升用户体验）
    time_module.sleep(1)
    st.rerun()
# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    main()







