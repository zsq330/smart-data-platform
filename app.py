# -*- coding: utf-8 -*-

"""
================================================================================
     智能数据清洗与可视化分析平台  (Smart Data Cleaning & Analytics Platform)
================================================================================

技术栈 : Streamlit + Pandas + Plotly + 通义千问 Qwen（openai 兼容接口） + wordcloud + matplotlib

【界面结构一览】
    ┌ 公共操作区（Tabs 标签页上方）：上传 CSV / 载入示例数据 → 一键清洗 → 清洗前后对比
    ├ Tab 1【通用清洗与探索】        异常值箱线图 / 自选交互图表 / 相关性热力图
    ├ Tab 2【商业与市场分析】        交叉分析表 / 文本词云图 / 对比图表
    └ Tab 3【科研与深度学习预处理】  数据标准化 / 特征分布直方图 / 建模可用性检查
    每个标签页内部：左侧操作区 + 右侧 AI 智能分析报告（st.columns 分栏展示）

【第一步：安装依赖】在 PyCharm 底部的 Terminal 里执行（只需要执行一次）:

    pip install streamlit pandas numpy plotly openai wordcloud matplotlib openpyxl

可选（强烈建议，中文词云分词效果会好很多）:

    pip install jieba

【第二步：运行】在 PyCharm 的 Terminal 里执行:

    streamlit run app.py

    （或者在 PyCharm 里右键本文件 -> Run 'app'，若没反应就用上面的命令）

    运行成功后浏览器会自动打开 http://localhost:8501

【安全说明】

    本代码里没有写死任何 API Key！

    Key 请在页面左侧侧边栏手动输入（type="password"，输入的内容会显示成圆点）。

    如果不想每次都输入，可以设置系统环境变量 DASHSCOPE_API_KEY，程序只会把它当作
    输入框的“默认值”，依然不会出现在代码里。

【代码地图】（新手调试请按下面的分区编号找代码）

    # ===== 分区 0 =====   依赖导入
    # ===== 分区 1 =====   页面全局配置 + CSS 美化
    # ===== 分区 2 =====   常量：智能角色设定、提示词模板
    # ===== 分区 3 =====   通用工具函数（读文件/示例数据/列类型识别/导出）
    # ===== 分区 4 =====   公共数据清洗模块（核心算法）
    # ===== 分区 5 =====   深度数据洞察模块（通义千问调用 + 三段式展示）
    # ===== 分区 6 =====   词云图模块
    # ===== 分区 7 =====   侧边栏（API Key 输入区）
    # ===== 分区 8 =====   页面顶部标题区
    # ===== 分区 9 =====   公共操作区：数据上传 + 清洗前后对比（三个标签页共享）
    # ===== 分区 10 =====  标签页1【通用清洗与探索】
    # ===== 分区 11 =====  标签页2【商业与市场分析】
    # ===== 分区 12 =====  标签页3【科研与深度学习预处理】
    # ===== 分区 13 =====  页脚
================================================================================
"""


# ==============================================================================
# ===== 分区 0 =====  依赖导入
# ==============================================================================

import io
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 关键：使用无界面后端，避免 PyCharm 里画图弹窗卡死

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from wordcloud import STOPWORDS, WordCloud

# jieba 是可选依赖：装了它中文分词更准，没装也能跑（会走正则兜底逻辑）
try:
    import jieba

    HAS_JIEBA = True
except Exception:  # noqa: BLE001
    HAS_JIEBA = False


# ==============================================================================
# ===== 分区 1 =====  页面全局配置 + CSS 美化
# ==============================================================================

# 注意：st.set_page_config 必须是第一个 Streamlit 命令，且只能出现一次
st.set_page_config(
    page_title="智能数据清洗与可视化分析平台",
    page_icon="📊",
    layout="wide",                     # 宽屏模式
    initial_sidebar_state="expanded",
)


# 一点自定义 CSS，让页面看起来更专业（纯装饰，不影响功能）
CUSTOM_CSS = """
<style>
    /* 关键修复：Streamlit 顶部有一条“固定定位”的工具栏（右侧菜单/运行状态），
       它悬浮在页面最上方。如果 padding-top 太小，大标题的上半截就会被它遮住。
       这里恢复为 6rem（Streamlit 默认值），确保标题完整显示在工具栏下方。 */
    .block-container { padding-top: 6rem; padding-bottom: 2.5rem; max-width: 100%; }

    .main-title {
        font-size: 2.35rem; font-weight: 800; letter-spacing: 1px;
        line-height: 1.4; padding: 0.3rem 0 0.2rem 0;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { font-size: 1.02rem; color: #64748b; margin-bottom: 0.6rem; }
    .pill {
        display: inline-block; padding: 3px 12px; margin-right: 8px; margin-bottom: 6px;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe;
    }
    .card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 10px;
    }
    .ai-box {
        background: #f5f3ff; border-left: 5px solid #7c3aed; border-radius: 10px;
        padding: 14px 18px; margin-top: 6px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 10px 14px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 20px; font-size: 1.0rem; font-weight: 700;
        border-radius: 10px 10px 0 0; background: #f1f5f9;
    }
    .stTabs [aria-selected="true"] { background: #e0e7ff !important; color: #3730a3 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Plotly 统一风格
PLOTLY_TEMPLATE = "plotly_white"

# 一份颜色很丰富的离散色板（箱线图、柱状图都会用到）
COLOR_SEQ = px.colors.qualitative.Bold

# 一份连续色板（热力图、词云会用到）
CONTINUOUS_SCALES = ["Viridis", "Turbo", "Plasma", "Cividis", "RdBu_r", "YlGnBu", "Spectral", "Magma"]

# matplotlib 中文字体设置（词云图/静态图不乱码的关键）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# ===== 分区 2 =====  常量：智能角色设定、提示词模板
# ==============================================================================

# 三种智能角色（对应右侧智能报告区的下拉框）
ROLE_PROMPTS = {
    "通用数据分析师": (
        "你是一位严谨又通俗的数据分析师，擅长数据质量评估、分布特征、相关性、异常值检测，"
        "并善于把统计结论翻译成业务人员能听懂的话。请避免堆砌专业术语，必要时用一句话解释术语。"
    ),
    "商业战略顾问": (
        "你是一位资深商业战略顾问（曾服务零售、快消、互联网行业），擅长从销售、市场、客户、"
        "渠道与运营数据中发现增长机会、风险点和可落地的策略。请用商业语言表达，强调优先级、"
        "投入产出比和可执行的下一步动作。"
    ),
    "学术科研助手": (
        "你是一位学术科研助手，熟悉统计推断、实验设计、数据预处理规范与论文写作规范。"
        "请客观、严谨、保守地表述结论，主动说明局限性、混杂因素与样本量问题，不要过度解读相关性。"
    ),
}


# 统一的三段式输出模板（要求模型必须按【核心发现】【原因分析】【行动建议】来写）
AI_SYSTEM_TEMPLATE = """{role_prompt}

【输出格式要求】必须严格遵守，只输出下面三段，不要输出任何开场白或结尾客套话：

## 一、核心发现
- 用 3~6 条要点列出数据中最重要的事实，尽量引用具体数字（如均值、占比、相关系数、异常比例）

## 二、原因分析
- 解释上述现象背后的可能原因、机制或数据层面的解释，并明确哪些是推测、哪些是数据直接支持的

## 三、行动建议
- 给出 3~5 条具体、可执行、有优先级的建议（按 P0/P1/P2 标注优先级），必要时给出下一步该做的分析或该补的数据

【硬性约束】
1. 全部用中文回答。
2. 结论必须基于我提供的数据摘要，禁止凭空编造字段名、数值或行业事实。
3. 如果数据不足以支撑判断，必须明确指出“需要补充哪些信息”，不要硬编结论。
4. 使用 Markdown 排版，要点清晰，不要写成一大段文字。"""


# 大模型接口默认配置（用户可在侧边栏自由替换为中转 API / 其他任意平台）
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


# 给三个标签页准备默认的分析诉求，方便新手一键体验
DEFAULT_FOCUS = {
    "tab1": "请评估这份数据的整体质量（缺失、异常、分布、相关性），并告诉我清洗结果是否可靠。",
    "tab2": "请从商业角度指出增长机会与风险，并给出可落地的运营或营销建议。",
    "tab3": "请从科研角度评估数据是否适合直接用于建模（如深度学习），并给出预处理与建模建议。",
}


# ==============================================================================
# ===== 分区 3 =====  通用工具函数
# ==============================================================================

def get_cjk_font_path():
    """找一个可用的中文字体文件，词云图必须用它才不会显示成方块。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",      # 黑体
        r"C:\Windows\Fonts\simsun.ttc",      # 宋体
        "/System/Library/Fonts/PingFang.ttc",             # macOS
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",   # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def dedupe_columns(columns):
    """把重复的列名改成 name / name.1 / name.2，避免后续操作报错。"""
    seen = {}
    new_cols = []
    for col in columns:
        name = str(col).strip() or "未命名列"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    return new_cols


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes, file_name: str = "data.csv") -> pd.DataFrame:
    """
    读取 CSV 的健壮版本：
    1) 依次尝试 utf-8-sig / utf-8 / gbk / gb18030 等编码（中文 CSV 常见乱码问题）
    2) 如果只解析出 1 列，说明分隔符可能不是逗号，自动嗅探分隔符重试
    """
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "big5", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            break
        except UnicodeDecodeError as err:
            last_error = err
            continue
        except Exception as err:  # noqa: BLE001  例如分隔符异常
            last_error = err
            continue
    else:
        raise ValueError(f"无法读取该 CSV 文件，请检查文件格式。原始错误：{last_error}")

    # 分隔符嗅探：只解析出一列且列名里含分隔符时，用 python 引擎自动嗅探
    if df.shape[1] == 1 and (";" in str(df.columns[0]) or "\t" in str(df.columns[0]) or "|" in str(df.columns[0])):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            pass

    df.columns = dedupe_columns(df.columns)
    return df


@st.cache_data(show_spinner=False)
def build_demo_data(kind: str) -> pd.DataFrame:
    """内置示例数据：没有 CSV 文件也能立刻体验全部功能（故意掺入缺失值和重复行）。"""
    rng = np.random.default_rng(20240501)

    if kind == "商业销售数据":
        n = 320
        reviews = [
            "物流很快包装完好非常满意", "产品质量一般希望改进", "性价比很高会回购",
            "客服态度好回复及时", "发货太慢体验不好", "外观设计漂亮值得推荐",
            "价格偏高质量不错", "用了两天就出问题", "整体满意继续关注", "活动力度大很划算",
        ]
        df = pd.DataFrame({
            "订单编号": [f"SO{10000 + i}" for i in range(n)],
            "下单日期": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "地区": rng.choice(["华东", "华北", "华南", "西南", "东北", "西北"], n),
            "销售渠道": rng.choice(["线上自营", "直播带货", "线下门店", "代理商"], n),
            "商品品类": rng.choice(["家电", "数码", "服饰", "食品", "美妆"], n),
            "销售金额": np.round(rng.normal(5200, 1900, n).clip(180, None), 2),
            "销售数量": rng.integers(1, 60, n),
            "利润率": np.round(rng.normal(0.18, 0.07, n), 4),
            "客户评分": np.round(rng.normal(4.2, 0.55, n).clip(1, 5), 1),
            "客户评价": rng.choice(reviews, n),
        })

        # 故意制造缺失值 + 重复行 + 首尾空格，用来演示清洗效果
        df.loc[rng.choice(n, 18, replace=False), "利润率"] = np.nan
        df.loc[rng.choice(n, 10, replace=False), "客户评分"] = np.nan
        df.loc[rng.choice(n, 8, replace=False), "销售渠道"] = "  线上自营  "
        df = pd.concat([df, df.head(6)], ignore_index=True)
        return df

    if kind == "医疗健康数据":
        n = 280
        df = pd.DataFrame({
            "患者编号": [f"P{2000 + i}" for i in range(n)],
            "性别": rng.choice(["男", "女"], n),
            "年龄段": rng.choice(["0-17", "18-34", "35-49", "50-64", "65+"], n),
            "科室": rng.choice(["心血管内科", "内分泌科", "呼吸内科", "消化内科", "神经内科"], n),
            "收缩压": np.round(rng.normal(128, 16, n), 1),
            "舒张压": np.round(rng.normal(80, 11, n), 1),
            "空腹血糖": np.round(rng.normal(5.6, 1.1, n), 2),
            "总胆固醇": np.round(rng.normal(4.7, 0.9, n), 2),
            "BMI": np.round(rng.normal(23.8, 3.4, n), 2),
            "住院天数": rng.integers(1, 22, n),
            "诊断描述": rng.choice(
                ["高血压合并血脂异常", "2型糖尿病血糖控制不佳", "慢性支气管炎急性发作",
                 "胃食管反流伴慢性胃炎", "腔隙性脑梗死恢复期", "甲状腺功能减退"], n),
        })
        df.loc[rng.choice(n, 15, replace=False), "空腹血糖"] = np.nan
        df.loc[rng.choice(n, 9, replace=False), "BMI"] = np.nan
        df = pd.concat([df, df.head(5)], ignore_index=True)
        return df

    # 科研实验数据
    n = 300
    df = pd.DataFrame({
        "样本编号": [f"S{3000 + i}" for i in range(n)],
        "实验组别": rng.choice(["对照组", "低剂量组", "中剂量组", "高剂量组"], n),
        "温度_摄氏度": np.round(rng.normal(37, 2.6, n), 2),
        "反应时间_分钟": np.round(rng.normal(45, 9, n), 1),
        "底物浓度_mM": np.round(rng.normal(12, 3.2, n), 3),
        "pH值": np.round(rng.normal(7.1, 0.35, n), 2),
        "产率_百分比": np.round(rng.normal(72, 11, n).clip(5, 100), 2),
        "荧光强度": np.round(rng.normal(1500, 260, n), 1),
        "实验备注": rng.choice(
            ["反应体系澄清无沉淀", "出现少量絮状物需过滤", "重复实验三次结果稳定",
             "存在批次间差异", "仪器基线漂移已校正"], n),
    })

    # 制造几个明显异常值，方便演示异常值检测功能
    df.loc[rng.choice(n, 6, replace=False), "产率_百分比"] = rng.uniform(150, 220, 6)
    df.loc[rng.choice(n, 4, replace=False), "荧光强度"] = rng.uniform(4000, 6000, 4)
    df.loc[rng.choice(n, 12, replace=False), "pH值"] = np.nan
    df = pd.concat([df, df.head(4)], ignore_index=True)
    return df


def numeric_columns(df: pd.DataFrame):
    """找出所有数值型列。"""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def ordered_columns(df: pd.DataFrame):
    """下拉框选项排序：数值列自动排在前面，文本列排在后面。"""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rest = [c for c in df.columns if c not in num]
    return num + rest


def col_labeler(df: pd.DataFrame):
    """
    生成一个“列名格式化器”，把下拉框里的列名显示成「销量 (数值)」「城市 (文本)」的样子。
    通过 st.selectbox / st.multiselect 的 format_func 参数接入——只改显示文字，
    组件返回值仍然是真实列名，所以不影响后续任何数据处理逻辑。
    以“（”开头的占位选项（如“（自动：计数）”）原样显示，不打标签。
    """
    num_set = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}

    def _label(c):
        if not isinstance(c, str) or c.startswith("（"):
            return c
        return f"{c} (数值)" if c in num_set else f"{c} (文本)"

    return _label


def is_text_like(s: pd.Series) -> bool:
    """
    判断一列是否属于“文本型”。
    注意：pandas 2.x 里文本列是 object，pandas 3.x 里是新的 str dtype，
    所以这里同时兼容两种判断，避免升级 pandas 后功能失效。
    """
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return False
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return False
    if isinstance(s.dtype, pd.CategoricalDtype):
        return False
    try:
        return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s))
    except Exception:  # noqa: BLE001
        return False


def is_categorical_like(s: pd.Series) -> bool:
    """判断一列是否属于“分类型”（文本 / 布尔 / category 都算）。"""
    return (
        is_text_like(s)
        or pd.api.types.is_bool_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


def categorical_columns(df: pd.DataFrame, max_unique_ratio: float = 0.6):
    """找出分类列（文本/布尔/类别型，且取值不要太散）。"""
    cols = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if is_categorical_like(s):
            try:
                if s.nunique(dropna=True) <= max(30, int(n * max_unique_ratio)):
                    cols.append(c)
            except TypeError:  # 列里混了不可哈希对象，跳过
                continue
    return cols


def looks_like_date(s: pd.Series, threshold: float = 0.8) -> bool:
    """粗略判断一列是不是日期（用于把日期列从“长文本/词云”候选里剔除）。"""
    sample = s.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= threshold


def text_columns(df: pd.DataFrame, min_avg_len: float = 8.0):
    """找出“长文本”列（平均长度较长的字符串列），用于词云。"""
    cols = []
    for c in df.columns:
        if is_text_like(df[c]):
            s = df[c].dropna().astype(str)
            if len(s) == 0 or s.str.len().mean() < min_avg_len:
                continue
            # 日期列（如 2024-01-01）虽然长，但不适合做词云，直接排除
            if looks_like_date(df[c]):
                continue
            cols.append(c)
    return cols


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """把 DataFrame 转成 CSV 字节流（utf-8-sig，Excel 打开不乱码）。"""
    return df.to_csv(index=False).encode("utf-8-sig")


def human_int(x) -> str:
    """把数字格式化成带千分位的字符串，用于展示。"""
    try:
        return f"{int(x):,}"
    except Exception:  # noqa: BLE001
        return str(x)


def safe_selectbox(label, options, key, index=0, **kwargs):
    """一个不会崩掉的选择框封装：没有可选项时返回 None 并给出提示。"""
    opts = list(options)
    if not opts:
        st.info(f"⚠️ 当前数据中没有可用于「{label}」的字段，请先检查数据类型或更换数据。")
        return None
    idx = index if 0 <= index < len(opts) else 0
    return st.selectbox(label, opts, index=idx, key=key, **kwargs)


def require_data(df, tab_name: str) -> bool:
    """检查数据是否就绪，未就绪时给出友好提示。"""
    if df is None or len(df) == 0:
        st.warning(f"📭 [{tab_name}] 还没有可用数据。请先在上方「数据上传与公共清洗」区域上传 CSV 或载入示例数据。")
        return False
    return True


def add_margins(table: pd.DataFrame, name: str = "合计") -> pd.DataFrame:
    """给交叉表补上“合计”行与“合计”列。"""
    out = table.copy()
    out[name] = out.sum(axis=1)
    total_row = out.sum(axis=0).to_frame().T
    total_row.index = [name]
    out = pd.concat([out, total_row], axis=0)
    return out


# ==============================================================================
# ===== 分区 4 =====  公共数据清洗模块（核心算法，任何标签页都共享同一份清洗结果）
# ==============================================================================

def clean_dataframe(
    df: pd.DataFrame,
    missing_strategy: str = "智能填充（数值→中位数，文本→众数）",
    drop_duplicates: bool = True,
    strip_whitespace: bool = True,
    auto_convert_numeric: bool = True,
    outlier_action: str = "仅标记，不删除",
):
    """
    通用清洗流程，返回 (清洗后的DataFrame, 指标字典, 日志列表)。

    步骤顺序很关键，新手可以对照注释理解每一步在干什么：
      1. 备份原始数据
      2. 清理列名（去空格、去重）
      3. 删除“整列全空”的列
      4. 清理文本单元格的首尾空格、不可见字符
      5. 智能类型转换（把看着像数字的文本列转成数值列）
      6. 缺失值处理（4 种策略可选）
      7. 重复行处理
      8. 异常值处理（可选：按 IQR 规则删除）
    """
    logs = []
    metrics = {}
    work = df.copy()

    metrics["原始行数"] = int(work.shape[0])
    metrics["原始列数"] = int(work.shape[1])
    logs.append(f"📥 读入原始数据：{work.shape[0]} 行 × {work.shape[1]} 列")

    # ---------- 步骤 2：清理列名 ----------
    work.columns = dedupe_columns(work.columns)
    logs.append("🏷️ 已统一列名格式（去首尾空格 + 重名列自动改名）")

    # ---------- 步骤 3：删除整列全空 ----------
    all_na_cols = [c for c in work.columns if work[c].isna().all()]
    if all_na_cols:
        work = work.drop(columns=all_na_cols)
        logs.append(f"🗑️ 删除 {len(all_na_cols)} 个“整列全空”的列：{', '.join(map(str, all_na_cols))}")
    metrics["删除空列数"] = len(all_na_cols)

    # ---------- 步骤 4：清理文本空格 ----------
    if strip_whitespace:
        obj_cols = [c for c in work.columns if is_text_like(work[c])]
        for c in obj_cols:
            work[c] = work[c].map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
        if obj_cols:
            logs.append(f"🧹 已清理 {len(obj_cols)} 个文本列中的多余空格与换行符")

    # ---------- 步骤 5：智能类型转换 ----------
    converted = []
    if auto_convert_numeric:
        for c in work.columns:
            if not is_text_like(work[c]):
                continue
            s = work[c].dropna().astype(str)
            if len(s) == 0:
                continue
            # 去掉千分位逗号和百分号后再尝试转换
            cleaned = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() >= 0.8:  # 80% 以上能转成数字才认为它是数值列
                full_cleaned = (
                    work[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                work[c] = pd.to_numeric(full_cleaned, errors="coerce")
                converted.append(c)
        if converted:
            logs.append(f"🔢 智能识别并转换 {len(converted)} 个“数字文本”列为数值类型：{', '.join(converted)}")
    metrics["类型转换列数"] = len(converted)

    # ---------- 步骤 6：缺失值处理 ----------
    missing_before = int(work.isna().sum().sum())
    filled_cells = 0

    if missing_strategy == "删除含缺失值的行":
        rows_before = len(work)
        work = work.dropna(how="any")
        logs.append(f"🗑️ 删除含缺失值的行：{rows_before - len(work)} 行")
    elif missing_strategy == "仅删除全空行（保留部分缺失）":
        rows_before = len(work)
        work = work.dropna(how="all")
        logs.append(f"🗑️ 删除完全空白的行：{rows_before - len(work)} 行")
    elif missing_strategy == "智能填充（数值→中位数，文本→众数）":
        for c in work.columns:
            na_count = int(work[c].isna().sum())
            if na_count == 0:
                continue
            if pd.api.types.is_numeric_dtype(work[c]):
                fill_value = work[c].median()
                if pd.isna(fill_value):
                    fill_value = 0
                work[c] = work[c].fillna(fill_value)
                logs.append(f"🩹 数值列「{c}」用中位数 {round(float(fill_value), 4)} 填充 {na_count} 个缺失值")
            else:
                mode_series = work[c].mode(dropna=True)
                fill_value = mode_series.iloc[0] if len(mode_series) > 0 else "未知"
                work[c] = work[c].fillna(fill_value)
                logs.append(f"🩹 文本列「{c}」用众数「{fill_value}」填充 {na_count} 个缺失值")
            filled_cells += na_count
    else:  # 不做处理
        logs.append("ℹ️ 缺失值保持原样（按你的选择未做处理）")

    metrics["填充单元格数"] = filled_cells
    missing_after = int(work.isna().sum().sum())

    # ---------- 步骤 7：重复行处理 ----------
    dup_count = 0
    if drop_duplicates:
        dup_count = int(work.duplicated().sum())
        if dup_count > 0:
            work = work.drop_duplicates()
        logs.append(f"🧾 删除完全重复的行：{dup_count} 行")
    metrics["删除重复行数"] = dup_count

    # ---------- 步骤 8：异常值处理（IQR 法） ----------
    outlier_removed = 0
    if outlier_action == "按 IQR 规则删除异常行":
        num_cols = numeric_columns(work)
        if num_cols:
            mask = pd.Series(False, index=work.index)
            for c in num_cols:
                q1, q3 = work[c].quantile(0.25), work[c].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask |= (work[c] < lower) | (work[c] > upper)
            outlier_removed = int(mask.sum())
            work = work.loc[~mask]
            logs.append(f"📉 按 IQR 规则删除异常行：{outlier_removed} 行")
    metrics["删除异常行数"] = outlier_removed

    work = work.reset_index(drop=True)

    metrics["清洗后行数"] = int(work.shape[0])
    metrics["清洗后列数"] = int(work.shape[1])
    metrics["删除行数合计"] = metrics["原始行数"] - metrics["清洗后行数"]
    metrics["缺失值（清洗前）"] = missing_before
    metrics["缺失值（清洗后）"] = missing_after
    metrics["完整率"] = round(1 - missing_after / max(work.size, 1), 4)

    logs.append(f"✅ 清洗完成：{work.shape[0]} 行 × {work.shape[1]} 列")

    return work, metrics, logs


# ==============================================================================
# ===== 分区 5 =====  深度数据洞察模块（通义千问调用 + 三段式展示）
# ==============================================================================

def build_data_context(df: pd.DataFrame, extra_lines: str = "", sample_rows: int = 5) -> str:
    """
    把当前数据整理成一段“给模型看的文本摘要”。
    注意控制长度：只发送统计摘要 + 少量样本，不发送整张表，既省钱又快。
    """
    lines = []
    lines.append("========== 数据摘要 ==========")
    lines.append(f"【数据规模】共 {df.shape[0]} 行 × {df.shape[1]} 列")
    lines.append(f"【字段列表】{', '.join(map(str, df.columns))}")

    # 字段信息表
    try:
        info = pd.DataFrame({
            "字段": list(df.columns),
            "类型": [str(t) for t in df.dtypes],
            "非空数": df.notna().sum().values,
            "缺失数": df.isna().sum().values,
            "缺失率%": (df.isna().mean().values * 100).round(2),
            "唯一值数": df.nunique().values,
        })
        lines.append("\n【字段概览】")
        lines.append(info.to_string(index=False))
    except Exception as err:  # noqa: BLE001
        lines.append(f"（字段概览生成失败：{err}）")

    # 数值列统计
    num_cols = numeric_columns(df)
    if num_cols:
        lines.append("\n【数值字段统计】")
        try:
            desc = df[num_cols].describe().T
            desc["偏度"] = df[num_cols].skew()
            desc["峰度"] = df[num_cols].kurtosis()
            keep = [c for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "偏度", "峰度"] if c in desc.columns]
            lines.append(desc[keep].round(4).to_string())
        except Exception as err:  # noqa: BLE001
            lines.append(f"（数值统计生成失败：{err}）")

        # 异常值情况（IQR 规则）
        try:
            out_rows = []
            for c in num_cols:
                s = df[c].dropna()
                if len(s) < 5:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                cnt = int(((s < low) | (s > high)).sum())
                if cnt > 0:
                    out_rows.append(f"{c}: {cnt} 个异常值（占比 {cnt / len(s) * 100:.1f}%，正常区间 {low:.2f} ~ {high:.2f}）")
            if out_rows:
                lines.append("\n【异常值检测（IQR 规则）】")
                lines.extend(out_rows)
        except Exception:  # noqa: BLE001
            pass

        # 相关性最强的变量对
        if len(num_cols) >= 2:
            try:
                corr = df[num_cols].corr().abs()
                pairs = (
                    corr.where(~np.eye(len(corr), dtype=bool))
                    .stack()
                    .sort_values(ascending=False)
                )
                pair_lines, seen = [], set()
                for (a, b), v in pairs.items():
                    key = tuple(sorted([str(a), str(b)]))
                    if key in seen:
                        continue
                    seen.add(key)
                    pair_lines.append(f"{a} ↔ {b}: 相关系数绝对值 {v:.3f}")
                    if len(pair_lines) >= 6:
                        break
                if pair_lines:
                    lines.append("\n【相关性最强的变量组合】")
                    lines.extend(pair_lines)
            except Exception:  # noqa: BLE001
                pass

    # 分类列取值分布
    cat_cols = categorical_columns(df)
    if cat_cols:
        lines.append("\n【分类字段取值分布（前 8 个字段）】")
        for c in cat_cols[:8]:
            try:
                vc = df[c].astype(str).value_counts().head(8)
                total = max(int(df[c].notna().sum()), 1)
                parts = [f"{k}({v}条,{v / total * 100:.1f}%)" for k, v in vc.items()]
                lines.append(f"- {c}: " + "；".join(parts))
            except Exception:  # noqa: BLE001
                continue

    # 额外的上下文（每个标签页会塞入自己的分析结果，比如交叉表、标准化说明）
    if extra_lines:
        lines.append("\n【本次分析产生的额外信息】")
        lines.append(extra_lines)

    # 少量样本行
    try:
        lines.append(f"\n【数据样例（前 {sample_rows} 行）】")
        lines.append(df.head(sample_rows).to_string(index=False))
    except Exception:  # noqa: BLE001
        pass

    context = "\n".join(str(x) for x in lines)

    # 保险丝：限制总长度，避免超出模型上下文
    if len(context) > 14000:
        context = context[:14000] + "\n...（摘要过长已截断）"

    return context


def call_qwen(api_key: str, base_url: str, system_prompt: str, role: str,
              user_focus: str, data_context: str, model: str = "qwen-plus",
              temperature: float = 0.4, max_tokens: int = 2000):
    """
    调用大模型（OpenAI 兼容接口）。
    base_url 和 model 都来自用户在侧边栏的输入，支持中转 API 与其他任意平台。
    返回 (报告文本, 错误信息)，成功时错误信息为 None。
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [
            # 注意：这里用 replace 而不是 str.format，
            # 防止用户自定义提示词里出现花括号 {xxx} 导致程序崩溃
            {"role": "system", "content": system_prompt.replace("{role_prompt}", ROLE_PROMPTS[role])},
            {
                "role": "user",
                "content": (
                    f"【我选择的智能角色】{role}\n"
                    f"【我的分析诉求】{user_focus or '请基于数据摘要给出全面洞察。'}\n\n"
                    f"{data_context}"
                ),
            },
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        content = resp.choices[0].message.content
        if not content or not str(content).strip():
            return None, "模型返回了空内容，请稍后重试或降低 temperature 再试。"
        return str(content), None

    except Exception as err:  # noqa: BLE001
        raw = str(err)
        low = raw.lower()
        if "401" in raw or "authentication" in low or "invalid api key" in low or "incorrect api key" in low:
            friendly = "❌ API Key 无效或已失效（HTTP 401）。请在左侧侧边栏重新填写你的 API Key。"
        elif "402" in raw or "insufficient" in low or "balance" in low or "quota" in low:
            friendly = "💳 账户余额不足或额度用尽（HTTP 402）。请到阿里云百炼官网充值后再试。"
        elif "429" in raw or "rate limit" in low:
            friendly = "⏳ 请求过于频繁或触发限流（HTTP 429）。请等几十秒后重试，或降低调用频率。"
        elif "connect" in low or "timeout" in low or "timed out" in low:
            friendly = "🌐 网络连接失败或超时。请检查网络/代理设置，确认能访问 https://dashscope.aliyuncs.com。"
        elif "model" in low and "not" in low:
            friendly = f"🧩 模型名称不可用：{raw}。请检查左侧侧边栏的「模型名称」是否填写正确。"
        else:
            friendly = f"⚠️ 调用智能分析接口失败：{raw}"
        return None, friendly


def render_ai_panel(df: pd.DataFrame, tab_key: str, title: str = "🤖 深度数据洞察报告", extra_context: str = ""):
    """
    【所有标签页共享】右侧智能报告区。
    tab_key 用于生成唯一组件 key（非常重要！三个标签页同时渲染，key 不能重复）
    """
    st.markdown(f"#### {title}")
    st.caption("选择智能角色 → 补充诉求 → 生成含【核心发现】【原因分析】【行动建议】的三段式报告")

    role = st.selectbox(
        "选择智能角色",
        list(ROLE_PROMPTS.keys()),
        key=f"{tab_key}_role",
        help="不同角色会用不同的视角和语言风格解读同一份数据。",
    )

    focus = st.text_area(
        "补充你的分析诉求（可留空）",
        value=DEFAULT_FOCUS.get(tab_key, ""),
        height=90,
        key=f"{tab_key}_focus",
        placeholder="例如：重点关注华东地区的销售下滑原因 / 帮我判断这份数据能否直接用于训练神经网络",
    )

    api_key = (st.session_state.get("api_key_input", "") or "").strip()
    base_url = (st.session_state.get("base_url_input", "") or "").strip() or DEFAULT_BASE_URL
    model = (st.session_state.get("model_input", "") or "").strip() or DEFAULT_MODEL
    temperature = st.session_state.get("temperature", 0.4)
    max_tokens = st.session_state.get("max_tokens", 2000)

    # 需求3：用户在侧边栏自定义了提示词，就以用户的为准；留空则用内置三段式模板
    custom_prompt = (st.session_state.get("custom_prompt", "") or "").strip()
    system_prompt = custom_prompt if custom_prompt else AI_SYSTEM_TEMPLATE

    col_btn, col_info = st.columns([1, 1.2])
    with col_btn:
        clicked = st.button("🚀 生成智能洞察报告", key=f"{tab_key}_run", type="primary", use_container_width=True)
    with col_info:
        if not api_key:
            st.warning("请先在左侧侧边栏填入 API Key")
        else:
            st.success(f"已就绪 · 模型：{model}")

    if clicked:
        if not api_key:
            st.warning("⚠️ 还没有填写 API Key，暂时无法生成报告。请在左侧侧边栏输入 API Key 后再点一次。")
        elif df is None or len(df) == 0:
            st.warning("⚠️ 当前还没有数据。请先在上方「数据上传与公共清洗」区域上传 CSV 或载入示例数据。")
        else:
            with st.spinner("智能分析引擎正在深度解析数据，请稍候..."):
                context = build_data_context(df, extra_lines=extra_context)
                report, error = call_qwen(
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=system_prompt,
                    role=role,
                    user_focus=focus,
                    data_context=context,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if error:
                st.error(error)
            else:
                st.session_state[f"{tab_key}_report"] = report
                st.session_state[f"{tab_key}_report_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state[f"{tab_key}_report_role"] = role

    report = st.session_state.get(f"{tab_key}_report")
    if report:
        when = st.session_state.get(f"{tab_key}_report_time", "")
        who = st.session_state.get(f"{tab_key}_report_role", role)

        st.markdown("---")
        st.markdown(
            f"<div class='ai-box'><b>报告生成时间：</b>{when} &nbsp;|&nbsp; <b>智能角色：</b>{who}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(report)  # 用 st.markdown 优雅展示模型的 Markdown 报告

        st.download_button(
            "⬇️ 下载这份智能分析报告（.md）",
            data=report.encode("utf-8"),
            file_name=f"智能分析报告_{tab_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key=f"{tab_key}_dl_report",
            use_container_width=True,
        )
    else:
        st.info("报告区还空着 —— 填好 API Key 后点击「🚀 生成智能洞察报告」即可。")


# ==============================================================================
# ===== 分区 6 =====  词云图模块
# ==============================================================================

def segment_text(text: str, use_jieba: bool = True) -> str:
    """
    把长文本切成“词 + 空格”的形式，方便 WordCloud 统计词频。
    - 装了 jieba：用 jieba 精确分词（中文效果最好）
    - 没装 jieba：英文按单词切，中文按 2~4 字滑窗切（兜底方案，能用但不如 jieba 准）
    """
    text = str(text)
    if use_jieba and HAS_JIEBA:
        return " ".join(jieba.lcut(text))

    tokens = re.findall(r"[A-Za-z][A-Za-z\-']{1,}", text)
    tokens += re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    tokens += re.findall(r"\d+", text)
    return " ".join(tokens)


def make_wordcloud_figure(text: str, max_words: int, bg_color: str, colormap: str, use_jieba: bool = True):
    """生成词云图并返回 (matplotlib Figure, 字体路径, 词元数量)。调用方负责 plt.close。"""
    font_path = get_cjk_font_path()
    segmented = segment_text(text, use_jieba=use_jieba)

    # 停用词：英文用内置的，中文补一份常见虚词
    stopwords = set(STOPWORDS)
    stopwords |= {
        "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个", "上", "也",
        "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
        "以及", "并", "与", "对", "为", "等", "中", "可以", "进行", "这个", "那个", "我们", "他们",
        "the", "and", "for", "with", "that", "this", "are", "was", "from",
    }

    wc = WordCloud(
        font_path=font_path,          # 中文字体，None 时中文会变方块
        width=1600,
        height=800,
        background_color=bg_color,    # 背景色（美观考虑，可切换深色/浅色）
        colormap=colormap,            # 配色方案
        max_words=int(max_words),
        stopwords=stopwords,
        random_state=42,
        collocations=False,           # 关闭二元词组，避免出现奇怪的组合
        prefer_horizontal=0.9,
        margin=4,
    )
    wc.generate(segmented)

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=140)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig, font_path, len(segmented.split())


# ==============================================================================
# ===== 分区 7 =====  侧边栏（API Key 输入区）
# ==============================================================================

with st.sidebar:
    st.markdown("### 🔐 API 配置")
    st.caption("Key 只保存在当前浏览器会话的内存中，不会被写入代码或磁盘。")

    # 关键点：绝不硬编码 Key。默认值仅从环境变量读取（读不到就是空字符串）
    # 规范写法：默认值只通过 session_state 初始化一次
    st.session_state.setdefault("api_key_input", os.environ.get("DASHSCOPE_API_KEY", ""))
    st.text_input(
        "API Key",
        type="password",                                  # 密码框，输入内容显示成圆点
        key="api_key_input",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="到阿里云百炼平台的 API Keys 页面创建。",
    )

    _key_now = (st.session_state.get("api_key_input", "") or "").strip()
    if not _key_now:
        st.error("尚未填写 Key —— 智能分析功能不可用（其它分析功能仍可正常使用）")
    elif _key_now.startswith("sk-"):
        st.success(f"Key 已载入：{_key_now[:6]}...{_key_now[-4:]}（共 {len(_key_now)} 位）")
    else:
        st.warning("这个 Key 不是以 sk- 开头，请确认是否复制完整。")

    # —— 全平台自定义接入：接口地址与模型名称都可以随意替换 ——
    # 规范写法：默认值只通过 session_state 初始化一次，
    # 之后不要在每次重跑时重复传 value=，否则会与组件状态机制冲突
    st.session_state.setdefault("base_url_input", DEFAULT_BASE_URL)
    st.session_state.setdefault("model_input", DEFAULT_MODEL)
    st.session_state.setdefault("custom_prompt", AI_SYSTEM_TEMPLATE)

    base_url_value = st.text_input(
        "接口地址（base_url）",
        key="base_url_input",
        help="接口地址（base_url）一般需要以 /v1 结尾。",
    )
    st.text_input(
        "模型名称（model）",
        key="model_input",
        placeholder="qwen-plus / deepseek-chat / gpt-4o-mini ...",
        help="填你的接口服务支持的任意模型名称。",
    )
    st.caption("💡 如果您使用中转API或其他平台，请在此处替换为您自己的接口地址和模型名称。")

    if not ((base_url_value or "").strip().rstrip("/").endswith("/v1")):
        st.caption("⚠️ 接口地址（base_url）一般需要以 /v1 结尾，请确认是否填写正确。")

    with st.expander("⚙️ 高级参数（可选）", expanded=False):
        st.slider("temperature（越低越严谨）", 0.0, 1.5, 0.4, 0.1, key="temperature")
        st.slider("max_tokens（报告最大长度）", 500, 4000, 2000, 100, key="max_tokens")

    with st.expander("🧠 AI 提示词自定义（高级，可选）", expanded=False):
        st.caption(
            "这是发给大模型的系统提示词模板：清空 = 使用内置三段式分析；"
            "修改后将以你的为准。模板中的 {role_prompt} 会自动替换为你选择的智能角色设定。"
        )
        st.text_area(
            "自定义系统提示词",
            height=260,
            key="custom_prompt",
        )
        if st.button("♻️ 恢复默认提示词", key="reset_prompt", use_container_width=True,
                     on_click=lambda: st.session_state.update({"custom_prompt": AI_SYSTEM_TEMPLATE})):
            st.success("已恢复默认提示词。")

    st.markdown("---")

    if st.button("🧹 清空所有智能分析报告缓存", use_container_width=True, key="clear_ai_cache"):
        for k in list(st.session_state.keys()):
            if str(k).endswith("_report") or str(k).endswith("_report_time") or str(k).endswith("_report_role"):
                del st.session_state[k]
        st.success("已清空智能分析报告缓存。")

    if not HAS_JIEBA:
        st.info("💡 未检测到 jieba，中文词云会用兜底分词。执行 `pip install jieba` 效果更好。")
    if get_cjk_font_path() is None:
        st.warning("未找到中文字体文件，词云中的中文可能显示为方块。")

    st.markdown("---")
    st.caption("© 智能数据清洗与可视化分析平台 · 支持商业 / 医疗 / 科研数据")


# ==============================================================================
# ===== 分区 8 =====  页面顶部标题区
# ==============================================================================

st.markdown("<div class='main-title'>📊 智能数据清洗与可视化分析平台</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='sub-title'>上传 CSV → 一键清洗 → 交互式可视化 → 让智能分析帮你写出专业的分析报告"
    "（商业 / 医疗 / 科研数据通用）</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<span class='pill'>🧹 智能清洗</span>"
    "<span class='pill'>📈 Plotly 交互图表</span>"
    "<span class='pill'>🔗 相关性分析</span>"
    "<span class='pill'>☁️ 中文词云</span>"
    "<span class='pill'>🧪 标准化预处理</span>"
    "<span class='pill'>🤖 深度数据洞察</span>",
    unsafe_allow_html=True,
)

st.markdown("")


# ==============================================================================
# ===== 分区 9 =====  公共操作区：数据上传 + 清洗前后对比（三个标签页共享）
# ==============================================================================

st.markdown("## 📂 数据上传与公共清洗")
st.caption("这一步的清洗结果会被下面三个标签页共同使用，清洗一次，全站复用。")

# session_state 初始化：只需要初始化一次
st.session_state.setdefault("raw_df", None)
st.session_state.setdefault("clean_df", None)
st.session_state.setdefault("clean_metrics", {})
st.session_state.setdefault("clean_logs", [])
st.session_state.setdefault("source_name", "")

with st.container(border=True):

    up_col, demo_col = st.columns([1.5, 1])

    # ---------- 上传区 ----------
    with up_col:
        uploaded_file = st.file_uploader(
            "📂 上传 CSV 文件（支持 UTF-8 / GBK 等中文编码自动识别）",
            type=["csv", "txt"],
            key="file_uploader_main",
            help="如果上传 Excel 请先另存为 CSV；单文件建议不超过 200MB。",
        )
        if uploaded_file is not None:
            try:
                raw_df = load_csv_bytes(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state["raw_df"] = raw_df
                st.session_state["source_name"] = uploaded_file.name
                st.success(f"✅ 已读取「{uploaded_file.name}」：{raw_df.shape[0]} 行 × {raw_df.shape[1]} 列")
            except Exception as err:  # noqa: BLE001
                st.error(f"读取文件失败：{err}")

    # ---------- 示例数据区 ----------
    with demo_col:
        st.markdown("**🎁 没有数据？一键载入内置示例数据**")
        d1, d2, d3 = st.columns(3)
        if d1.button("商业销售", use_container_width=True, key="demo_biz"):
            st.session_state["raw_df"] = build_demo_data("商业销售数据")
            st.session_state["source_name"] = "示例数据_商业销售"
        if d2.button("医疗健康", use_container_width=True, key="demo_med"):
            st.session_state["raw_df"] = build_demo_data("医疗健康数据")
            st.session_state["source_name"] = "示例数据_医疗健康"
        if d3.button("科研实验", use_container_width=True, key="demo_sci"):
            st.session_state["raw_df"] = build_demo_data("科研实验数据")
            st.session_state["source_name"] = "示例数据_科研实验"
        st.caption("示例数据里故意混入了缺失值、重复行和异常值，方便你观察清洗效果。")

    raw_df = st.session_state.get("raw_df")

    if raw_df is not None and len(raw_df) > 0:

        st.markdown("---")
        st.markdown("#### ⚙️ 清洗参数设置")

        p1, p2, p3 = st.columns(3)
        with p1:
            missing_strategy = st.radio(
                "缺失值处理策略",
                [
                    "智能填充（数值→中位数，文本→众数）",
                    "删除含缺失值的行",
                    "仅删除全空行（保留部分缺失）",
                    "不做处理，只去重",
                ],
                key="missing_strategy",
                help="推荐：数据量大用填充，数据量小且缺失随机可用删除。",
            )
        with p2:
            drop_dup = st.checkbox("删除完全重复的行", value=True, key="opt_drop_dup")
            strip_ws = st.checkbox("清理文本中的多余空格 / 换行", value=True, key="opt_strip")
        with p3:
            auto_num = st.checkbox("智能类型转换（把数字文本转成数值）", value=True, key="opt_auto_num")
            outlier_action = st.selectbox(
                "异常值处理",
                ["仅标记，不删除", "按 IQR 规则删除异常行"],
                key="opt_outlier",
                help="IQR = 四分位距，超出 Q1-1.5×IQR ~ Q3+1.5×IQR 视为异常值。",
            )

        # 执行清洗
        try:
            clean_df, metrics, logs = clean_dataframe(
                raw_df,
                missing_strategy=missing_strategy,
                drop_duplicates=drop_dup,
                strip_whitespace=strip_ws,
                auto_convert_numeric=auto_num,
                outlier_action=outlier_action,
            )
            st.session_state["clean_df"] = clean_df
            st.session_state["clean_metrics"] = metrics
            st.session_state["clean_logs"] = logs
        except Exception as err:  # noqa: BLE001
            st.error(f"清洗过程出现异常：{err}")
            clean_df = raw_df
            metrics, logs = {}, []

        # ---------- 清洗指标卡片 ----------
        st.markdown("#### 📌 清洗结果总览")
        m = metrics
        removed_rows = m.get("删除行数合计", 0)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("原始行数", human_int(m.get("原始行数", 0)))
        c2.metric("清洗后行数", human_int(m.get("清洗后行数", 0)),
                  delta=f"-{human_int(removed_rows)}" if removed_rows else None)
        c3.metric("删除重复行", human_int(m.get("删除重复行数", 0)))
        c4.metric("填充缺失值", human_int(m.get("填充单元格数", 0)))
        c5.metric("删除异常行", human_int(m.get("删除异常行数", 0)))
        c6.metric("数据完整率", f"{m.get('完整率', 0) * 100:.2f}%")

        # ---------- 清洗前后对比 ----------
        st.markdown("#### 🔍 清洗前后对比（各展示前 200 行）")

        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown(f"**🧾 清洗前 · 原始数据**　`{raw_df.shape[0]} 行 × {raw_df.shape[1]} 列`")
            st.dataframe(raw_df.head(200), use_container_width=True, height=340)
        with after_col:
            st.markdown(f"**✨ 清洗后 · 干净数据**　`{clean_df.shape[0]} 行 × {clean_df.shape[1]} 列`")
            st.dataframe(clean_df.head(200), use_container_width=True, height=340)

        # ---------- 缺失值对比表 ----------
        with st.expander("📋 查看各字段缺失值变化明细", expanded=False):
            miss = pd.DataFrame({
                "字段": list(raw_df.columns),
                "清洗前缺失数": [int(raw_df[c].isna().sum()) if c in raw_df.columns else 0 for c in raw_df.columns],
            })
            miss["清洗前缺失率%"] = (miss["清洗前缺失数"] / max(len(raw_df), 1) * 100).round(2)
            miss["清洗后缺失数"] = [
                int(clean_df[c].isna().sum()) if c in clean_df.columns else "（该列已删除）" for c in raw_df.columns
            ]
            st.dataframe(miss.sort_values("清洗前缺失数", ascending=False), use_container_width=True)

        # ---------- 下载区 ----------
        st.markdown("#### ⬇️ 一键下载")
        dl1, dl2, dl3 = st.columns([1, 1, 2])
        with dl1:
            st.download_button(
                "⬇️ 下载清洗后的数据（CSV）",
                data=df_to_csv_bytes(clean_df),
                file_name=f"cleaned_{st.session_state.get('source_name', 'data')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="dl_clean_csv",
                type="primary",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ 下载原始数据（CSV）",
                data=df_to_csv_bytes(raw_df),
                file_name=f"raw_{st.session_state.get('source_name', 'data')}.csv",
                mime="text/csv",
                key="dl_raw_csv",
                use_container_width=True,
            )
        with dl3:
            st.caption("提示：下载文件使用 UTF-8-SIG 编码，Excel 双击打开不会乱码。")

        # ---------- 清洗日志 ----------
        with st.expander("🧾 查看清洗过程日志（新手调试推荐看这里）", expanded=False):
            for line in logs:
                st.write(line)

    else:
        st.info("👆 请上传一个 CSV 文件，或点击右侧的示例数据按钮开始体验。")


# 取出最终使用的清洗后数据（三个标签页共用这一份）
df = st.session_state.get("clean_df")
if df is None:
    df = st.session_state.get("raw_df")

raw_df = st.session_state.get("raw_df")

st.markdown("---")


# ==============================================================================
# ===== 分区 10 =====  标签页1【通用清洗与探索】
# ==============================================================================

# st.tabs 用法：一次解包出三个标签页对象，之后用 with 标签页对象: 把内容画进对应标签页
tab1, tab2, tab3 = st.tabs(["🧹 通用清洗与探索", "💼 商业与市场分析", "🔬 科研与深度学习预处理"])

# ------------------------------------------------------------------ 标签页 1
with tab1:

    st.caption("适合任何行业的数据：先看异常值，再画图，最后看变量之间的相关性。")

    left1, right1 = st.columns([1.15, 1], gap="large")

    # ========== 左侧：操作区 ==========
    with left1:

        if require_data(df, "通用清洗与探索"):

            # ---------- 异常值检测 ----------
            st.markdown("#### 📦 异常值检测（箱线图）")

            num_cols = numeric_columns(df)
            cat_cols = categorical_columns(df)

            oc1, oc2 = st.columns(2)
            with oc1:
                y_col = safe_selectbox("数值列（要检测的字段）", num_cols, key="t1_box_y",
                                       format_func=col_labeler(df))
            with oc2:
                group_col = st.selectbox(
                    "分组字段（可选，用于对比不同组）",
                    ["（不分组）"] + cat_cols,
                    key="t1_box_group",
                    format_func=col_labeler(df),
                )

            if y_col:
                series = pd.to_numeric(df[y_col], errors="coerce").dropna()

                if len(series) >= 4:
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = series[(series < lower) | (series > upper)]

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("均值", f"{series.mean():.3f}")
                    s2.metric("中位数", f"{series.median():.3f}")
                    s3.metric("异常值个数", f"{len(outliers)}")
                    s4.metric("异常值占比", f"{len(outliers) / max(len(series), 1) * 100:.2f}%")
                    st.caption(f"IQR = {iqr:.3f}；正常值区间：{lower:.3f} ~ {upper:.3f}")

                    try:
                        if group_col != "（不分组）":
                            fig_box = px.box(
                                df, x=group_col, y=y_col, color=group_col,
                                points="outliers", template=PLOTLY_TEMPLATE,
                                color_discrete_sequence=COLOR_SEQ,
                                title=f"{y_col} 按「{group_col}」分组的箱线图",
                            )
                        else:
                            fig_box = px.box(
                                df, y=y_col, points="all", template=PLOTLY_TEMPLATE,
                                color_discrete_sequence=COLOR_SEQ,
                                title=f"{y_col} 的箱线图（圆点 = 每个数据点）",
                            )
                        fig_box.update_layout(height=430, showlegend=(group_col != "（不分组）"))
                        st.plotly_chart(fig_box, use_container_width=True)
                    except Exception as err:  # noqa: BLE001
                        st.error(f"箱线图绘制失败：{err}")

                    if len(outliers) > 0:
                        with st.expander(f"🔎 查看 {len(outliers)} 个异常值明细", expanded=False):
                            out_table = df.loc[outliers.index].copy()
                            out_table["_异常程度(偏离中位数倍数)"] = (
                                (out_table[y_col] - series.median()).abs() / (series.std(ddof=0) + 1e-9)
                            ).round(2)
                            st.dataframe(
                                out_table.sort_values("_异常程度(偏离中位数倍数)", ascending=False).head(100),
                                use_container_width=True,
                            )
                else:
                    st.info("该字段有效数据少于 4 条，无法做箱线图分析。")

            st.markdown("---")

            # ---------- 通用绘图 ----------
            st.markdown("#### 📈 通用绘图（交互式图表）")

            g1, g2, g3, g4 = st.columns(4)
            with g1:
                chart_type = st.selectbox("图表类型", ["柱状图", "折线图", "饼图", "散点图"], key="t1_chart_type")
            with g2:
                # 数值列自动排在前面，列名后带类型标识（如：销量 (数值) / 城市 (文本)）
                x_col = st.selectbox("X 轴（分类/横轴）", ordered_columns(df), key="t1_x",
                                     format_func=col_labeler(df))
            with g3:
                y_candidates = ["（自动：计数）"] + num_cols
                y_col_plot = st.selectbox("Y 轴（数值/纵轴）", y_candidates, key="t1_y",
                                          format_func=col_labeler(df))
            with g4:
                color_col = st.selectbox("颜色分组（可选）", ["（无）"] + cat_cols, key="t1_color",
                                         format_func=col_labeler(df))

            # 饼图/柱状图需要一个“数值”，没有数值列时用计数
            agg_choice = st.selectbox(
                "数值聚合方式（当 Y 轴为数值时生效）",
                ["求和 sum", "平均 mean", "计数 count", "最大值 max"],
                key="t1_agg",
            )
            agg_map = {"求和 sum": "sum", "平均 mean": "mean", "计数 count": "count", "最大值 max": "max"}
            agg_func = agg_map[agg_choice]

            if st.button("🎨 生成图表", key="t1_draw", type="primary", use_container_width=True):
                st.session_state["t1_draw_flag"] = True

            if st.session_state.get("t1_draw_flag", False):
                try:
                    color_arg = None if color_col == "（无）" else color_col
                    fig = None

                    if chart_type == "饼图":
                        if y_col_plot == "（自动：计数）":
                            pie_data = df[x_col].astype(str).value_counts().reset_index()
                            pie_data.columns = [x_col, "计数"]
                            fig = px.pie(
                                pie_data, names=x_col, values="计数", hole=0.35,
                                template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                                title=f"「{x_col}」占比分布（按计数）",
                            )
                        else:
                            pie_data = df.groupby(x_col, dropna=False)[y_col_plot].agg(agg_func).reset_index()
                            fig = px.pie(
                                pie_data, names=x_col, values=y_col_plot, hole=0.35,
                                template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                                title=f"「{x_col}」的 {y_col_plot}（{agg_choice}）占比",
                            )
                        fig.update_traces(textposition="inside", textinfo="percent+label")

                    elif chart_type == "散点图":
                        if y_col_plot == "（自动：计数）":
                            st.warning("散点图需要选择数值型 Y 轴，已自动使用第一个数值列。")
                            y_use = num_cols[0] if num_cols else None
                        else:
                            y_use = y_col_plot

                        if y_use is None:
                            st.error("数据里没有数值列，无法绘制散点图。")
                        else:
                            fig = px.scatter(
                                df, x=x_col, y=y_use, color=color_arg,
                                template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                                opacity=0.75, title=f"{x_col} 与 {y_use} 的散点关系",
                                hover_data=[c for c in df.columns[:6]],
                            )

                    elif chart_type == "折线图":
                        if y_col_plot == "（自动：计数）":
                            line_data = df.groupby(x_col, dropna=False).size().reset_index(name="计数")
                            fig = px.line(
                                line_data, x=x_col, y="计数", markers=True,
                                template=PLOTLY_TEMPLATE, title=f"「{x_col}」的计数趋势",
                            )
                        else:
                            line_data = df.groupby(x_col, dropna=False)[y_col_plot].agg(agg_func).reset_index()
                            fig = px.line(
                                line_data, x=x_col, y=y_col_plot, color=color_arg, markers=True,
                                template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                                title=f"{y_col_plot} 随「{x_col}」的变化（{agg_choice}）",
                            )

                    else:  # 柱状图
                        if y_col_plot == "（自动：计数）":
                            bar_data = df.groupby(x_col, dropna=False).size().reset_index(name="计数")
                            fig = px.bar(
                                bar_data, x=x_col, y="计数", color=color_arg, text_auto=True,
                                template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                                title=f"「{x_col}」的计数对比",
                            )
                        else:
                            bar_data = df.groupby([x_col] + ([color_arg] if color_arg else []), dropna=False)[y_col_plot].agg(agg_func).reset_index()
                            fig = px.bar(
                                bar_data, x=x_col, y=y_col_plot, color=color_arg, barmode="group",
                                text_auto=".2s", template=PLOTLY_TEMPLATE,
                                color_discrete_sequence=COLOR_SEQ,
                                title=f"{y_col_plot} 按「{x_col}」对比（{agg_choice}）",
                            )

                    if fig is not None:
                        fig.update_layout(height=470, xaxis_tickangle=-30)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as err:  # noqa: BLE001
                    st.error(f"图表生成失败：{err}。可以尝试换一个 X 轴或 Y 轴字段。")

            st.markdown("---")

            # ---------- 相关性热力图 ----------
            st.markdown("#### 🔥 相关性热力图")

            if len(num_cols) < 2:
                st.info("数值列少于 2 个，无法计算相关性。请检查数据类型，或在上方开启「智能类型转换」。")
            else:
                hc1, hc2 = st.columns([2, 1])
                with hc1:
                    default_cols = num_cols[: min(12, len(num_cols))]
                    corr_cols = st.multiselect(
                        "选择参与计算的数值列（建议 3~12 个，太多会看不清）",
                        num_cols, default=default_cols, key="t1_corr_cols",
                        format_func=col_labeler(df),
                    )
                with hc2:
                    corr_method = st.selectbox("相关系数类型", ["pearson", "spearman", "kendall"], key="t1_corr_method")
                    show_text = st.checkbox("在格子里显示数值", value=True, key="t1_corr_text")

                if len(corr_cols) >= 2:
                    try:
                        corr = df[corr_cols].corr(method=corr_method).round(3)
                        fig_heat = go.Figure(
                            data=go.Heatmap(
                                z=corr.values,
                                x=[str(c) for c in corr.columns],
                                y=[str(c) for c in corr.index],
                                colorscale="RdBu_r",   # 蓝(负相关) → 白(无) → 红(正相关)
                                zmin=-1, zmax=1,
                                text=corr.values,
                                texttemplate="%{text}" if show_text else None,
                                textfont={"size": 10},
                                colorbar={"title": "相关系数"},
                                hovertemplate="%{y} ↔ %{x}<br>相关系数 = %{z:.3f}<extra></extra>",
                            )
                        )
                        fig_heat.update_layout(
                            template=PLOTLY_TEMPLATE,
                            height=max(420, 45 * len(corr_cols)),
                            title=f"{corr_method} 相关系数热力图",
                            xaxis={"tickangle": -30},
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

                        # 顺带列出最强相关组合，方便写进上下文
                        with st.expander("🔗 查看相关性最强的变量组合", expanded=False):
                            abs_corr = corr.abs()
                            pairs = abs_corr.where(~np.eye(len(abs_corr), dtype=bool)).stack().sort_values(ascending=False)
                            rows, seen = [], set()
                            for (a, b), v in pairs.items():
                                k = tuple(sorted([str(a), str(b)]))
                                if k in seen:
                                    continue
                                seen.add(k)
                                rows.append({"变量A": a, "变量B": b, "相关系数": round(float(v), 3),
                                             "方向": "正相关" if corr.loc[a, b] > 0 else "负相关"})
                                if len(rows) >= 10:
                                    break
                            if rows:
                                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    except Exception as err:  # noqa: BLE001
                        st.error(f"相关性热力图生成失败：{err}")
                else:
                    st.info("请至少选择 2 个数值列。")

            # ---------- 数据质量体检表（给模型提供上下文） ----------
            with st.expander("🩺 数据质量体检表（清洗后）", expanded=False):
                quality = pd.DataFrame({
                    "字段": list(df.columns),
                    "类型": [str(t) for t in df.dtypes],
                    "非空数": df.notna().sum().values,
                    "缺失率%": (df.isna().mean().values * 100).round(2),
                    "唯一值数": df.nunique().values,
                })
                st.dataframe(quality, use_container_width=True)

    # ========== 右侧：智能报告区 ==========
    with right1:

        extra_t1 = ""
        if df is not None and len(df) > 0:
            try:
                num_cols_now = numeric_columns(df)
                if num_cols_now:
                    corr_now = df[num_cols_now].corr().round(3)
                    extra_t1 = "通用探索视角计算出的相关系数矩阵：\n" + corr_now.to_string()
            except Exception:  # noqa: BLE001
                extra_t1 = ""

        render_ai_panel(df, tab_key="tab1", title="🤖 通用智能数据洞察", extra_context=extra_t1)


# ==============================================================================
# ===== 分区 11 =====  标签页2【商业与市场分析】
# ==============================================================================

with tab2:

    st.caption("适合销售、市场、运营、客户数据：交叉分析找结构，词云看口碑，图表做汇报。")

    left2, right2 = st.columns([1.15, 1], gap="large")

    # ========== 左侧：操作区 ==========
    with left2:

        if require_data(df, "商业与市场分析"):

            cat_cols = categorical_columns(df)
            num_cols = numeric_columns(df)
            txt_cols = text_columns(df)

            # ---------- 交叉分析 ----------
            st.markdown("#### 🔀 交叉分析（两个分类维度）")

            if len(cat_cols) < 2:
                st.info("分类列少于 2 个，无法做交叉分析。请检查数据，或使用「商业销售」示例数据体验。")
            else:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    row_dim = st.selectbox("行维度（第一个分类列）", cat_cols, key="t2_row",
                                           format_func=col_labeler(df))
                with cc2:
                    candidates_col = [c for c in cat_cols if c != row_dim] or cat_cols
                    col_dim = st.selectbox("列维度（第二个分类列）", candidates_col, key="t2_col",
                                           format_func=col_labeler(df))
                with cc3:
                    value_col = st.selectbox("统计值字段", ["（只统计数量）"] + num_cols, key="t2_value",
                                             format_func=col_labeler(df))

                ac1, ac2 = st.columns(2)
                with ac1:
                    agg_label = st.selectbox("聚合方式", ["计数", "求和", "平均", "最大值"], key="t2_agg")
                with ac2:
                    normalize_label = st.selectbox(
                        "显示方式", ["原始数值", "行百分比%", "列百分比%"], key="t2_norm"
                    )
                agg_func_map = {"计数": "count", "求和": "sum", "平均": "mean", "最大值": "max"}

                try:
                    if value_col == "（只统计数量）" or agg_label == "计数":
                        base = pd.crosstab(df[row_dim], df[col_dim])
                        table_title = f"「{row_dim}」 × 「{col_dim}」数量交叉表"
                    else:
                        base = pd.pivot_table(
                            df, index=row_dim, columns=col_dim, values=value_col,
                            aggfunc=agg_func_map[agg_label], fill_value=0,
                        )
                        table_title = f"「{row_dim}」 × 「{col_dim}」的 {value_col}（{agg_label}）交叉表"

                    base = base.astype(float)

                    if normalize_label == "行百分比%":
                        display = base.div(base.sum(axis=1).replace(0, np.nan), axis=0) * 100
                    elif normalize_label == "列百分比%":
                        display = base.div(base.sum(axis=0).replace(0, np.nan), axis=1) * 100
                    else:
                        display = base.copy()

                    display = display.round(2)
                    display_with_margin = add_margins(display)

                    st.markdown(f"**{table_title}**（{normalize_label}）")
                    st.dataframe(display_with_margin, use_container_width=True)

                    # 交叉表热力图
                    try:
                        fig_cross = go.Figure(
                            data=go.Heatmap(
                                z=display.values,
                                x=[str(c) for c in display.columns],
                                y=[str(i) for i in display.index],
                                colorscale="YlGnBu",
                                text=display.values,
                                texttemplate="%{text}",
                                textfont={"size": 10},
                                hovertemplate=f"{row_dim}=%{{y}}<br>{col_dim}=%{{x}}<br>数值=%{{z}}<extra></extra>",
                                colorbar={"title": normalize_label},
                            )
                        )
                        fig_cross.update_layout(
                            template=PLOTLY_TEMPLATE,
                            height=max(380, 40 * len(display.index)),
                            title=f"交叉分析热力图 · {row_dim} × {col_dim}",
                            xaxis={"tickangle": -30},
                        )
                        st.plotly_chart(fig_cross, use_container_width=True)
                    except Exception as err:  # noqa: BLE001
                        st.warning(f"交叉表热力图绘制失败：{err}")

                    # 交叉表下载
                    st.download_button(
                        "⬇️ 下载这张交叉表（CSV）",
                        data=df_to_csv_bytes(display_with_margin.reset_index()),
                        file_name=f"交叉表_{row_dim}_{col_dim}.csv",
                        mime="text/csv",
                        key="t2_dl_cross",
                        use_container_width=True,
                    )

                except Exception as err:  # noqa: BLE001
                    st.error(f"交叉分析失败：{err}")

            st.markdown("---")

            # ---------- 词云图 ----------
            st.markdown("#### ☁️ 文本词云图（看口碑 / 看反馈热点）")

            if not txt_cols:
                st.info("没有检测到长文本列（平均长度 ≥ 8 个字符的文本字段），词云功能暂不可用。")
            else:
                wc1, wc2 = st.columns(2)
                with wc1:
                    text_col = st.selectbox("选择文本列", txt_cols, key="t2_text_col",
                                            format_func=col_labeler(df))
                    max_words = st.slider("最多显示多少个词", 30, 400, 150, 10, key="t2_wc_words")
                with wc2:
                    bg_color = st.selectbox("背景色", ["white", "black", "#f8fafc", "#0f172a", "#1e293b"], key="t2_wc_bg")
                    colormap = st.selectbox("配色方案", CONTINUOUS_SCALES, key="t2_wc_cmap")

                use_jieba_flag = st.checkbox(
                    f"使用 jieba 中文分词（当前 {'已安装 ✅' if HAS_JIEBA else '未安装，将使用兜底分词'}）",
                    value=True, key="t2_use_jieba", disabled=not HAS_JIEBA,
                )

                if st.button("☁️ 生成词云图", key="t2_wc_run", type="primary", use_container_width=True):
                    st.session_state["t2_wc_flag"] = True

                # 用 session_state 记住“已生成”，这样后续交互时词云不会消失
                if st.session_state.get("t2_wc_flag", False):
                    with st.spinner("正在统计词频并绘制词云..."):
                        try:
                            text_blob = " ".join(
                                df[text_col].dropna().astype(str).tolist()[:20000]
                            )
                            if not text_blob.strip():
                                st.warning("这一列没有可用文本内容。")
                            else:
                                fig_wc, font_used, token_count = make_wordcloud_figure(
                                    text_blob, max_words, bg_color, colormap, use_jieba=use_jieba_flag
                                )
                                st.pyplot(fig_wc)

                                # 先把图片存进内存缓冲区（必须在 plt.close 之前做）
                                buf = io.BytesIO()
                                fig_wc.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=bg_color)
                                buf.seek(0)
                                png_bytes = buf.getvalue()
                                plt.close(fig_wc)  # 释放内存，必须关闭

                                st.caption(
                                    f"共分析 {token_count} 个词元；字体：{font_used or '未找到中文字体（中文可能显示为方块）'}"
                                )
                                st.download_button(
                                    "⬇️ 下载词云图（PNG）",
                                    data=png_bytes,
                                    file_name=f"词云_{text_col}.png",
                                    mime="image/png",
                                    key="t2_wc_png",
                                    use_container_width=True,
                                )
                        except Exception as err:  # noqa: BLE001
                            st.error(f"词云生成失败：{err}")

            st.markdown("---")

            # ---------- 对比图表 ----------
            st.markdown("#### 📊 对比图表（柱状图 / 饼图）")

            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                biz_cat = st.selectbox("分类维度", cat_cols if cat_cols else list(df.columns), key="t2_chart_cat",
                                       format_func=col_labeler(df))
            with bc2:
                biz_val = st.selectbox("数值指标", ["（计数）"] + num_cols, key="t2_chart_val",
                                       format_func=col_labeler(df))
            with bc3:
                biz_agg = st.selectbox("聚合方式", ["求和", "平均", "计数", "最大值"], key="t2_chart_agg")
            with bc4:
                biz_type = st.selectbox("图表类型", ["柱状图", "条形图", "饼图"], key="t2_chart_type")

            top_n = st.slider("只显示前 N 个类别（0 = 全部显示）", 0, 40, 12, 1, key="t2_topn")

            if st.button("🎨 生成对比图表", key="t2_chart_run", type="primary", use_container_width=True):
                st.session_state["t2_chart_flag"] = True

            # 用 session_state 记住“已生成”，避免图表在下一次交互后消失
            if st.session_state.get("t2_chart_flag", False):
                try:
                    biz_agg_map = {"求和": "sum", "平均": "mean", "计数": "count", "最大值": "max"}

                    if biz_val == "（计数）":
                        agg_data = df.groupby(biz_cat, dropna=False).size().reset_index(name="计数")
                        val_field = "计数"
                    else:
                        agg_data = df.groupby(biz_cat, dropna=False)[biz_val].agg(biz_agg_map[biz_agg]).reset_index()
                        val_field = biz_val

                    agg_data = agg_data.sort_values(val_field, ascending=False)
                    if top_n > 0:
                        agg_data = agg_data.head(top_n)

                    if biz_type == "饼图":
                        fig_biz = px.pie(
                            agg_data, names=biz_cat, values=val_field, hole=0.32,
                            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                            title=f"「{biz_cat}」的 {val_field} 占比",
                        )
                        fig_biz.update_traces(textposition="inside", textinfo="percent+label")
                    elif biz_type == "条形图":
                        fig_biz = px.bar(
                            agg_data.sort_values(val_field), x=val_field, y=biz_cat,
                            orientation="h", text_auto=".2s", template=PLOTLY_TEMPLATE,
                            color=biz_cat, color_discrete_sequence=COLOR_SEQ,
                            title=f"「{biz_cat}」的 {val_field} 排名（{biz_agg}）",
                        )
                    else:
                        fig_biz = px.bar(
                            agg_data, x=biz_cat, y=val_field, text_auto=".2s",
                            template=PLOTLY_TEMPLATE, color=biz_cat,
                            color_discrete_sequence=COLOR_SEQ,
                            title=f"「{biz_cat}」的 {val_field} 对比（{biz_agg}）",
                        )

                    fig_biz.update_layout(height=470, showlegend=(biz_type == "饼图"), xaxis_tickangle=-30)
                    st.plotly_chart(fig_biz, use_container_width=True)

                    with st.expander("📋 查看聚合后的明细数据", expanded=False):
                        st.dataframe(agg_data, use_container_width=True)
                        st.download_button(
                            "⬇️ 下载聚合明细（CSV）",
                            data=df_to_csv_bytes(agg_data),
                            file_name=f"聚合明细_{biz_cat}.csv",
                            mime="text/csv",
                            key="t2_dl_agg",
                        )

                except Exception as err:  # noqa: BLE001
                    st.error(f"图表生成失败：{err}")

    # ========== 右侧：智能报告区 ==========
    with right2:

        extra_t2 = ""
        if df is not None and len(df) > 0:
            try:
                cat_cols_now = categorical_columns(df)
                num_cols_now = numeric_columns(df)
                pieces = []

                for c in cat_cols_now[:3]:
                    pieces.append(f"字段「{c}」的取值分布：\n" + df[c].astype(str).value_counts().head(10).to_string())

                if cat_cols_now and num_cols_now:
                    first_cat, first_num = cat_cols_now[0], num_cols_now[0]
                    pieces.append(
                        f"「{first_cat}」分组下「{first_num}」的均值：\n"
                        + df.groupby(first_cat)[first_num].mean().round(3).to_string()
                    )

                extra_t2 = "\n\n".join(pieces)
            except Exception:  # noqa: BLE001
                extra_t2 = ""

        render_ai_panel(df, tab_key="tab2", title="🤖 商业智能洞察", extra_context=extra_t2)


# ==============================================================================
# ===== 分区 12 =====  标签页3【科研与深度学习预处理】
# ==============================================================================

with tab3:

    st.caption("适合实验数据、医疗指标、模型训练前的特征工程：标准化 + 分布检查 + 建模可用性检查。")

    left3, right3 = st.columns([1.15, 1], gap="large")

    # ========== 左侧：操作区 ==========
    with left3:

        if require_data(df, "科研与深度学习预处理"):

            num_cols = numeric_columns(df)

            # ---------- 数据标准化 ----------
            st.markdown("#### ⚖️ 数据标准化（Z-score / Min-Max）")

            if not num_cols:
                st.info("没有数值列可用于标准化。请检查数据，或使用「科研实验」示例数据体验。")
            else:
                sc1, sc2 = st.columns([2, 1])
                with sc1:
                    scale_cols = st.multiselect(
                        "选择需要标准化的数值列",
                        num_cols,
                        default=num_cols[: min(5, len(num_cols))],
                        key="t3_scale_cols",
                        format_func=col_labeler(df),
                    )
                with sc2:
                    scale_method = st.radio("标准化方法", ["Z-score 标准化", "Min-Max 归一化"], key="t3_scale_method")

                st.markdown(
                    """
                    <div class='card'>
                    <b>Z-score 标准化</b>：x' = (x - 均值) / 标准差 → 结果均值≈0、标准差≈1，适合大多数深度学习模型。<br>
                    <b>Min-Max 归一化</b>：x' = (x - 最小值) / (最大值 - 最小值) → 结果压缩到 [0, 1]，适合图像、神经网络输入。
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if scale_cols:
                    try:
                        original = df[scale_cols].astype(float).copy()

                        if scale_method == "Z-score 标准化":
                            mean_v = original.mean()
                            std_v = original.std(ddof=0).replace(0, np.nan)  # 防止某列全相同导致除 0
                            scaled = (original - mean_v) / std_v
                            scaled = scaled.fillna(0.0)
                            formula_note = "Z-score：(x - 均值) / 标准差（标准差为 0 的列已置 0）"
                        else:
                            min_v = original.min()
                            max_v = original.max()
                            rng_v = (max_v - min_v).replace(0, np.nan)
                            scaled = (original - min_v) / rng_v
                            scaled = scaled.fillna(0.0)
                            formula_note = "Min-Max：(x - 最小值) / (最大值 - 最小值)（极差为 0 的列已置 0）"

                        st.caption(f"📐 采用公式：{formula_note}")

                        # 前后统计对比
                        stat_before = original.describe().T[["mean", "std", "min", "max"]].round(4)
                        stat_before.columns = ["均值(前)", "标准差(前)", "最小值(前)", "最大值(前)"]
                        stat_after = scaled.describe().T[["mean", "std", "min", "max"]].round(4)
                        stat_after.columns = ["均值(后)", "标准差(后)", "最小值(后)", "最大值(后)"]
                        compare = pd.concat([stat_before, stat_after], axis=1)

                        st.markdown("**标准化前后统计量对比**")
                        st.dataframe(compare, use_container_width=True)

                        # 标准化结果预览
                        with st.expander("👀 查看标准化后的数据（前 200 行）", expanded=True):
                            st.dataframe(scaled.head(200), use_container_width=True)

                        # 对比箱线图
                        try:
                            plot_cols = scale_cols[: min(6, len(scale_cols))]
                            merged = pd.concat(
                                [
                                    original[plot_cols].assign(_阶段="标准化前"),
                                    scaled[plot_cols].assign(_阶段="标准化后"),
                                ],
                                axis=0,
                                ignore_index=True,
                            )
                            fig_scale = px.box(
                                merged, x="_阶段", y=plot_cols, color="_阶段",
                                points="outliers", template=PLOTLY_TEMPLATE,
                                color_discrete_sequence=["#ef4444", "#2563eb"],
                                title="标准化前后分布对比（箱线图）",
                            )
                            fig_scale.update_layout(height=470)
                            st.plotly_chart(fig_scale, use_container_width=True)
                        except Exception as err:  # noqa: BLE001
                            st.warning(f"对比图绘制失败：{err}")

                        # 下载标准化结果
                        export_scaled = pd.concat(
                            [df.drop(columns=scale_cols, errors="ignore").reset_index(drop=True),
                             scaled.add_suffix("_标准化").reset_index(drop=True)],
                            axis=1,
                        )

                        st.download_button(
                            "⬇️ 下载标准化后的数据（CSV，可用于建模）",
                            data=df_to_csv_bytes(export_scaled),
                            file_name=f"scaled_{scale_method.split()[0]}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            key="t3_dl_scaled",
                            type="primary",
                            use_container_width=True,
                        )

                    except Exception as err:  # noqa: BLE001
                        st.error(f"标准化处理失败：{err}")
                else:
                    st.info("请至少选择一个数值列。")

            st.markdown("---")

            # ---------- 特征分布图 ----------
            st.markdown("#### 📊 特征分布图（直方图 / 密度曲线）")

            if not num_cols:
                st.info("没有数值列可绘制分布图。")
            else:
                fc1, fc2, fc3 = st.columns([2, 1, 1])
                with fc1:
                    dist_cols = st.multiselect(
                        "选择要查看分布的连续变量（可多选）",
                        num_cols,
                        default=num_cols[: min(4, len(num_cols))],
                        key="t3_dist_cols",
                        format_func=col_labeler(df),
                    )
                with fc2:
                    bins = st.slider("分箱数量（bins）", 5, 120, 30, key="t3_bins")
                with fc3:
                    show_box = st.checkbox("显示边际箱线图", value=True, key="t3_marginal")

                if dist_cols:
                    try:
                        fig_dist = px.histogram(
                            df, x=dist_cols, nbins=bins, barmode="overlay", opacity=0.62,
                            template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                            marginal="box" if show_box else None,
                            title="连续变量的分布直方图（叠加显示）",
                        )
                        fig_dist.update_layout(height=520, bargap=0.02)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    except Exception as err:  # noqa: BLE001
                        st.error(f"分布图绘制失败：{err}")

                    # 单变量的精确直方图 + 均值/中位数参考线
                    focus_col = st.selectbox("单独细看某一个变量", dist_cols, key="t3_focus_col",
                                             format_func=col_labeler(df))

                    try:
                        fig_single = px.histogram(
                            df, x=focus_col, nbins=bins, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=["#6366f1"], title=f"「{focus_col}」的分布细节",
                        )

                        mean_val = float(pd.to_numeric(df[focus_col], errors="coerce").mean())
                        median_val = float(pd.to_numeric(df[focus_col], errors="coerce").median())

                        fig_single.add_vline(x=mean_val, line_dash="dash", line_color="#ef4444",
                                             annotation_text=f"均值 {mean_val:.3f}", annotation_position="top")
                        fig_single.add_vline(x=median_val, line_dash="dot", line_color="#16a34a",
                                             annotation_text=f"中位数 {median_val:.3f}", annotation_position="bottom")
                        fig_single.update_layout(height=430)
                        st.plotly_chart(fig_single, use_container_width=True)

                        sk = float(pd.to_numeric(df[focus_col], errors="coerce").skew())
                        ku = float(pd.to_numeric(df[focus_col], errors="coerce").kurtosis())

                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("偏度 skew", f"{sk:.3f}", help="|偏度|>1 说明分布明显不对称，建模时可考虑对数变换。")
                        k2.metric("峰度 kurtosis", f"{ku:.3f}", help="峰度高说明尖峰厚尾，异常值影响较大。")
                        k3.metric("均值", f"{mean_val:.3f}")
                        k4.metric("中位数", f"{median_val:.3f}")

                    except Exception as err:  # noqa: BLE001
                        st.warning(f"单变量分布图绘制失败：{err}")

            st.markdown("---")

            # ---------- 缺失值与建模可用性检查 ----------
            st.markdown("#### 🧩 建模可用性检查")

            c1, c2 = st.columns(2)
            with c1:
                completeness = df.notna().mean().sort_values()
                fig_miss = px.bar(
                    x=completeness.values * 100, y=[str(i) for i in completeness.index],
                    orientation="h", template=PLOTLY_TEMPLATE,
                    color=completeness.values * 100, color_continuous_scale="RdYlGn",
                    labels={"x": "完整率 %", "y": "字段"}, title="各字段数据完整率",
                )
                fig_miss.update_layout(height=max(360, 26 * len(completeness)), coloraxis_showscale=False)
                st.plotly_chart(fig_miss, use_container_width=True)

            with c2:
                st.markdown("**训练前检查清单**")

                row_ok = len(df) >= 100
                miss_rate = float(df.isna().mean().mean())
                numeric_enough = len(num_cols) >= 2
                dup_rate = float(df.duplicated().mean()) if len(df) else 0.0

                checklist = pd.DataFrame([
                    {"检查项": "样本量是否充足（≥100 行）", "结果": "✅ 通过" if row_ok else "⚠️ 样本偏少", "实际值": f"{len(df)} 行"},
                    {"检查项": "整体缺失率是否较低（<5%）", "结果": "✅ 通过" if miss_rate < 0.05 else "⚠️ 缺失偏高", "实际值": f"{miss_rate * 100:.2f}%"},
                    {"检查项": "数值特征是否足够（≥2 个）", "结果": "✅ 通过" if numeric_enough else "⚠️ 特征不足", "实际值": f"{len(num_cols)} 个"},
                    {"检查项": "是否仍有重复样本", "结果": "✅ 通过" if dup_rate == 0 else "⚠️ 存在重复", "实际值": f"{dup_rate * 100:.2f}%"},
                ])

                st.dataframe(checklist, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ 下载清洗后建模数据集（CSV）",
                    data=df_to_csv_bytes(df),
                    file_name=f"model_ready_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="t3_dl_model",
                    use_container_width=True,
                )

                st.caption("提示：深度学习通常还需要「类别特征独热编码 / 文本特征向量化 / 划分训练集与测试集」，"
                           "可以先完成标准化，再进入建模环节。")

    # ========== 右侧：智能报告区 ==========
    with right3:

        extra_t3 = ""
        if df is not None and len(df) > 0:
            try:
                num_cols_now = numeric_columns(df)
                if num_cols_now:
                    desc = df[num_cols_now].describe().T[["mean", "std", "min", "50%", "max"]].round(4)
                    desc["偏度"] = df[num_cols_now].skew().round(4)
                    extra_t3 = (
                        "科研预处理视角的关键统计：\n" + desc.to_string()
                        + f"\n\n整体缺失率：{df.isna().mean().mean() * 100:.3f}%；"
                        f"数值特征数：{len(num_cols_now)}；样本量：{len(df)}"
                    )
            except Exception:  # noqa: BLE001
                extra_t3 = ""

        render_ai_panel(df, tab_key="tab3", title="🤖 科研智能助手洞察", extra_context=extra_t3)


# ==============================================================================
# ===== 分区 13 =====  页脚
# ==============================================================================

st.markdown("---")

# 需求4：页脚安全与品牌声明（居中展示）
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.92rem; line-height:1.9; margin:0.5rem 0 1.2rem 0;">
        🔒 数据安全声明：原始数据仅在本地处理，仅统计摘要发送至大模型。<br>
        Powered by Streamlit + Pandas + LLM&nbsp;|&nbsp;开发者：赵舒琪
    </div>
    """,
    unsafe_allow_html=True,
)
