"""
Portfolio Command Center v3
============================
Single source of truth architecture:
- ALL weights flow through Engine.weights() with an explicit denominator
- ALL current-vs-target comparisons flow through Engine.current_vs_target()
- Global toggle: Total Portfolio View (incl. cash) vs Invested Only View
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DashboardConfig:
    SHEET_URL: str = "https://docs.google.com/spreadsheets/d/14IGIMj9iR5qOtmYT1e6FgN8t2tdQ5M1R_-hS6rw1RQs/export?format=csv"
    DEFAULT_FX_RATE: float = 0.66
    CACHE_TTL: int = 30
    YF_PERIOD: str = "5d"

    # Strategic bucket targets (% of TOTAL portfolio incl. cash)
    STRATEGIC_TARGETS: dict = field(default_factory=lambda: {
        'Core': 55.0,
        'Growth': 25.0,
        'Tactical': 10.0,
        'Cash': 10.0,
    })

    # Names treated as high-beta for risk reporting
    HIGH_BETA: tuple = ('IBIT', 'TSLA', 'PLTR', 'ALAB', 'MU', 'VICR', 'STRC',
                        'SIVE.ST', 'VIVO', 'NET', 'VPG', 'SNDK', 'UUUU')

    ROLE_COLORS: dict = field(default_factory=lambda: {
        'Core': '#2E4053',
        'Growth': '#1a9655',
        'Tactical': '#F5B041',
        'Cash': '#95a5a6',
    })

    PALETTE: tuple = ('#2E4053', '#1a9655', '#F5B041', '#5DADE2', '#AF7AC5',
                      '#E59866', '#48C9B0', '#EC7063', '#A6ACAF', '#F7DC6F')

config = DashboardConfig()

# ============================================================================
# PAGE SETUP / CSS
# ============================================================================

def setup_page():
    st.set_page_config(
        page_title="Portfolio Command Center",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="collapsed",
    )
    st.markdown("""
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], .main, section.main > div {
            background-color: #ffffff !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th, caption {
            color: #1a1a1a !important;
            text-shadow: none !important;
        }
        h1 { font-size: 1.55rem !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
        h2 { font-size: 1.2rem  !important; font-weight: 600 !important; }
        h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

        [data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 12px !important;
            padding: 0.8rem 0.9rem !important;
        }
        [data-testid="stMetric"] * { color: #1a1a1a !important; }
        [data-testid="stMetricLabel"] { font-size: 0.74rem !important; font-weight: 600 !important; color: #555 !important; }
        [data-testid="stMetricValue"] { font-size: 1.22rem !important; font-weight: 700 !important; }
        [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

        .stButton > button, .stDownloadButton > button {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 2px solid #dee2e6 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            min-height: 42px !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #2E4053 !important;
            color: #ffffff !important;
            border-color: #2E4053 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #555 !important;
            border-radius: 8px !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            padding: 6px 12px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        }

        .stRadio label { color: #1a1a1a !important; font-weight: 500 !important; }
        [data-testid="stAlert"] { border-radius: 10px !important; }
        [data-testid="stDataFrame"] { border-radius: 10px !important; }
        [data-testid="stProgressBar"] > div { background-color: #2E4053 !important; }
        footer, #MainMenu, header { visibility: hidden !important; }
        [data-testid="stToolbar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LAYER
# ============================================================================

class DataManager:

    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def load_portfolio_data() -> pd.DataFrame:
        try:
            df = pd.read_csv(config.SHEET_URL)
            df.columns = df.columns.str.strip()
            # Normalise column names
            df = df.rename(columns={'Strategy Role': 'Strategy_Role'})

            required = ['Ticker', 'Shares', 'Avg_Cost']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            for col in ['Shares', 'Avg_Cost', 'Stop_Price', 'Stop_Loss_Price']:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', ''), errors='coerce'
                    )
            df['Shares'] = df['Shares'].fillna(0)
            df['Avg_Cost'] = df['Avg_Cost'].fillna(0)

            if 'Target_Weight' in df.columns:
                df['Target_Weight'] = pd.to_numeric(
                    df['Target_Weight'].astype(str).str.replace('%', ''), errors='coerce'
                )
                df.loc[df['Target_Weight'] > 1.0, 'Target_Weight'] /= 100

            # Strip whitespace on text columns
            for col in ['Ticker', 'Platform', 'Currency', 'Strategy_Role', 'Sector', 'Name']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().replace({'nan': None, '': None})

            df = df[df['Ticker'].notna()]
            logger.info(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            st.error(f"⚠️ Failed to load portfolio data: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_market_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, dict]:
        """Fetch prices (incl. extended hours) and convert to AUD.
        Returns (df, aud_usd_rate, fx_rates)."""
        fx_rates = {'AUD': 1.0}
        try:
            tickers = df[df['Ticker'] != 'Cash']['Ticker'].unique().tolist()
            currencies = [c for c in df['Currency'].dropna().unique().tolist() if c and c != 'AUD']
            fx_tickers = [f"AUD{c}=X" for c in currencies]
            all_tickers = tickers + fx_tickers

            with st.spinner('📡 Syncing market data…'):
                data = yf.download(all_tickers, period=config.YF_PERIOD,
                                   progress=False, prepost=True)['Close']
                if data.empty:
                    raise ValueError("No market data received")
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                filled = data.ffill()
                latest = filled.iloc[-1]
                prev = filled.iloc[-2] if len(filled) >= 2 else latest
                day_chg = ((latest / prev) - 1) * 100

            fallback = {'USD': 0.66, 'JPY': 99.0, 'SEK': 6.85, 'EUR': 0.61, 'GBP': 0.52}
            for c in currencies:
                rate = latest.get(f"AUD{c}=X", np.nan)
                if pd.notna(rate) and rate > 0:
                    fx_rates[c] = float(rate)
                else:
                    fx_rates[c] = fallback.get(c, 1.0)
                    logger.warning(f"Fallback FX for {c}: {fx_rates[c]}")

            df = df.copy()
            df['Current_Price'] = df['Ticker'].map(latest)
            df['Price_Missing'] = df['Current_Price'].isna() & (df['Ticker'] != 'Cash')
            df['Current_Price'] = df['Current_Price'].fillna(df['Avg_Cost'])
            df.loc[df['Ticker'] == 'Cash', 'Current_Price'] = 1.0
            df.loc[df['Ticker'] == 'Cash', 'Price_Missing'] = False
            df['Day_%'] = df['Ticker'].map(day_chg).fillna(0.0)
            df.loc[df['Ticker'] == 'Cash', 'Day_%'] = 0.0

            def to_aud(row, col):
                native = row[col] * row['Shares']
                curr = row.get('Currency') or 'AUD'
                rate = fx_rates.get(curr, 1.0)
                return native / rate if rate > 0 else native

            df['MV_AUD'] = df.apply(lambda r: to_aud(r, 'Current_Price'), axis=1)
            df['Cost_AUD'] = df.apply(lambda r: to_aud(r, 'Avg_Cost'), axis=1)
            df['PnL_AUD'] = df['MV_AUD'] - df['Cost_AUD']

            return df, fx_rates.get('USD', config.DEFAULT_FX_RATE), fx_rates

        except Exception as e:
            logger.error(f"Market data error: {e}")
            st.warning(f"⚠️ Using cost basis as prices: {e}")
            df = df.copy()
            df['Current_Price'] = df['Avg_Cost']
            df['Price_Missing'] = df['Ticker'] != 'Cash'
            df['Day_%'] = 0.0
            df['MV_AUD'] = df['Avg_Cost'] * df['Shares']
            df['Cost_AUD'] = df['MV_AUD']
            df['PnL_AUD'] = 0.0
            return df, config.DEFAULT_FX_RATE, fx_rates

# ============================================================================
# CANONICAL CALCULATION ENGINE  —  single source of truth
# ============================================================================

class Engine:
    """Every allocation number on every page comes from these functions."""

    @staticmethod
    def derive_region(row) -> str:
        t, c = str(row.get('Ticker', '')), row.get('Currency') or 'AUD'
        if t.endswith('.AX') or c == 'AUD':
            return 'Australia'
        if t.endswith('.T') or c == 'JPY':
            return 'Japan'
        if t.endswith('.ST') or c == 'SEK':
            return 'Sweden'
        if c == 'EUR':
            return 'Europe'
        if c == 'GBP':
            return 'UK'
        return 'United States'

    @staticmethod
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse platform-level rows into one row per ticker.
        Target_Weight uses FIRST (same target duplicated across platforms)."""
        eq = df[df['Ticker'] != 'Cash'].copy()
        if eq.empty:
            return pd.DataFrame()

        for col in ['Strategy_Role', 'Sector', 'Currency', 'Platform', 'Name']:
            if col not in eq.columns:
                eq[col] = None
        if 'Target_Weight' not in eq.columns:
            eq['Target_Weight'] = np.nan
        stop_col = 'Stop_Price' if 'Stop_Price' in eq.columns else (
            'Stop_Loss_Price' if 'Stop_Loss_Price' in eq.columns else None)
        eq['_Stop'] = eq[stop_col] if stop_col else np.nan
        eq['_NativeCost'] = eq['Shares'] * eq['Avg_Cost']
        eq['_PriceMissing'] = eq.get('Price_Missing', False)
        if 'Day_%' not in eq.columns:
            eq['Day_%'] = 0.0

        agg = eq.groupby('Ticker').agg(
            Name=('Name', 'first'),
            Role=('Strategy_Role', 'first'),
            Sector=('Sector', 'first'),
            Currency=('Currency', 'first'),
            Platforms=('Platform', lambda x: ' + '.join(sorted(set(str(v) for v in x if v)))),
            Shares=('Shares', 'sum'),
            NativeCost=('_NativeCost', 'sum'),
            Current_Price=('Current_Price', 'mean'),
            MV_AUD=('MV_AUD', 'sum'),
            Cost_AUD=('Cost_AUD', 'sum'),
            PnL_AUD=('PnL_AUD', 'sum'),
            Target_Weight=('Target_Weight', 'first'),
            Stop=('_Stop', 'max'),
            Price_Missing=('_PriceMissing', 'any'),
            Day_pct=('Day_%', 'mean'),
        ).reset_index().rename(columns={'Day_pct': 'Day_%'})

        agg['Avg_Cost_Native'] = np.where(agg['Shares'] > 0,
                                          agg['NativeCost'] / agg['Shares'], np.nan)
        agg['PnL_%'] = np.where(agg['Cost_AUD'] > 0,
                                agg['PnL_AUD'] / agg['Cost_AUD'] * 100, 0)
        agg['Region'] = agg.apply(Engine.derive_region, axis=1)
        agg['Theme'] = agg['Sector']  # theme uses Sector column
        agg['High_Beta'] = agg['Ticker'].isin(config.HIGH_BETA)
        # Distance to stop (in native price terms)
        agg['Stop_Dist_%'] = np.where(
            (agg['Stop'] > 0) & (agg['Current_Price'] > 0),
            (agg['Current_Price'] - agg['Stop']) / agg['Current_Price'] * 100,
            np.nan)
        return agg

    @staticmethod
    def totals(df: pd.DataFrame, capital: float) -> dict:
        cash = df[df['Ticker'] == 'Cash']['MV_AUD'].sum()
        equity = df[df['Ticker'] != 'Cash']['MV_AUD'].sum()
        total = cash + equity
        return {
            'cash': cash,
            'equity': equity,
            'total': total,
            'capital': capital,
            'total_pnl': total - capital,
            'total_pnl_pct': (total - capital) / capital * 100 if capital else 0,
            'stock_pnl': df[df['Ticker'] != 'Cash']['PnL_AUD'].sum(),
            'cash_pct_total': cash / total * 100 if total else 0,
        }

    @staticmethod
    def weights(agg: pd.DataFrame, denominator: float, label: str) -> pd.DataFrame:
        """THE weight function. denominator is explicit; label documents it."""
        out = agg.copy()
        out['Weight_%'] = np.where(denominator > 0,
                                   out['MV_AUD'] / denominator * 100, 0)
        out.attrs['denominator'] = denominator
        out.attrs['denominator_label'] = label
        return out

    @staticmethod
    def group_weights(agg_w: pd.DataFrame, by: str,
                      cash_value: float = 0.0,
                      include_cash_row: bool = False) -> pd.DataFrame:
        """Group ticker-level weighted table by a dimension. Weights stay
        consistent because they were computed against one denominator."""
        g = agg_w.groupby(agg_w[by].fillna('⚠️ Unassigned')).agg(
            MV_AUD=('MV_AUD', 'sum'),
            Weight_pct=('Weight_%', 'sum'),
            Positions=('Ticker', 'count'),
        ).reset_index().rename(columns={by: 'Group', 'Weight_pct': 'Weight_%'})
        if include_cash_row and cash_value > 0:
            denom = agg_w.attrs.get('denominator', 0)
            cash_w = cash_value / denom * 100 if denom else 0
            g = pd.concat([g, pd.DataFrame([{
                'Group': 'Cash', 'MV_AUD': cash_value,
                'Weight_%': cash_w, 'Positions': 1}])], ignore_index=True)
        return g.sort_values('MV_AUD', ascending=False).reset_index(drop=True)

    @staticmethod
    def current_vs_target(agg_w: pd.DataFrame, cash_value: float) -> pd.DataFrame:
        """THE current-vs-target function (position level).
        Target_% is interpreted against the SAME denominator as Weight_%."""
        denom = agg_w.attrs.get('denominator', 0)
        cvt = agg_w.copy()
        cvt['Target_%'] = cvt['Target_Weight'] * 100
        cvt['Drift_%'] = cvt['Weight_%'] - cvt['Target_%']
        cvt['Rebalance_AUD'] = -(cvt['Drift_%'] / 100 * denom)  # + = buy, − = sell
        # Native currency amount for execution
        cvt['Rebalance_Native'] = cvt.apply(
            lambda r: r['Rebalance_AUD'] * st.session_state.get('_fx_rates', {}).get(r['Currency'] or 'AUD', 1.0),
            axis=1)
        cvt.attrs['denominator'] = denom
        cvt.attrs['denominator_label'] = agg_w.attrs.get('denominator_label', '')
        return cvt

    @staticmethod
    def role_rollup(cvt: pd.DataFrame, cash_value: float,
                    denominator: float, include_cash: bool) -> pd.DataFrame:
        """Bucket-level current vs strategic target, SAME denominator."""
        roll = cvt.groupby(cvt['Role'].fillna('⚠️ Unassigned')).agg(
            MV_AUD=('MV_AUD', 'sum'),
            Current_pct=('Weight_%', 'sum'),
            Position_Target_pct=('Target_%', 'sum'),
            Positions=('Ticker', 'count'),
        ).reset_index().rename(columns={'Role': 'Bucket',
                                        'Current_pct': 'Current_%',
                                        'Position_Target_pct': 'Position_Targets_%'})
        if include_cash:
            cash_w = cash_value / denominator * 100 if denominator else 0
            roll = pd.concat([roll, pd.DataFrame([{
                'Bucket': 'Cash', 'MV_AUD': cash_value, 'Current_%': cash_w,
                'Position_Targets_%': np.nan, 'Positions': 1}])], ignore_index=True)

        # Strategic targets only meaningful vs total portfolio.
        roll['Strategic_%'] = roll['Bucket'].map(config.STRATEGIC_TARGETS)
        roll['Gap_%'] = roll['Current_%'] - roll['Strategic_%']
        roll['Gap_AUD'] = roll['Gap_%'] / 100 * denominator
        roll['Dry_Powder_%'] = roll['Strategic_%'] - roll['Position_Targets_%']
        roll['Dry_Powder_AUD'] = roll['Dry_Powder_%'] / 100 * denominator
        return roll

# ============================================================================
# CHARTS
# ============================================================================

class Charts:

    @staticmethod
    def pie(g: pd.DataFrame, title: str, denom_label: str,
            color_map: Optional[dict] = None) -> go.Figure:
        """Pie chart driven by the SAME Weight_% used everywhere else.
        Slice values are AUD; hover shows the canonical weight."""
        fig = px.pie(
            g, values='MV_AUD', names='Group', hole=0.5,
            color='Group',
            color_discrete_map=color_map or {},
            color_discrete_sequence=list(config.PALETTE),
        )
        # Use canonical weights in labels (NOT plotly's auto-normalised %)
        labels = [f"{row.Group}<br>{row._2:.1f}%" if False else
                  f"{row.Group}: {row.Weight_pct:.1f}%"
                  for row in g.rename(columns={'Weight_%': 'Weight_pct'}).itertuples()]
        fig.update_traces(
            text=labels, textinfo='text', textposition='inside',
            textfont_size=11,
            marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)),
            hovertemplate='%{label}<br>$%{value:,.0f}<extra></extra>',
        )
        fig.update_layout(
            margin=dict(t=44, b=10, l=0, r=0), height=380, showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5,
                        xanchor="left", x=1.02, font=dict(size=11)),
            title=dict(text=f"{title}<br><sup>{denom_label}</sup>",
                       x=0.5, xanchor='center',
                       font=dict(size=15, color='#2E4053')),
        )
        return fig

    @staticmethod
    def treemap(cvt: pd.DataFrame, cash_value: float,
                include_cash: bool) -> go.Figure:
        """Portfolio map: box size = value, colour = P&L%. Role → Ticker."""
        d = cvt[['Ticker', 'Role', 'MV_AUD', 'PnL_%', 'Weight_%']].copy()
        d['Role'] = d['Role'].fillna('Unassigned')
        if include_cash and cash_value > 0:
            denom = cvt.attrs.get('denominator', 0)
            d = pd.concat([d, pd.DataFrame([{
                'Ticker': 'Cash', 'Role': 'Cash', 'MV_AUD': cash_value,
                'PnL_%': 0.0,
                'Weight_%': cash_value / denom * 100 if denom else 0}])],
                ignore_index=True)
        fig = px.treemap(
            d, path=[px.Constant('Portfolio'), 'Role', 'Ticker'],
            values='MV_AUD', color='PnL_%',
            color_continuous_scale=['#c0392b', '#f5f5f5', '#1a9655'],
            color_continuous_midpoint=0, range_color=[-40, 40],
            custom_data=['Weight_%', 'PnL_%'],
        )
        fig.update_traces(
            textinfo='label+percent parent',
            hovertemplate=('<b>%{label}</b><br>$%{value:,.0f}<br>'
                           'Weight %{customdata[0]:.1f}%<br>'
                           'P&L %{customdata[1]:+.1f}%<extra></extra>'),
            marker=dict(line=dict(color='#ffffff', width=1.5)),
        )
        fig.update_layout(
            margin=dict(t=30, b=10, l=0, r=0), height=430,
            coloraxis_colorbar=dict(title='P&L %'),
        )
        return fig

    @staticmethod
    def drift_bar(cvt: pd.DataFrame) -> go.Figure:
        d = cvt[cvt['Target_%'].notna()].sort_values('Drift_%')
        colors = ['#dc3545' if v > 0 else '#1a9655' for v in d['Drift_%']]
        fig = go.Figure(go.Bar(
            x=d['Drift_%'], y=d['Ticker'], orientation='h',
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in d['Drift_%']],
            textposition='outside', textfont_size=10,
            hovertemplate='%{y}: %{x:+.2f}%<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text="Drift vs Target (red = overweight → trim, green = underweight → add)",
                       font=dict(size=13, color='#2E4053')),
            height=max(300, 24 * len(d) + 80),
            margin=dict(t=50, b=20, l=10, r=40),
            xaxis_title="Drift %", yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor='#ffffff',
        )
        fig.add_vline(x=0, line_color='#666', line_width=1)
        return fig

# ============================================================================
# RENDER SECTIONS
# ============================================================================

CARD = ("background:#f8f9fa;border:2px solid {b};border-radius:12px;"
        "padding:13px 15px;margin:6px 0;")
RNAME = ("font-size:0.82rem;font-weight:700;color:{c};text-transform:uppercase;"
         "letter-spacing:0.5px;margin-bottom:6px;")
MROW = "display:flex;justify-content:space-between;align-items:center;margin:3px 0;"
LBL = "font-size:0.76rem;color:#666;font-weight:600;"
VAL = "font-size:1.0rem;font-weight:700;color:#1a1a1a;"
G_UP = "font-size:0.82rem;font-weight:600;color:#1a9655;"
G_DN = "font-size:0.82rem;font-weight:600;color:#dc3545;"


def market_session() -> Tuple[str, str]:
    ny = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-5)))
    h, m, wd = ny.hour, ny.minute, ny.weekday()
    if wd >= 5:
        return "🌙 Weekend (US Closed)", "#6c757d"
    if h < 4 or h >= 20:
        return "🌙 Overnight (US Closed)", "#6c757d"
    if h < 9 or (h == 9 and m < 30):
        return "🌅 US Pre-Market", "#f39c12"
    if h < 16:
        return "🟢 US Market Open", "#1a9655"
    return "🌆 US Post-Market", "#3498db"


def render_kpi_header(t: dict, agg_w: pd.DataFrame, fx_rates: dict):
    st.title("📊 Portfolio Command Center")
    session, s_color = market_session()
    fx_txt = ' · '.join(f"{c} {r:,.2f}" if c == 'JPY' else f"{c} {r:.4f}"
                        for c, r in fx_rates.items() if c != 'AUD')
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'<span style="color:#666;font-size:0.84rem;">Updated {datetime.now().strftime("%d %b %Y %H:%M")}'
        f' · AUD → {fx_txt}</span>'
        f'<span style="color:{s_color};font-size:0.84rem;font-weight:600;">{session}</span></div>',
        unsafe_allow_html=True)

    # Daily movers strip
    if 'Day_%' in agg_w.columns and agg_w['Day_%'].abs().sum() > 0:
        movers = agg_w[['Ticker', 'Day_%']].dropna()
        gainers = movers.nlargest(3, 'Day_%')
        losers = movers.nsmallest(3, 'Day_%')
        chips = ''.join(
            f'<span style="background:#eafaf1;color:#1a9655;border-radius:8px;'
            f'padding:2px 8px;margin-right:6px;font-size:0.78rem;font-weight:600;">'
            f'{r.Ticker} +{r._2:.1f}%</span>' for r in gainers.itertuples()) + \
            ''.join(
            f'<span style="background:#fdecea;color:#dc3545;border-radius:8px;'
            f'padding:2px 8px;margin-right:6px;font-size:0.78rem;font-weight:600;">'
            f'{r.Ticker} {r._2:.1f}%</span>' for r in losers.itertuples())
        st.markdown(f'<div style="margin-bottom:8px;">📈 Today: {chips}</div>',
                    unsafe_allow_html=True)

    largest = agg_w.nlargest(1, 'MV_AUD') if not agg_w.empty else None
    largest_txt = (f"{largest['Ticker'].iloc[0]} {largest['MV_AUD'].iloc[0]/t['total']*100:.1f}%"
                   if largest is not None and not largest.empty and t['total'] else "—")

    top3_pct = agg_w.nlargest(3, 'MV_AUD')['MV_AUD'].sum() / t['total'] * 100 if t['total'] else 0
    cash_pct = t['cash_pct_total']
    if top3_pct > 45 or cash_pct < 5:
        risk_status, risk_delta = "🔴 Elevated", f"Top3 {top3_pct:.0f}% · Cash {cash_pct:.0f}%"
    elif top3_pct > 35 or cash_pct < 8:
        risk_status, risk_delta = "🟡 Watch", f"Top3 {top3_pct:.0f}% · Cash {cash_pct:.0f}%"
    else:
        risk_status, risk_delta = "🟢 Normal", f"Top3 {top3_pct:.0f}% · Cash {cash_pct:.0f}%"

    r1 = st.columns(3)
    r1[0].metric("Total Value (AUD)", f"${t['total']:,.0f}",
                 f"{t['total_pnl_pct']:+.2f}% vs capital")
    r1[1].metric("Lifetime P&L", f"${t['total_pnl']:,.0f}",
                 f"capital ${t['capital']:,.0f}")
    r1[2].metric("Risk Status", risk_status, risk_delta, delta_color="off")

    r2 = st.columns(3)
    r2[0].metric("Cash", f"${t['cash']:,.0f}", f"{cash_pct:.1f}% of total")
    r2[1].metric("Invested", f"${t['equity']:,.0f}",
                 f"{100-cash_pct:.1f}% of total")
    r2[2].metric("Positions", f"{len(agg_w)}", f"largest: {largest_txt}", delta_color="off")


def render_allocation(agg_w: pd.DataFrame, t: dict, denom_label: str, include_cash: bool):
    st.subheader("🥧 Allocation")
    st.caption(f"**Denominator: {denom_label}.** All percentages below use this same base "
               f"and therefore reconcile with the Current vs Target section.")

    dim = st.radio("Dimension",
                   ["Holdings", "Strategy Role", "Sector / Theme",
                    "Currency", "Platform", "Region"],
                   horizontal=True, label_visibility="collapsed")

    if dim == "Holdings":
        # Per-stock pie: every position is its own slice, small ones grouped
        h = agg_w[['Ticker', 'MV_AUD', 'Weight_%']].copy()
        big = h[h['Weight_%'] >= 1.0]
        small = h[h['Weight_%'] < 1.0]
        g = big.rename(columns={'Ticker': 'Group'})
        g['Positions'] = 1
        if not small.empty:
            g = pd.concat([g, pd.DataFrame([{
                'Group': f"Other ({len(small)})",
                'MV_AUD': small['MV_AUD'].sum(),
                'Weight_%': small['Weight_%'].sum(),
                'Positions': len(small)}])], ignore_index=True)
        if include_cash and t['cash'] > 0:
            denom = agg_w.attrs.get('denominator', 0)
            g = pd.concat([g, pd.DataFrame([{
                'Group': 'Cash', 'MV_AUD': t['cash'],
                'Weight_%': t['cash'] / denom * 100 if denom else 0,
                'Positions': 1}])], ignore_index=True)
        g = g.sort_values('MV_AUD', ascending=False).reset_index(drop=True)
        cmap = {'Cash': '#95a5a6'}
        tip = ("💡 Click **Cash** (or any name) in the legend to hide that slice. "
               "Labels keep their true weights from the selected basis — "
               "use the toggle above for the official excl-cash numbers.")
    else:
        dim_col = {"Strategy Role": "Role", "Sector / Theme": "Theme",
                   "Currency": "Currency", "Platform": "Platforms",
                   "Region": "Region"}[dim]
        g = Engine.group_weights(agg_w, dim_col,
                                 cash_value=t['cash'],
                                 include_cash_row=include_cash)
        cmap = config.ROLE_COLORS if dim_col == 'Role' else None
        tip = None

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(Charts.pie(g, dim, denom_label, cmap),
                        use_container_width=True)
        if tip:
            st.caption(tip)
    with c2:
        show = g.copy()
        show['MV_AUD'] = show['MV_AUD'].map('${:,.0f}'.format)
        show['Weight_%'] = show['Weight_%'].map('{:.1f}%'.format)
        st.dataframe(show, use_container_width=True, hide_index=True, height=340)
        st.caption(f"Sum of weights: {g['Weight_%'].sum():.1f}% "
                   f"({'includes' if include_cash else 'excludes'} cash)")

    if dim == "Holdings":
        st.markdown("##### 🗺️ Portfolio map — size = value · colour = P&L")
        st.plotly_chart(Charts.treemap(agg_w, t['cash'], include_cash),
                        use_container_width=True)


def render_current_vs_target(cvt: pd.DataFrame, roll: pd.DataFrame,
                             t: dict, denominator: float,
                             denom_label: str, include_cash: bool):
    st.subheader("🎯 Current vs Target")
    st.caption(f"**Denominator: {denom_label}.** Strategic bucket targets "
               f"(Core {config.STRATEGIC_TARGETS['Core']:.0f} / Growth {config.STRATEGIC_TARGETS['Growth']:.0f} / "
               f"Tactical {config.STRATEGIC_TARGETS['Tactical']:.0f} / Cash {config.STRATEGIC_TARGETS['Cash']:.0f}) "
               f"are defined against the total portfolio."
               + ("" if include_cash else " ⚠️ In Invested-Only view bucket gaps vs strategic targets are hidden "
                  "because those targets include cash."))

    # ── Bucket cards ──
    order = ['Core', 'Growth', 'Tactical'] + (['Cash'] if include_cash else [])
    cols = st.columns(len(order))
    for i, bucket in enumerate(order):
        row = roll[roll['Bucket'] == bucket]
        if row.empty:
            continue
        r = row.iloc[0]
        color = config.ROLE_COLORS.get(bucket, '#666')
        html = (f'<div style="{CARD.format(b=color)}">'
                f'<div style="{RNAME.format(c=color)}">{bucket}</div>'
                f'<div style="{MROW}"><span style="{LBL}">Current</span>'
                f'<span style="{VAL}">{r["Current_%"]:.1f}%</span></div>')
        if include_cash and pd.notna(r.get('Strategic_%')):
            gap = r['Gap_%']
            g_style = G_UP if gap >= 0 else G_DN
            arrow = "▲" if gap >= 0 else "▼"
            html += (f'<div style="{MROW}"><span style="{LBL}">Strategic Target</span>'
                     f'<span style="{VAL}">{r["Strategic_%"]:.0f}%</span></div>'
                     f'<div style="{MROW}"><span style="{LBL}">Gap</span>'
                     f'<span style="{g_style}">{arrow} {abs(gap):.1f}% (${abs(r["Gap_AUD"]):,.0f})</span></div>')
            if bucket != 'Cash' and pd.notna(r.get('Dry_Powder_%')) and r['Dry_Powder_AUD'] > 1000:
                html += (f'<div style="font-size:0.76rem;color:#6c757d;font-weight:600;margin-top:4px;">'
                         f'🎯 Unallocated: {r["Dry_Powder_%"]:.1f}% '
                         f'(${r["Dry_Powder_AUD"]:,.0f} dry powder)</div>')
        html += (f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #e0e0e0;">'
                 f'<span style="{LBL}">Value </span>'
                 f'<span style="font-size:0.86rem;font-weight:600;">${r["MV_AUD"]:,.0f}</span></div></div>')
        cols[i].markdown(html, unsafe_allow_html=True)

    # ── Position-level table ──
    st.markdown("##### Position drift vs target")
    d = cvt.copy()
    d = d[['Ticker', 'Role', 'Weight_%', 'Target_%', 'Drift_%',
           'Rebalance_AUD', 'Rebalance_Native', 'Currency']]
    d = d.sort_values('Drift_%', key=lambda s: s.abs(), ascending=False)
    d['Action'] = np.select(
        [d['Target_%'].isna(), d['Rebalance_AUD'] > 1000, d['Rebalance_AUD'] < -1000],
        ['⚠️ No target', '🟢 ADD', '🔴 TRIM'], default='✅ OK')
    d['Rebalance (Native)'] = d.apply(
        lambda r: f"{r['Rebalance_Native']:+,.0f} {r['Currency'] or 'AUD'}"
        if pd.notna(r['Rebalance_Native']) and pd.notna(r['Target_%']) else "—", axis=1)

    st.dataframe(
        d.drop(columns=['Rebalance_Native', 'Currency']),
        use_container_width=True, hide_index=True, height=420,
        column_config={
            'Ticker': st.column_config.TextColumn('Stock', width='small'),
            'Role': st.column_config.TextColumn('Role', width='small'),
            'Weight_%': st.column_config.NumberColumn('Current %', format='%.2f%%'),
            'Target_%': st.column_config.NumberColumn('Target %', format='%.2f%%'),
            'Drift_%': st.column_config.NumberColumn('Drift %', format='%+.2f%%'),
            'Rebalance_AUD': st.column_config.NumberColumn('Rebalance (AUD)', format='$%+,.0f'),
            'Rebalance (Native)': st.column_config.TextColumn('Rebalance (Native)'),
            'Action': st.column_config.TextColumn('Action', width='small'),
        })
    st.caption("Rebalance: + means buy that amount, − means sell. Native column is the "
               "order size in the stock's own trading currency.")

    with st.expander("📉 Drift chart"):
        st.plotly_chart(Charts.drift_bar(cvt), use_container_width=True)


def render_positions(cvt: pd.DataFrame, fx_rates: dict):
    st.subheader("📋 Positions")

    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        roles = ['All'] + sorted(cvt['Role'].dropna().unique().tolist())
        f_role = st.selectbox("Role", roles)
    with f2:
        curs = ['All'] + sorted(cvt['Currency'].dropna().unique().tolist())
        f_cur = st.selectbox("Currency", curs)
    with f3:
        search = st.text_input("Search ticker / name", "")

    d = cvt.copy()
    if f_role != 'All':
        d = d[d['Role'] == f_role]
    if f_cur != 'All':
        d = d[d['Currency'] == f_cur]
    if search:
        s = search.lower()
        d = d[d['Ticker'].str.lower().str.contains(s, na=False) |
              d['Name'].astype(str).str.lower().str.contains(s, na=False)]

    d = d.sort_values('MV_AUD', ascending=False)
    d['Note'] = np.select(
        [d['Target_%'].isna(),
         (d['Stop_Dist_%'].notna()) & (d['Stop_Dist_%'] < 5),
         d['Rebalance_AUD'] < -1000,
         d['Rebalance_AUD'] > 1000],
        ['Set target', '⚠️ Near stop', 'Trim to target', 'Add to target'],
        default='Hold')

    view = d[['Ticker', 'Role', 'Platforms', 'Shares', 'Avg_Cost_Native',
              'Current_Price', 'Day_%', 'Currency', 'MV_AUD', 'PnL_AUD',
              'PnL_%', 'Weight_%', 'Target_%', 'Drift_%', 'Stop',
              'Stop_Dist_%', 'Note']]

    styled = view.style \
        .format({'Shares': '{:,.2f}', 'Avg_Cost_Native': '{:,.2f}',
                 'Current_Price': '{:,.2f}', 'Day_%': '{:+.2f}%',
                 'MV_AUD': '${:,.0f}',
                 'PnL_AUD': '${:,.0f}', 'PnL_%': '{:+.1f}%',
                 'Weight_%': '{:.2f}%', 'Target_%': '{:.2f}%',
                 'Drift_%': '{:+.2f}%', 'Stop': '{:,.2f}',
                 'Stop_Dist_%': '{:.1f}%'}, na_rep='—') \
        .apply(lambda col: ['color:#1a9655;font-weight:600' if isinstance(v, (int, float)) and v > 0
                            else 'color:#dc3545;font-weight:600' if isinstance(v, (int, float)) and v < 0
                            else '' for v in col],
               subset=['Day_%', 'PnL_AUD', 'PnL_%', 'Drift_%'])

    st.dataframe(styled, use_container_width=True, height=480,
                 column_config={
                     'Ticker': 'Stock', 'Avg_Cost_Native': 'Avg Cost',
                     'Current_Price': 'Price', 'Day_%': 'Day %',
                     'MV_AUD': 'Mkt Val (AUD)',
                     'PnL_AUD': 'P&L (AUD)', 'PnL_%': 'P&L %',
                     'Weight_%': 'Weight %', 'Target_%': 'Target %',
                     'Drift_%': 'Drift %', 'Stop_Dist_%': 'To Stop %'})
    st.caption(f"{len(d)} positions shown · Avg Cost & Price in native currency · "
               f"Add a Stop_Price column in the sheet to populate stop levels.")


def render_execution(cvt: pd.DataFrame, roll: pd.DataFrame, include_cash: bool):
    st.subheader("⚡ Execution")
    st.caption("Everything here derives from the same drift table above — no separate math.")

    trims = cvt[cvt['Rebalance_AUD'] < -1000].sort_values('Rebalance_AUD')
    adds = cvt[(cvt['Rebalance_AUD'] > 1000) & cvt['Target_%'].notna()] \
        .sort_values('Rebalance_AUD', ascending=False)
    near_stop = cvt[(cvt['Stop_Dist_%'].notna()) & (cvt['Stop_Dist_%'] < 10)] \
        .sort_values('Stop_Dist_%')
    watch = cvt[cvt['Target_%'].notna() & (cvt['Shares'] <= 0)]
    no_target = cvt[cvt['Target_%'].isna()]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔴 Trim candidates (overweight)")
        if trims.empty:
            st.success("None — nothing overweight by >$1k")
        else:
            for _, r in trims.iterrows():
                st.markdown(f"- **{r['Ticker']}** trim **${abs(r['Rebalance_AUD']):,.0f}** "
                            f"({r['Rebalance_Native']:+,.0f} {r['Currency'] or 'AUD'}) · "
                            f"drift {r['Drift_%']:+.1f}%")
    with c2:
        st.markdown("##### 🟢 Add candidates (underweight)")
        if adds.empty:
            st.success("None — nothing underweight by >$1k")
        else:
            for _, r in adds.iterrows():
                st.markdown(f"- **{r['Ticker']}** add **${r['Rebalance_AUD']:,.0f}** "
                            f"({r['Rebalance_Native']:+,.0f} {r['Currency'] or 'AUD'}) · "
                            f"drift {r['Drift_%']:+.1f}%")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### ⚠️ Closest to stop")
        if near_stop.empty:
            st.info("No positions within 10% of a stop (or no Stop_Price column).")
        else:
            for _, r in near_stop.iterrows():
                st.markdown(f"- **{r['Ticker']}** {r['Stop_Dist_%']:.1f}% above stop "
                            f"({r['Current_Price']:,.2f} vs stop {r['Stop']:,.2f})")
    with c4:
        st.markdown("##### 👀 Watchlist / unassigned")
        if not watch.empty:
            for _, r in watch.iterrows():
                st.markdown(f"- **{r['Ticker']}** target {r['Target_%']:.1f}% — not yet bought")
        if not no_target.empty:
            names = ', '.join(no_target['Ticker'].tolist())
            st.markdown(f"- No target set: **{names}**")
        if watch.empty and no_target.empty:
            st.success("All positions have targets.")

    if include_cash:
        dry = roll[roll['Bucket'].isin(['Core', 'Growth', 'Tactical'])]
        total_dry = dry['Dry_Powder_AUD'].clip(lower=0).sum()
        if total_dry > 1000:
            st.info(f"💰 Total unallocated dry powder across buckets: **${total_dry:,.0f}** "
                    f"(strategic target space without named positions)")


def render_risk(cvt: pd.DataFrame, t: dict, denom_label: str):
    st.subheader("🛡️ Risk")
    st.caption(f"Denominator: {denom_label}")

    denom = cvt.attrs.get('denominator', t['total'])
    top10 = cvt.nlargest(10, 'MV_AUD')
    top10_pct = top10['MV_AUD'].sum() / denom * 100 if denom else 0
    hb = cvt[cvt['High_Beta']]
    hb_pct = hb['MV_AUD'].sum() / denom * 100 if denom else 0
    jp = cvt[cvt['Currency'] == 'JPY']
    jp_pct = jp['MV_AUD'].sum() / denom * 100 if denom else 0
    cash_pct = t['cash_pct_total']

    m = st.columns(4)
    m[0].metric("Top 10 concentration", f"{top10_pct:.1f}%",
                f"${top10['MV_AUD'].sum():,.0f}", delta_color="off")
    m[1].metric("High-beta exposure", f"{hb_pct:.1f}%",
                f"{len(hb)} names", delta_color="off")
    m[2].metric("🇯🇵 Japan components basket", f"{jp_pct:.1f}%",
                f"${jp['MV_AUD'].sum():,.0f}", delta_color="off")
    cash_state = ("🟢 within band" if 5 <= cash_pct <= 15 else
                  ("🔴 below floor" if cash_pct < 5 else "🟡 above band"))
    m[3].metric("Cash buffer", f"{cash_pct:.1f}%", cash_state, delta_color="off")

    # Second row: concentration statistics (computed on invested capital)
    eq_total = cvt['MV_AUD'].sum()
    if eq_total > 0:
        shares_of_eq = cvt['MV_AUD'] / eq_total
        hhi = float((shares_of_eq ** 2).sum())
        eff_n = 1 / hhi if hhi > 0 else 0
        usd_pct = cvt[cvt['Currency'] == 'USD']['MV_AUD'].sum() / denom * 100 if denom else 0
        hhi_state = ("🟢 Diversified" if hhi < 0.10 else
                     "🟡 Moderate" if hhi < 0.18 else "🔴 Concentrated")
        m2 = st.columns(4)
        m2[0].metric("HHI (invested)", f"{hhi:.3f}", hhi_state, delta_color="off")
        m2[1].metric("Effective # positions", f"{eff_n:.1f}",
                     f"of {len(cvt)} held", delta_color="off")
        m2[2].metric("USD exposure", f"{usd_pct:.1f}%",
                     "FX risk vs AUD", delta_color="off")
        day_pnl = (cvt['Day_%'] / 100 * cvt['MV_AUD']).sum() if 'Day_%' in cvt.columns else 0
        m2[3].metric("Today's move (est.)", f"${day_pnl:+,.0f}",
                     "sum of position day changes", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Top 10 positions")
        show = top10[['Ticker', 'Role', 'MV_AUD', 'Weight_%']].copy()
        show['MV_AUD'] = show['MV_AUD'].map('${:,.0f}'.format)
        show['Weight_%'] = show['Weight_%'].map('{:.1f}%'.format)
        st.dataframe(show, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("##### Theme concentration")
        theme = Engine.group_weights(cvt, 'Theme')
        show = theme[['Group', 'MV_AUD', 'Weight_%', 'Positions']].copy()
        show['MV_AUD'] = show['MV_AUD'].map('${:,.0f}'.format)
        show['Weight_%'] = show['Weight_%'].map('{:.1f}%'.format)
        st.dataframe(show, hide_index=True, use_container_width=True)
        if not hb.empty:
            st.caption("High-beta names: " + ', '.join(hb['Ticker'].tolist()))


def render_data_quality(raw: pd.DataFrame, cvt: pd.DataFrame, t: dict):
    st.subheader("🔍 Data Quality")
    issues = 0

    checks = []
    missing_price = cvt[cvt['Price_Missing']]
    checks.append(("Missing live prices (using cost basis)",
                   missing_price['Ticker'].tolist()))
    checks.append(("Missing target weights",
                   cvt[cvt['Target_Weight'].isna()]['Ticker'].tolist()))
    checks.append(("Missing sector / theme",
                   cvt[cvt['Sector'].isna()]['Ticker'].tolist()))
    checks.append(("Missing strategy role",
                   cvt[cvt['Role'].isna()]['Ticker'].tolist()))

    eq = raw[raw['Ticker'] != 'Cash']
    dupes = eq.groupby(['Ticker', 'Platform']).size()
    dupe_list = [f"{tk} ({pf})" for (tk, pf), n in dupes.items() if n > 1]
    checks.append(("Duplicate ticker+platform rows", dupe_list))

    for label, items in checks:
        if items:
            issues += 1
            st.warning(f"**{label}:** {', '.join(items)}")
        else:
            st.success(f"**{label}:** none ✓")

    # Reconciliation: parts must sum to the whole
    parts = cvt['MV_AUD'].sum() + t['cash']
    diff = abs(parts - t['total'])
    if diff < 1:
        st.success(f"**Reconciliation:** positions (${cvt['MV_AUD'].sum():,.0f}) + "
                   f"cash (${t['cash']:,.0f}) = total (${t['total']:,.0f}) ✓")
    else:
        issues += 1
        st.error(f"**Reconciliation mismatch:** parts sum to ${parts:,.0f} but "
                 f"total is ${t['total']:,.0f} (diff ${diff:,.0f})")

    tgt_sum = cvt['Target_Weight'].fillna(0).sum() * 100
    strat_sum = sum(v for k, v in config.STRATEGIC_TARGETS.items() if k != 'Cash')
    st.info(f"Position targets sum to **{tgt_sum:.1f}%** vs strategic stock buckets "
            f"**{strat_sum:.0f}%** (+{config.STRATEGIC_TARGETS['Cash']:.0f}% cash). "
            f"Unallocated target space: **{strat_sum - tgt_sum:.1f}%**")

    if issues == 0:
        st.balloons()


def render_download(cvt: pd.DataFrame, t: dict):
    st.subheader("📥 Export")
    out = cvt[['Ticker', 'Role', 'Sector', 'Currency', 'Platforms', 'Shares',
               'Avg_Cost_Native', 'Current_Price', 'Cost_AUD', 'MV_AUD',
               'PnL_AUD', 'PnL_%', 'Weight_%', 'Target_%', 'Drift_%',
               'Rebalance_AUD']].copy()
    out = out.sort_values('MV_AUD', ascending=False)
    cash_row = pd.DataFrame([{'Ticker': 'Cash', 'MV_AUD': t['cash'],
                              'Cost_AUD': t['cash']}])
    total_row = pd.DataFrame([{'Ticker': 'TOTAL', 'MV_AUD': t['total'],
                               'Cost_AUD': out['Cost_AUD'].sum() + t['cash'],
                               'PnL_AUD': t['total_pnl']}])
    out = pd.concat([out, cash_row, total_row], ignore_index=True)
    csv = out.to_csv(index=False).encode('utf-8')

    c1, c2 = st.columns(2)
    c1.download_button("📊 Download Portfolio CSV", csv,
                       f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       "text/csv", use_container_width=True)
    if c2.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# MAIN
# ============================================================================

def main():
    setup_page()

    raw = DataManager.load_portfolio_data()
    if raw.empty:
        st.error("❌ No portfolio data available")
        st.stop()

    capital_row = raw[raw['Ticker'] == 'CAPITAL']
    capital = capital_row['Shares'].sum() if not capital_row.empty else 0
    if capital == 0 and not capital_row.empty:
        capital = capital_row['Avg_Cost'].sum()
    df = raw[raw['Ticker'] != 'CAPITAL'].copy()

    df, fx_rate, fx_rates = DataManager.fetch_market_data(df)
    st.session_state['_fx_rates'] = fx_rates

    # ── Canonical pipeline: aggregate → totals → weights → CvT → rollup ──
    agg = Engine.aggregate(df)
    t = Engine.totals(df, capital)

    # Global denominator toggle
    view = st.radio("Allocation basis",
                    ["💼 Total Portfolio View (incl. cash)",
                     "📈 Invested Only View (excl. cash)"],
                    horizontal=True)
    include_cash = "Total" in view
    denominator = t['total'] if include_cash else t['equity']
    denom_label = ("Total portfolio incl. cash" if include_cash
                   else "Invested capital only, cash excluded")

    agg_w = Engine.weights(agg, denominator, denom_label)
    cvt = Engine.current_vs_target(agg_w, t['cash'])
    roll = Engine.role_rollup(cvt, t['cash'], denominator, include_cash)

    render_kpi_header(t, agg_w, fx_rates)
    st.markdown("---")

    tabs = st.tabs(["📊 Overview", "🥧 Allocation", "📋 Positions",
                    "⚡ Execution", "🛡️ Risk", "🔍 Data Quality"])

    with tabs[0]:
        render_current_vs_target(cvt, roll, t, denominator, denom_label, include_cash)
        st.markdown("---")
        render_download(cvt, t)
    with tabs[1]:
        render_allocation(agg_w, t, denom_label, include_cash)
    with tabs[2]:
        render_positions(cvt, fx_rates)
    with tabs[3]:
        render_execution(cvt, roll, include_cash)
    with tabs[4]:
        render_risk(cvt, t, denom_label)
    with tabs[5]:
        render_data_quality(df, cvt, t)


if __name__ == "__main__":
    main()
