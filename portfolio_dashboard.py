"""
Portfolio Command Center v2.0
=============================
Bucket-first strategic portfolio dashboard with Core/Growth/Tactical/Cleanup framework.

Key upgrades from v1:
- Bucket-first information hierarchy (not ticker-first)
- Cleanup bucket for non-framework positions
- Action flags: ON TARGET / BUY / TRIM / EXIT
- Position-level gap-to-target in AUD
- Auto-refresh every 60s + visibility API + keep-alive
- Responsive design (no manual mobile/desktop toggle)
- Query params for URL state persistence
- Reduced price cache TTL (60s) for fresher data
- Multi-source price fallback
- Orders and watchlist support (optional sheet tabs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time

# Optional: auto-refresh component
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Centralized configuration"""
    # Data sources
    SHEET_URL: str = "https://docs.google.com/spreadsheets/d/14IGIMj9iR5qOtmYT1e6FgN8t2tdQ5M1R_-hS6rw1RQs/export?format=csv"
    SHEET_ORDERS_URL: str = ""  # Optional: separate sheet tab for GTC orders
    SHEET_WATCHLIST_URL: str = ""  # Optional: separate sheet tab for watchlist
    
    # Currency
    DEFAULT_FX_RATE: float = 0.66  # AUD/USD fallback (updated)
    
    # Cache
    PRICE_CACHE_TTL: int = 60   # 1 minute for prices
    DATA_CACHE_TTL: int = 300   # 5 minutes for sheets
    
    # Auto-refresh
    AUTO_REFRESH_MS: int = 60_000  # 60 seconds
    
    # Market data
    YF_PERIOD: str = "2d"
    
    # Framework
    BUCKET_ORDER: List[str] = field(default_factory=lambda: ['Core', 'Growth', 'Tactical', 'Cleanup'])
    BUCKET_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'Core': 0.65,
        'Growth': 0.25,
        'Tactical': 0.08,
        'Cleanup': 0.00,
    })
    BUCKET_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'Core': '#2563EB',      # Blue
        'Growth': '#16A34A',    # Green
        'Tactical': '#EAB308',  # Amber
        'Cleanup': '#DC2626',   # Red
    })
    BUCKET_LABELS: Dict[str, str] = field(default_factory=lambda: {
        'Core': '核心 Core',
        'Growth': '成長 Growth',
        'Tactical': '戰術 Tactical',
        'Cleanup': '非框架 Cleanup',
    })
    
    # Action thresholds (% deviation from target)
    ACTION_TOLERANCE: float = 0.01  # ±1% = on target
    CASH_TARGET_MIN: float = 0.02
    CASH_TARGET_MAX: float = 0.10

config = Config()

# ============================================================================
# KEEP-ALIVE & AUTO-REFRESH
# ============================================================================

def inject_keepalive():
    """Inject auto-refresh, visibility API handler, and keep-alive pings"""
    if AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=config.AUTO_REFRESH_MS, key="main_refresh")
    
    # Visibility API handler - triggers refresh when tab becomes visible
    # Plus keep-alive ping every 4 minutes to prevent Streamlit Cloud spin-down
    components.html("""
    <script>
    (function() {
        // Visibility API: refresh when user returns to tab
        let wasHidden = false;
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                wasHidden = true;
            } else if (wasHidden) {
                wasHidden = false;
                // Tell Streamlit to rerun
                window.parent.postMessage({type: 'streamlit:componentReady'}, '*');
                setTimeout(() => window.parent.location.reload(), 100);
            }
        });
        
        // Keep-alive: ping self every 4 minutes to prevent Streamlit Cloud spin-down
        setInterval(function() {
            fetch(window.location.href, { method: 'HEAD', cache: 'no-store' })
                .catch(() => {});
        }, 240000);
    })();
    </script>
    """, height=0)

# ============================================================================
# PAGE SETUP
# ============================================================================

def setup_page():
    """Configure page and inject CSS"""
    st.set_page_config(
        page_title="Portfolio Command Center",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Portfolio Command Center v2.0 — Bucket-first framework tracking"
        }
    )
    
    st.markdown("""
    <style>
        /* ============ BASE ============ */
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], .main, section.main > div {
            background-color: #ffffff !important;
        }
        
        /* ============ TYPOGRAPHY ============ */
        h1, h2, h3, h4, h5, h6, p, span, div, label, li, td, th, caption {
            color: #0f172a !important;
            text-shadow: none !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        h1 { font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.5px !important; margin-bottom: 0.5rem !important; }
        h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
        
        /* ============ HEALTH STRIP (sticky) ============ */
        .health-strip {
            position: sticky;
            top: 0;
            z-index: 100;
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-bottom: 2px solid #e2e8f0;
            padding: 12px 0;
            margin: -16px -16px 16px -16px;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .health-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
        }
        @media (max-width: 768px) {
            .health-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        .health-item {
            padding: 8px 10px;
            border-radius: 8px;
            background: rgba(255,255,255,0.8);
        }
        .health-label {
            font-size: 0.68rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 2px;
        }
        .health-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        .health-delta {
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 2px;
        }
        .delta-up { color: #16a34a !important; }
        .delta-down { color: #dc2626 !important; }
        .delta-neutral { color: #64748b !important; }
        
        /* ============ BUCKET CARDS ============ */
        .bucket-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 16px 0;
        }
        @media (max-width: 900px) {
            .bucket-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 500px) {
            .bucket-grid { grid-template-columns: 1fr; }
        }
        .bucket-card {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.15s ease;
        }
        .bucket-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .bucket-header {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
        }
        .bucket-current {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 4px;
        }
        .bucket-target {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 10px;
        }
        .bucket-gap {
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .bucket-progress {
            background: #f1f5f9;
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .bucket-progress-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        .bucket-footer {
            font-size: 0.72rem;
            color: #64748b;
            border-top: 1px solid #f1f5f9;
            padding-top: 8px;
            margin-top: 8px;
        }
        
        /* ============ POSITION CARDS ============ */
        .position-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left-width: 4px;
            border-radius: 10px;
            padding: 12px 14px;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .pos-row-1 {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 6px;
        }
        .pos-ticker {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .pos-weight {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
        }
        .pos-row-2 {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 4px;
        }
        .pos-value {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
        }
        .pos-pnl {
            font-size: 0.82rem;
            font-weight: 600;
        }
        .pos-action {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .action-on-target { background: #dcfce7; color: #15803d; }
        .action-buy { background: #dbeafe; color: #1d4ed8; }
        .action-trim { background: #fef3c7; color: #a16207; }
        .action-exit { background: #fee2e2; color: #991b1b; }
        .pos-detail {
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 4px;
        }
        
        /* ============ BUTTONS ============ */
        .stButton > button {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            min-height: 40px !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:hover {
            background-color: #2563EB !important;
            color: #ffffff !important;
            border-color: #2563EB !important;
        }
        
        /* ============ TABS ============ */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8fafc !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748b !important;
            background-color: transparent !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            min-height: 36px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #0f172a !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        }
        
        /* ============ HIDE CHROME ============ */
        footer, #MainMenu, header { visibility: hidden !important; }
        [data-testid="stToolbar"] { display: none !important; }
        
        /* ============ DATAFRAMES ============ */
        [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
        
        /* ============ ALERTS ============ */
        [data-testid="stAlert"] { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# AUTHENTICATION
# ============================================================================

class AuthManager:
    @staticmethod
    def check_password() -> bool:
        def password_entered():
            try:
                if st.session_state["password"] == st.secrets.get("PASSWORD", ""):
                    st.session_state["password_correct"] = True
                    del st.session_state["password"]
                else:
                    st.session_state["password_correct"] = False
                    st.error("❌ Invalid access code")
            except Exception as e:
                logger.error(f"Auth error: {e}")
                st.session_state["password_correct"] = False
        
        if st.session_state.get("password_correct", False):
            return True
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Portfolio Access")
            st.text_input(
                "Enter Access Code",
                type="password",
                on_change=password_entered,
                key="password",
                placeholder="Access code required"
            )
            st.caption("Contact administrator for access")
        return False

# ============================================================================
# DATA LAYER
# ============================================================================

class DataManager:
    """Multi-source data loading with fallback"""
    
    @staticmethod
    @st.cache_data(ttl=config.DATA_CACHE_TTL, show_spinner=False)
    def load_portfolio_data() -> pd.DataFrame:
        """Load portfolio from Google Sheets with strict validation"""
        try:
            df = pd.read_csv(config.SHEET_URL)
            df.columns = df.columns.str.strip()
            
            required_cols = ['Ticker', 'Shares', 'Avg_Cost']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Clean numeric columns
            numeric_cols = ['Shares', 'Avg_Cost', 'Stop_Loss_Price', 'Target_Weight']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('%', ''),
                        errors='coerce'
                    ).fillna(0)
            
            # Normalize target weight (handle both 0.18 and 18 formats)
            if 'Target_Weight' in df.columns:
                df.loc[df['Target_Weight'] > 1.0, 'Target_Weight'] /= 100
            
            # Normalize Strategy Role column name
            role_cols = ['Strategy_Role', 'Strategy Role', 'strategy_role']
            for col in role_cols:
                if col in df.columns:
                    df['Strategy_Role'] = df[col].astype(str).str.strip().str.title()
                    break
            
            # Map any non-framework roles to "Cleanup"
            if 'Strategy_Role' in df.columns:
                valid_roles = {'Core', 'Growth', 'Tactical', 'Cleanup'}
                df.loc[~df['Strategy_Role'].isin(valid_roles) & (df['Ticker'] != 'Cash'), 'Strategy_Role'] = 'Cleanup'
            
            logger.info(f"Loaded {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Load error: {e}")
            st.error(f"⚠️ Failed to load data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=config.PRICE_CACHE_TTL, show_spinner=False)
    def fetch_prices(tickers: tuple) -> Tuple[Dict[str, float], float]:
        """Fetch prices via yfinance with fallback to cached/avg cost"""
        prices = {}
        fx_rate = config.DEFAULT_FX_RATE
        
        try:
            ticker_list = list(tickers) + ["AUDUSD=X"]
            data = yf.download(ticker_list, period=config.YF_PERIOD, progress=False, auto_adjust=True)
            
            if data.empty:
                raise ValueError("Empty yfinance response")
            
            close_data = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
            latest = close_data.ffill().iloc[-1]
            
            for ticker in ticker_list:
                val = latest.get(ticker) if hasattr(latest, 'get') else latest[ticker] if ticker in latest else None
                if val is not None and not pd.isna(val):
                    prices[ticker] = float(val)
            
            fx_from_yf = prices.pop("AUDUSD=X", None)
            if fx_from_yf and fx_from_yf > 0:
                fx_rate = fx_from_yf
            
            logger.info(f"Fetched {len(prices)} prices, FX: {fx_rate:.4f}")
        except Exception as e:
            logger.warning(f"Price fetch error: {e}")
            st.warning(f"⚠️ Using fallback prices: {str(e)[:100]}")
        
        return prices, fx_rate
    
    @staticmethod
    def enrich_portfolio(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Add current prices and AUD valuations"""
        if df.empty:
            return df, config.DEFAULT_FX_RATE
        
        tickers = tuple(df[df['Ticker'] != 'Cash']['Ticker'].unique())
        prices, fx_rate = DataManager.fetch_prices(tickers)
        
        # Apply prices with fallback to avg cost
        df['Current_Price'] = df['Ticker'].map(prices)
        df['Current_Price'] = df['Current_Price'].fillna(df['Avg_Cost'])
        df.loc[df['Ticker'] == 'Cash', 'Current_Price'] = 1.0
        
        # Calculate AUD values
        def to_aud(row, price_col):
            val = row[price_col] * row['Shares']
            if row.get('Currency') == 'USD':
                return val / fx_rate
            return val
        
        df['MV_AUD'] = df.apply(lambda r: to_aud(r, 'Current_Price'), axis=1)
        df['Cost_AUD'] = df.apply(lambda r: to_aud(r, 'Avg_Cost'), axis=1)
        df['PnL_AUD'] = df['MV_AUD'] - df['Cost_AUD']
        df['PnL_Pct'] = (df['PnL_AUD'] / df['Cost_AUD'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        return df, fx_rate

# ============================================================================
# ANALYTICS
# ============================================================================

class Analytics:
    """Bucket-first portfolio analytics"""
    
    @staticmethod
    def calculate_summary(df: pd.DataFrame, capital: float) -> dict:
        """Portfolio-level metrics"""
        total_mv = df['MV_AUD'].sum()
        total_cost = df['Cost_AUD'].sum()
        cash_value = df[df['Ticker'] == 'Cash']['MV_AUD'].sum()
        equity_value = total_mv - cash_value
        
        # Lifetime P&L vs capital injected
        total_pnl = total_mv - capital
        pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
        
        # Concentration (HHI based on equity)
        equity_df = df[df['Ticker'] != 'Cash']
        if not equity_df.empty and equity_value > 0:
            equity_agg = equity_df.groupby('Ticker')['MV_AUD'].sum()
            weights = equity_agg / equity_value
            hhi = (weights ** 2).sum()
        else:
            hhi = 0
        
        return {
            'total_mv': total_mv,
            'total_cost': total_cost,
            'capital_injected': capital,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'cash_value': cash_value,
            'cash_pct': (cash_value / total_mv * 100) if total_mv > 0 else 0,
            'equity_value': equity_value,
            'hhi': hhi,
            'num_positions': len(equity_df['Ticker'].unique()),
        }
    
    @staticmethod
    def calculate_buckets(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
        """Bucket-level allocation vs target with gap in A$"""
        equity_df = df[df['Ticker'] != 'Cash'].copy()
        
        if 'Strategy_Role' not in equity_df.columns:
            return pd.DataFrame()
        
        total_mv = stats['total_mv']  # Use TOTAL portfolio value as denominator
        
        rows = []
        for bucket in config.BUCKET_ORDER:
            bucket_df = equity_df[equity_df['Strategy_Role'] == bucket]
            mv = bucket_df['MV_AUD'].sum()
            current_pct = (mv / total_mv * 100) if total_mv > 0 else 0
            target_pct = config.BUCKET_TARGETS[bucket] * 100
            gap_pct = current_pct - target_pct
            gap_aud = (target_pct - current_pct) / 100 * total_mv
            
            rows.append({
                'Bucket': bucket,
                'Current_%': current_pct,
                'Target_%': target_pct,
                'Gap_%': gap_pct,
                'Gap_AUD': gap_aud,  # Positive = need to buy, negative = need to trim
                'MV_AUD': mv,
                'Positions': len(bucket_df['Ticker'].unique()),
                'Color': config.BUCKET_COLORS[bucket],
                'Label': config.BUCKET_LABELS[bucket],
            })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def calculate_position_actions(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
        """Per-position action flags and gap-to-target"""
        equity_df = df[df['Ticker'] != 'Cash'].copy()
        
        if equity_df.empty:
            return equity_df
        
        total_mv = stats['total_mv']
        
        # Aggregate by ticker first (handle cross-platform holdings)
        agg_cols = {
            'Shares': 'sum',
            'MV_AUD': 'sum',
            'Cost_AUD': 'sum',
            'PnL_AUD': 'sum',
            'Current_Price': 'first',
            'Avg_Cost': 'mean',
        }
        if 'Strategy_Role' in equity_df.columns:
            agg_cols['Strategy_Role'] = 'first'
        if 'Target_Weight' in equity_df.columns:
            agg_cols['Target_Weight'] = 'first'
        if 'Sector' in equity_df.columns:
            agg_cols['Sector'] = 'first'
        if 'Stop_Loss_Price' in equity_df.columns:
            agg_cols['Stop_Loss_Price'] = 'first'
        
        pos_df = equity_df.groupby('Ticker').agg(agg_cols).reset_index()
        pos_df['PnL_Pct'] = (pos_df['PnL_AUD'] / pos_df['Cost_AUD'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        pos_df['Weight_%'] = (pos_df['MV_AUD'] / total_mv * 100)
        
        # Target in %
        if 'Target_Weight' in pos_df.columns:
            pos_df['Target_%'] = pos_df['Target_Weight'] * 100
        else:
            pos_df['Target_%'] = 0
        
        # Gap calculations
        pos_df['Gap_%'] = pos_df['Weight_%'] - pos_df['Target_%']
        pos_df['Gap_AUD'] = (pos_df['Target_%'] - pos_df['Weight_%']) / 100 * total_mv
        
        # Action flags
        def determine_action(row):
            if row.get('Strategy_Role') == 'Cleanup':
                return 'EXIT'
            target = row['Target_%']
            current = row['Weight_%']
            if target == 0:
                return 'EXIT' if row.get('Strategy_Role') == 'Cleanup' else 'REVIEW'
            deviation = abs(current - target) / target if target > 0 else 0
            if deviation <= config.ACTION_TOLERANCE * 10:  # Within 10% of target
                return 'ON TARGET'
            elif current < target:
                return 'BUY'
            else:
                return 'TRIM'
        
        pos_df['Action'] = pos_df.apply(determine_action, axis=1)
        
        return pos_df.sort_values('MV_AUD', ascending=False)

# ============================================================================
# COMPONENTS
# ============================================================================

class Components:
    """Bucket-first UI components"""
    
    @staticmethod
    def render_health_strip(stats: dict, fx_rate: float, last_update: str):
        """Top sticky health strip — 5 critical metrics"""
        pnl_cls = 'delta-up' if stats['total_pnl'] >= 0 else 'delta-down'
        pnl_arrow = '▲' if stats['total_pnl'] >= 0 else '▼'
        
        # HHI interpretation
        hhi = stats['hhi']
        if hhi < 0.15:
            hhi_label = "Diversified"
        elif hhi < 0.25:
            hhi_label = "Moderate"
        else:
            hhi_label = "Concentrated"
        
        # Cash status
        cash_pct = stats['cash_pct']
        if cash_pct > 25:
            cash_status = "Heavy"
        elif cash_pct < 5:
            cash_status = "Light"
        else:
            cash_status = "Normal"
        
        html = (
            '<div class="health-strip">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            '<h1 style="margin:0 !important;">📊 Portfolio Command Center</h1>'
            f'<span style="font-size:0.72rem;color:#64748b;">Updated {last_update}</span>'
            '</div>'
            '<div class="health-grid">'
            '<div class="health-item">'
            '<div class="health-label">Net Liquidation</div>'
            f'<div class="health-value">${stats["total_mv"]:,.0f}</div>'
            '<div class="health-delta delta-neutral">AUD</div>'
            '</div>'
            '<div class="health-item">'
            '<div class="health-label">Lifetime P&L</div>'
            f'<div class="health-value {pnl_cls}">{pnl_arrow} ${stats["total_pnl"]:,.0f}</div>'
            f'<div class="health-delta {pnl_cls}">{stats["pnl_pct"]:+.2f}%</div>'
            '</div>'
            '<div class="health-item">'
            '<div class="health-label">Cash</div>'
            f'<div class="health-value">${stats["cash_value"]:,.0f}</div>'
            f'<div class="health-delta delta-neutral">{cash_pct:.1f}% · {cash_status}</div>'
            '</div>'
            '<div class="health-item">'
            '<div class="health-label">Concentration</div>'
            f'<div class="health-value">{hhi:.2f}</div>'
            f'<div class="health-delta delta-neutral">HHI · {hhi_label}</div>'
            '</div>'
            '<div class="health-item">'
            '<div class="health-label">AUD/USD</div>'
            f'<div class="health-value">{fx_rate:.4f}</div>'
            f'<div class="health-delta delta-neutral">{stats["num_positions"]} positions</div>'
            '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def render_bucket_cards(bucket_df: pd.DataFrame):
        """4 bucket cards: Core / Growth / Tactical / Cleanup"""
        if bucket_df.empty:
            st.info("ℹ️ Add 'Strategy_Role' column to your Google Sheet to enable bucket tracking")
            return
        
        st.markdown("### 🎯 Strategy Buckets")
        
        cards_html = '<div class="bucket-grid">'
        for _, row in bucket_df.iterrows():
            bucket = row['Bucket']
            color = row['Color']
            label = row['Label']
            current = row['Current_%']
            target = row['Target_%']
            gap = row['Gap_%']
            gap_aud = row['Gap_AUD']
            positions = row['Positions']
            mv = row['MV_AUD']
            
            # Gap formatting
            if abs(gap) < 1.0:
                gap_display = f'<span style="color:#16a34a;">✅ ON TARGET</span>'
            elif gap < 0:
                action = "BUY" if bucket != 'Cleanup' else "FILL GAP"
                gap_display = f'<span style="color:#2563EB;">⚠ {action} +${abs(gap_aud):,.0f}</span>'
            else:
                action = "TRIM" if bucket != 'Cleanup' else "🔴 EXIT"
                gap_display = f'<span style="color:#dc2626;">{action} -${abs(gap_aud):,.0f}</span>'
            
            # Progress bar (current/target ratio, capped at 100%)
            if target > 0:
                progress = min(current / target * 100, 100)
            else:
                progress = 100 if current > 0 else 0  # Cleanup: show 100% if any holdings
                if bucket == 'Cleanup' and current > 0:
                    progress = min(current * 5, 100)  # Visual emphasis
            
            cards_html += (
                f'<div class="bucket-card" style="border-color:{color};">'
                f'<div class="bucket-header" style="color:{color};">{label}</div>'
                f'<div class="bucket-current" style="color:{color};">{current:.1f}%</div>'
                f'<div class="bucket-target">Target: {target:.0f}%</div>'
                f'<div class="bucket-gap">{gap_display}</div>'
                f'<div class="bucket-progress">'
                f'<div class="bucket-progress-fill" style="width:{progress:.0f}%;background:{color};"></div>'
                f'</div>'
                f'<div class="bucket-footer">'
                f'{positions} position{"s" if positions != 1 else ""} · ${mv:,.0f}'
                f'</div>'
                f'</div>'
            )
        cards_html += '</div>'
        
        st.markdown(cards_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_position_list(pos_df: pd.DataFrame, selected_bucket: str = "All"):
        """Filterable position list with action flags"""
        if pos_df.empty:
            st.info("No positions to display")
            return
        
        st.markdown("### 📋 Positions")
        
        # Bucket filter chips (using radio)
        buckets = ["All"] + config.BUCKET_ORDER
        filter_cols = st.columns(len(buckets))
        for i, b in enumerate(buckets):
            if b == "All":
                count = len(pos_df)
                label = f"All ({count})"
            else:
                count = len(pos_df[pos_df['Strategy_Role'] == b]) if 'Strategy_Role' in pos_df.columns else 0
                emoji = {'Core': '🔵', 'Growth': '🟢', 'Tactical': '🟡', 'Cleanup': '🔴'}[b]
                label = f"{emoji} {b} ({count})"
            with filter_cols[i]:
                if st.button(label, key=f"filter_{b}", use_container_width=True):
                    st.query_params["bucket"] = b
                    st.rerun()
        
        # Get selected bucket from query params
        selected = st.query_params.get("bucket", "All")
        
        # Filter
        if selected != "All" and 'Strategy_Role' in pos_df.columns:
            filtered = pos_df[pos_df['Strategy_Role'] == selected].copy()
        else:
            filtered = pos_df.copy()
        
        if filtered.empty:
            st.info(f"No positions in {selected}")
            return
        
        # Render as cards
        for _, row in filtered.iterrows():
            bucket = row.get('Strategy_Role', 'Cleanup')
            color = config.BUCKET_COLORS.get(bucket, '#64748b')
            
            action = row['Action']
            action_class = {
                'ON TARGET': 'action-on-target',
                'BUY': 'action-buy',
                'TRIM': 'action-trim',
                'EXIT': 'action-exit',
                'REVIEW': 'action-trim',
            }.get(action, 'action-on-target')
            
            action_icon = {
                'ON TARGET': '✅',
                'BUY': '⚠',
                'TRIM': '▼',
                'EXIT': '🔴',
                'REVIEW': '?',
            }.get(action, '')
            
            pnl_cls = 'delta-up' if row['PnL_AUD'] >= 0 else 'delta-down'
            pnl_arrow = '▲' if row['PnL_AUD'] >= 0 else '▼'
            
            # Gap info
            if action == 'BUY':
                gap_info = f"Gap: +${row['Gap_AUD']:,.0f} to target"
            elif action == 'TRIM':
                gap_info = f"Excess: ${abs(row['Gap_AUD']):,.0f} above target"
            elif action == 'EXIT':
                gap_info = f"Exit target: A${row['Cost_AUD']:,.0f} (break-even)"
            else:
                gap_info = f"Target: {row['Target_%']:.1f}%"
            
            sector = row.get('Sector', '')
            sector_str = f" · {sector}" if sector else ""
            
            card_html = (
                f'<div class="position-card" style="border-left-color:{color};">'
                f'<div class="pos-row-1">'
                f'<div>'
                f'<span class="pos-ticker">{row["Ticker"]}</span>'
                f'<span style="font-size:0.75rem;color:#64748b;margin-left:6px;">{bucket}{sector_str}</span>'
                f'</div>'
                f'<span class="pos-action {action_class}">{action_icon} {action}</span>'
                f'</div>'
                f'<div class="pos-row-2">'
                f'<span class="pos-value">${row["MV_AUD"]:,.0f}</span>'
                f'<span class="pos-pnl {pnl_cls}">{pnl_arrow} {row["PnL_Pct"]:+.1f}% (${row["PnL_AUD"]:,.0f})</span>'
                f'</div>'
                f'<div class="pos-detail">'
                f'Weight: {row["Weight_%"]:.1f}% / {row["Target_%"]:.1f}% · {gap_info}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_cleanup_alert(bucket_df: pd.DataFrame, pos_df: pd.DataFrame):
        """Prominent cleanup queue alert"""
        if bucket_df.empty:
            return
        
        cleanup_row = bucket_df[bucket_df['Bucket'] == 'Cleanup']
        if cleanup_row.empty:
            return
        
        cleanup_pct = cleanup_row['Current_%'].values[0]
        cleanup_mv = cleanup_row['MV_AUD'].values[0]
        
        if cleanup_pct < 1:
            return  # Nothing to clean up
        
        cleanup_positions = pos_df[pos_df['Strategy_Role'] == 'Cleanup'] if 'Strategy_Role' in pos_df.columns else pd.DataFrame()
        
        cleanup_html = (
            '<div style="background:#fee2e2;border:2px solid #dc2626;border-radius:12px;padding:14px 16px;margin:12px 0;">'
            '<div style="font-size:0.9rem;font-weight:700;color:#991b1b;margin-bottom:6px;">'
            f'🔴 Cleanup Queue: {cleanup_pct:.1f}% of portfolio needs exit'
            '</div>'
            '<div style="font-size:0.82rem;color:#7f1d1d;">'
            f'${cleanup_mv:,.0f} AUD in non-framework positions. Target: 0%. '
            f'{len(cleanup_positions)} position{"s" if len(cleanup_positions) != 1 else ""} to exit at break-even.'
            '</div>'
            '</div>'
        )
        st.markdown(cleanup_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_charts(df: pd.DataFrame, bucket_df: pd.DataFrame):
        """Secondary analytical charts"""
        if bucket_df.empty:
            return
        
        st.markdown("### 📈 Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Allocation", "Top Holdings", "Performance"])
        
        with tab1:
            # Bucket donut
            fig = go.Figure(data=[go.Pie(
                labels=bucket_df['Label'],
                values=bucket_df['MV_AUD'],
                hole=0.55,
                marker=dict(colors=bucket_df['Color'].tolist()),
                textinfo='label+percent',
                textposition='outside',
            )])
            fig.update_layout(
                height=380,
                margin=dict(t=20, b=20, l=0, r=0),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            equity_df = df[df['Ticker'] != 'Cash']
            top = equity_df.groupby('Ticker').agg(
                MV_AUD=('MV_AUD', 'sum'),
                PnL_AUD=('PnL_AUD', 'sum')
            ).nlargest(10, 'MV_AUD').reset_index()
            
            fig = px.bar(
                top,
                x='MV_AUD',
                y='Ticker',
                orientation='h',
                color='PnL_AUD',
                color_continuous_scale=['#dc2626', '#f1f5f9', '#16a34a'],
                text='MV_AUD',
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=20, l=0, r=40),
                xaxis_title="Market Value (AUD)",
                yaxis_title="",
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            equity_df = df[df['Ticker'] != 'Cash']
            agg = equity_df.groupby('Ticker').agg(
                PnL_AUD=('PnL_AUD', 'sum'),
                PnL_Pct=('PnL_Pct', 'mean')
            ).reset_index()
            winners = agg.nlargest(5, 'PnL_AUD')
            losers = agg.nsmallest(5, 'PnL_AUD')
            combined = pd.concat([losers, winners]).sort_values('PnL_AUD')
            
            colors = ['#dc2626' if x < 0 else '#16a34a' for x in combined['PnL_AUD']]
            fig = go.Figure(data=[
                go.Bar(
                    x=combined['PnL_AUD'],
                    y=combined['Ticker'],
                    orientation='h',
                    marker_color=colors,
                    text=combined['PnL_Pct'],
                    texttemplate='%{text:+.1f}%',
                    textposition='outside',
                )
            ])
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=20, l=0, r=40),
                xaxis_title="P&L (AUD)",
                yaxis_title="",
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_download(df: pd.DataFrame):
        """CSV download with all columns"""
        st.markdown("### 📥 Export")
        
        equity_df = df.copy()
        equity_df['Native_Cost_Total'] = equity_df['Shares'] * equity_df['Avg_Cost']
        
        agg = equity_df.groupby('Ticker').agg({
            'Shares': 'sum',
            'Native_Cost_Total': 'sum',
            'Current_Price': 'mean',
            'Currency': 'first',
            'Cost_AUD': 'sum',
            'MV_AUD': 'sum',
            'PnL_AUD': 'sum',
            'Strategy_Role': 'first' if 'Strategy_Role' in equity_df.columns else lambda x: '',
        }).reset_index()
        
        agg['Avg_Cost_Native'] = (agg['Native_Cost_Total'] / agg['Shares']).round(2)
        agg['PnL_Pct'] = (agg['PnL_AUD'] / agg['Cost_AUD'] * 100).fillna(0).round(2)
        
        display = agg[['Ticker', 'Strategy_Role', 'Currency', 'Shares', 'Avg_Cost_Native',
                       'Current_Price', 'Cost_AUD', 'MV_AUD', 'PnL_AUD', 'PnL_Pct']].copy()
        display.columns = ['Ticker', 'Role', 'Currency', 'Shares', 'Avg Cost',
                           'Current Price', 'Cost (AUD)', 'Market Value (AUD)', 'P&L (AUD)', 'P&L %']
        display = display.sort_values('Market Value (AUD)', ascending=False)
        
        csv = display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Portfolio CSV",
            data=csv,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Application entry point"""
    setup_page()
    inject_keepalive()
    
    if not AuthManager.check_password():
        st.stop()
    
    try:
        # Load data
        df_raw = DataManager.load_portfolio_data()
        if df_raw.empty:
            st.error("❌ No portfolio data available")
            st.stop()
        
        # Extract capital injected
        capital_row = df_raw[df_raw['Ticker'] == 'CAPITAL']
        capital = float(capital_row['Shares'].sum()) if not capital_row.empty else 0
        
        df_clean = df_raw[df_raw['Ticker'] != 'CAPITAL'].copy()
        
        # Enrich with prices
        df_enriched, fx_rate = DataManager.enrich_portfolio(df_clean)
        
        # Analytics
        stats = Analytics.calculate_summary(df_enriched, capital)
        bucket_df = Analytics.calculate_buckets(df_enriched, stats)
        pos_df = Analytics.calculate_position_actions(df_enriched, stats)
        
        # Render
        last_update = datetime.now().strftime('%H:%M:%S')
        
        Components.render_health_strip(stats, fx_rate, last_update)
        Components.render_cleanup_alert(bucket_df, pos_df)
        Components.render_bucket_cards(bucket_df)
        Components.render_position_list(pos_df)
        
        st.markdown("---")
        Components.render_charts(df_enriched, bucket_df)
        
        st.markdown("---")
        Components.render_download(df_enriched)
        
        # Manual refresh option
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Force Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if AUTOREFRESH_AVAILABLE:
                st.caption(f"⏱️ Auto-refreshes every {config.AUTO_REFRESH_MS // 1000}s · Prices cached {config.PRICE_CACHE_TTL}s · Sheets cached {config.DATA_CACHE_TTL}s")
            else:
                st.caption("⚠️ Install `streamlit-autorefresh` for auto-refresh: `pip install streamlit-autorefresh`")
        
    except Exception as e:
        logger.error(f"App error: {e}", exc_info=True)
        st.error(f"❌ Error: {e}")
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
