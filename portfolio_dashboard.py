"""
Strategic Portfolio Dashboard
A modern, production-ready portfolio tracking application
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DashboardConfig:
    """Centralized configuration management"""
    SHEET_URL: str = "https://docs.google.com/spreadsheets/d/14IGIMj9iR5qOtmYT1e6FgN8t2tdQ5M1R_-hS6rw1RQs/export?format=csv"
    DEFAULT_FX_RATE: float = 0.70
    CACHE_TTL: int = 60  # 1 minute - keeps data fresh
    YF_PERIOD: str = "5d"
    PIE_THRESHOLD: float = 0.85
    
    # Color schemes
    COLORS_PRIMARY: list = None
    COLORS_ACCENT: list = None
    
    def __post_init__(self):
        self.COLORS_PRIMARY = ['#2E4053', '#5D6D7E', '#85929E', '#AED6F1', 
                               '#F5B041', '#EC7063', '#48C9B0', '#AF7AC5']
        self.COLORS_ACCENT = ['#1ABC9C', '#3498DB', '#9B59B6', '#E74C3C']

config = DashboardConfig()

# ============================================================================
# PAGE SETUP
# ============================================================================

def setup_page():
    """Configure page settings and custom CSS"""
    st.set_page_config(
        page_title="Portfolio Command Center",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <script>
    // Store viewport width in session storage for Python to read
    const width = window.innerWidth;
    const isMobile = width < 768;
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: {mobile: isMobile}}, '*');
    </script>
    <style>
        /* ── Base ── */
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], .main, section.main > div {
            background-color: #ffffff !important;
        }

        /* ── Typography ── */
        h1, h2, h3, h4, h5, h6,
        p, span, div, label, li, td, th, caption {
            color: #1a1a1a !important;
            text-shadow: none !important;
            background-color: transparent !important;
        }
        h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
        h2 { font-size: 1.25rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.1rem  !important; font-weight: 600 !important; }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 12px !important;
            padding: 0.9rem 1rem !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
        }
        [data-testid="stMetric"] * { color: #1a1a1a !important; }
        [data-testid="stMetricLabel"]  { font-size: 0.78rem !important; font-weight: 600 !important; color: #555 !important; }
        [data-testid="stMetricValue"]  { font-size: 1.3rem  !important; font-weight: 700 !important; }
        [data-testid="stMetricDelta"]  { font-size: 0.8rem  !important; font-weight: 500 !important; }

        /* ── Buttons ── */
        .stButton > button {
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            border: 2px solid #dee2e6 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            min-height: 44px !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background-color: #2E4053 !important;
            color: #ffffff !important;
            border-color: #2E4053 !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f8f9fa !important;
            border-radius: 10px !important;
            padding: 4px !important;
            gap: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #555 !important;
            background-color: transparent !important;
            border-radius: 8px !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            padding: 6px 10px !important;
            min-height: 36px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        }

        /* ── Radio ── */
        .stRadio label { color: #1a1a1a !important; font-weight: 500 !important; }

        /* ── Alerts ── */
        [data-testid="stAlert"] { border-radius: 10px !important; }
        [data-testid="stAlert"] * { color: #1a1a1a !important; }

        /* ── Tables ── */
        [data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
        .dataframe, .dataframe * { color: #1a1a1a !important; }
        .dataframe thead tr th {
            background-color: #f8f9fa !important;
            font-weight: 700 !important;
            font-size: 0.8rem !important;
        }
        .dataframe tbody tr td { background-color: #ffffff !important; font-size: 0.82rem !important; }

        /* ── Progress ── */
        [data-testid="stProgressBar"] > div { background-color: #2E4053 !important; }

        /* ── Hide chrome ── */
        footer, #MainMenu, header { visibility: hidden !important; }
        [data-testid="stToolbar"]  { display: none !important; }

        /* ── Mobile-specific hiding ── */
        @media (max-width: 768px) {
            .hide-on-mobile { display: none !important; }
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# AUTHENTICATION
# ============================================================================

class AuthManager:
    """Secure authentication handler"""
    
    @staticmethod
    def check_password() -> bool:
        """
        Verify user password against stored secret
        Returns: True if authenticated, False otherwise
        """
        def password_entered():
            try:
                if st.session_state["password"] == st.secrets.get("PASSWORD", ""):
                    st.session_state["password_correct"] = True
                    del st.session_state["password"]
                else:
                    st.session_state["password_correct"] = False
                    st.error("❌ Invalid access code")
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                st.session_state["password_correct"] = False
        
        if "password_correct" not in st.session_state:
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
        
        return st.session_state["password_correct"]

# ============================================================================
# DATA LAYER
# ============================================================================

class DataManager:
    """Handles all data operations with error handling"""
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def load_portfolio_data() -> pd.DataFrame:
        """
        Load and validate portfolio data from Google Sheets
        Returns: Cleaned DataFrame
        Raises: ValueError if data validation fails
        """
        try:
            df = pd.read_csv(config.SHEET_URL)
            df.columns = df.columns.str.strip()
            
            # Data validation
            required_cols = ['Ticker', 'Shares', 'Avg_Cost']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Clean numeric columns
            numeric_cols = ['Shares', 'Avg_Cost', 'Stop_Loss_Price']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', ''), 
                        errors='coerce'
                    ).fillna(0)
            
            # Handle target weights
            if 'Target_Weight' in df.columns:
                df['Target_Weight'] = pd.to_numeric(
                    df['Target_Weight'].astype(str).str.replace('%', ''), 
                    errors='coerce'
                )
                df.loc[df['Target_Weight'] > 1.0, 'Target_Weight'] /= 100
            
            logger.info(f"Successfully loaded {len(df)} portfolio entries")
            return df
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            st.error(f"⚠️ Failed to load portfolio data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def fetch_market_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Fetch live market prices and calculate valuations
        Args:
            df: Portfolio DataFrame
        Returns:
            Tuple of (enhanced DataFrame, FX rate)
        """
        try:
            # Get unique tickers
            tickers = df[df['Ticker'] != 'Cash']['Ticker'].unique().tolist()
            tickers.append("AUDUSD=X")
            
            # Fetch prices
            with st.spinner('📡 Syncing market data...'):
                data = yf.download(
                    tickers, 
                    period=config.YF_PERIOD, 
                    progress=False
                )['Close']
                
                if data.empty:
                    raise ValueError("No market data received")
                
                latest_prices = data.ffill().iloc[-1]
            
            # Extract FX rate
            fx_rate = latest_prices.get('AUDUSD=X', config.DEFAULT_FX_RATE)
            if pd.isna(fx_rate) or fx_rate == 0:
                logger.warning("Invalid FX rate, using default")
                fx_rate = config.DEFAULT_FX_RATE
            
            # Map prices to portfolio
            df['Current_Price'] = df['Ticker'].map(latest_prices).fillna(df['Avg_Cost'])
            df.loc[df['Ticker'] == 'Cash', 'Current_Price'] = 1.0
            
            # Calculate valuations in AUD
            df['MV_AUD'] = df.apply(
                lambda r: (r['Current_Price'] * r['Shares']) / fx_rate 
                if r.get('Currency') == 'USD' 
                else r['Current_Price'] * r['Shares'],
                axis=1
            )
            
            df['Cost_AUD'] = df.apply(
                lambda r: (r['Avg_Cost'] * r['Shares']) / fx_rate 
                if r.get('Currency') == 'USD' 
                else r['Avg_Cost'] * r['Shares'],
                axis=1
            )
            
            df['PnL_AUD'] = df['MV_AUD'] - df['Cost_AUD']
            df['PnL_Pct'] = (df['PnL_AUD'] / df['Cost_AUD'] * 100).fillna(0)
            
            logger.info(f"Market data fetched successfully (FX: {fx_rate:.4f})")
            return df, fx_rate
            
        except Exception as e:
            logger.error(f"Market data fetch error: {e}")
            st.warning(f"⚠️ Using cached prices: {str(e)}")
            # Return original df with FX default
            return df, config.DEFAULT_FX_RATE

# ============================================================================
# ANALYTICS
# ============================================================================

class PortfolioAnalytics:
    """Portfolio analysis and calculations"""
    
    @staticmethod
    def calculate_summary_stats(df: pd.DataFrame, capital: float) -> dict:
        """Calculate portfolio-level statistics"""
        total_mv = df['MV_AUD'].sum()
        total_cost = df['Cost_AUD'].sum()
        
        # Total P&L = Current Market Value - Capital Injected
        total_pnl = total_mv - capital
        pnl_pct = (total_pnl / capital * 100) if capital > 0 else 0
        
        # Stock performance P&L (for reference)
        stock_pnl = total_mv - total_cost
        stock_pnl_pct = (stock_pnl / total_cost * 100) if total_cost > 0 else 0
        
        cash_value = df[df['Ticker'] == 'Cash']['MV_AUD'].sum()
        equity_value = total_mv - cash_value
        
        # Count unique tickers excluding Cash
        unique_positions = len(df[df['Ticker'] != 'Cash']['Ticker'].unique())
        
        # Winner/Loser analysis
        winners = df[(df['PnL_AUD'] > 0) & (df['Ticker'] != 'Cash')]
        losers = df[(df['PnL_AUD'] < 0) & (df['Ticker'] != 'Cash')]
        
        return {
            'total_mv': total_mv,
            'total_cost': total_cost,
            'capital_injected': capital,
            'total_pnl': total_pnl,  # This is MV - Capital Injected
            'pnl_pct': pnl_pct,
            'stock_pnl': stock_pnl,  # This is performance-based P&L
            'stock_pnl_pct': stock_pnl_pct,
            'num_positions': unique_positions,  # Fixed: count unique tickers only
            'cash_value': cash_value,
            'cash_pct': (cash_value / total_mv * 100) if total_mv > 0 else 0,
            'equity_value': equity_value,
            'equity_pct': (equity_value / total_mv * 100) if total_mv > 0 else 0,
            'num_winners': len(winners['Ticker'].unique()),  # Count unique winners
            'num_losers': len(losers['Ticker'].unique()),  # Count unique losers
            'winners_value': winners.groupby('Ticker')['PnL_AUD'].sum().sum() if not winners.empty else 0,
            'losers_value': losers.groupby('Ticker')['PnL_AUD'].sum().sum() if not losers.empty else 0,
            'best_performer': winners.groupby('Ticker')['PnL_Pct'].mean().idxmax() if not winners.empty else 'N/A',
            'best_performer_pct': winners.groupby('Ticker')['PnL_Pct'].mean().max() if not winners.empty else 0,
            'worst_performer': losers.groupby('Ticker')['PnL_Pct'].mean().idxmin() if not losers.empty else 'N/A',
            'worst_performer_pct': losers.groupby('Ticker')['PnL_Pct'].mean().min() if not losers.empty else 0,
        }
    
    @staticmethod
    def prepare_pie_data(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Aggregate small positions into 'Others' category
        Args:
            df: Portfolio DataFrame
            threshold: Cumulative percentage threshold for grouping
        Returns:
            Cleaned DataFrame for pie charts
        """
        threshold = threshold or config.PIE_THRESHOLD
        
        agg = df.groupby('Ticker')['MV_AUD'].sum().sort_values(ascending=False)
        total = agg.sum()
        
        if total == 0:
            return pd.DataFrame({'Ticker': [], 'MV_AUD': []})
        
        cumsum = agg.cumsum() / total
        
        main = agg[cumsum <= threshold]
        others = agg[cumsum > threshold]
        
        if not others.empty and len(others) > 1:
            other_label = f"Others ({len(others)} positions)"
            result = pd.concat([
                main,
                pd.Series([others.sum()], index=[other_label])
            ])
            result_df = result.reset_index()
            result_df.columns = ['Ticker', 'MV_AUD']
            return result_df
        
        agg_df = agg.reset_index()
        agg_df.columns = ['Ticker', 'MV_AUD']
        return agg_df
    
    @staticmethod
    def calculate_strategy_allocation(df: pd.DataFrame, stats: dict) -> dict:
        """Calculate strategy role allocations vs targets"""
        equity_df = df[df['Ticker'] != 'Cash'].copy()
        
        if 'Strategy_Role' not in equity_df.columns and 'Strategy Role' not in equity_df.columns:
            return None
        
        role_col = 'Strategy_Role' if 'Strategy_Role' in equity_df.columns else 'Strategy Role'
        target_col = 'Target_Weight' if 'Target_Weight' in equity_df.columns else None
        
        # First aggregate by ticker to get unique position targets
        # (handles duplicate tickers across platforms)
        ticker_agg = equity_df.groupby(['Ticker', role_col]).agg({
            'MV_AUD': 'sum',
            'Target_Weight': 'first' if target_col else lambda x: 0  # Use .first() not .sum()
        }).reset_index()
        
        # Then aggregate by role
        role_agg = ticker_agg.groupby(role_col).agg({
            'MV_AUD': 'sum',
            'Target_Weight': 'sum'  # Now safe to sum since tickers are unique
        }).reset_index()
        
        equity_value = stats['equity_value'] if stats['equity_value'] > 0 else 1
        
        # Calculate current % and target %
        role_agg['Current_%'] = (role_agg['MV_AUD'] / equity_value * 100).round(1)
        role_agg['Target_%'] = (role_agg['Target_Weight'] * 100).round(1) if target_col else 0
        role_agg['Gap_%'] = (role_agg['Current_%'] - role_agg['Target_%']).round(1)
        
        # Define role colors and order
        role_colors = {
            'Core': '#2E4053',      # Dark blue
            'Growth': '#1a9655',    # Green
            'Tactical': '#F5B041',  # Yellow/Gold
        }
        
        role_order = ['Core', 'Growth', 'Tactical']
        
        return {
            'data': role_agg,
            'colors': role_colors,
            'order': role_order,
            'equity_value': equity_value
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class ChartBuilder:
    """Modern chart creation with consistent styling"""
    
    @staticmethod
    def create_pie_chart(
        data: pd.DataFrame, 
        title: str,
        colors: list = None
    ) -> go.Figure:
        """Create a modern donut chart"""
        colors = colors or config.COLORS_PRIMARY
        
        fig = px.pie(
            data,
            values='MV_AUD',
            names='Ticker',
            hole=0.5,
            color_discrete_sequence=colors
        )
        
        fig.update_layout(
            margin=dict(t=40, b=20, l=0, r=0),
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11)
            ),
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2E4053')
            )
        )
        
        fig.update_traces(
            textinfo='label+percent',
            textposition='inside',
            textfont_size=11,
            marker=dict(line=dict(color='rgba(0,0,0,0)', width=0))
        )
        
        return fig
    
    @staticmethod
    def create_allocation_bar(data: pd.DataFrame) -> go.Figure:
        """Create horizontal allocation bar chart"""
        top_10 = data.nlargest(10, 'MV_AUD')
        
        fig = px.bar(
            top_10,
            x='MV_AUD',
            y='Ticker',
            orientation='h',
            color='PnL_AUD',
            color_continuous_scale=['#EC7063', '#F5F5F5', '#48C9B0'],
            text='MV_AUD'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=0, r=0),
            xaxis_title="Market Value (AUD)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        
        fig.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='outside'
        )
        
        return fig
    
    @staticmethod
    def create_sector_chart(df: pd.DataFrame) -> go.Figure:
        """Create sector allocation pie chart"""
        if 'Sector' not in df.columns:
            return None
        
        sector_data = df[df['Ticker'] != 'Cash'].groupby('Sector')['MV_AUD'].sum().sort_values(ascending=False)
        
        if sector_data.empty:
            return None
        
        sector_df = sector_data.reset_index()
        sector_df.columns = ['Sector', 'MV_AUD']
        
        fig = px.pie(
            sector_df,
            values='MV_AUD',
            names='Sector',
            hole=0.5,
            color_discrete_sequence=config.COLORS_PRIMARY
        )
        
        fig.update_layout(
            margin=dict(t=40, b=20, l=0, r=0),
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11)
            ),
            title=dict(
                text="Sector Allocation",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2E4053')
            )
        )
        
        fig.update_traces(
            textinfo='label+percent',
            textposition='inside',
            textfont_size=11,
            marker=dict(line=dict(color='rgba(0,0,0,0)', width=0))
        )
        
        return fig
    
    @staticmethod
    def create_strategy_chart(df: pd.DataFrame) -> go.Figure:
        """Create strategy role allocation pie chart"""
        if 'Strategy_Role' not in df.columns and 'Strategy Role' not in df.columns:
            return None
        
        strategy_col = 'Strategy_Role' if 'Strategy_Role' in df.columns else 'Strategy Role'
        strategy_data = df[df['Ticker'] != 'Cash'].groupby(strategy_col)['MV_AUD'].sum().sort_values(ascending=False)
        
        if strategy_data.empty:
            return None
        
        strategy_df = strategy_data.reset_index()
        strategy_df.columns = ['Strategy', 'MV_AUD']
        
        fig = px.pie(
            strategy_df,
            values='MV_AUD',
            names='Strategy',
            hole=0.5,
            color_discrete_sequence=config.COLORS_PRIMARY
        )
        
        fig.update_layout(
            margin=dict(t=40, b=20, l=0, r=0),
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11)
            ),
            title=dict(
                text="Strategy Allocation",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2E4053')
            )
        )
        
        fig.update_traces(
            textinfo='label+percent',
            textposition='inside',
            textfont_size=11,
            marker=dict(line=dict(color='rgba(0,0,0,0)', width=0))
        )
        
        return fig
    
    @staticmethod
    def create_performance_chart(df: pd.DataFrame) -> go.Figure:
        """Create top winners vs losers chart"""
        equity_df = df[df['Ticker'] != 'Cash'].copy()
        
        if equity_df.empty:
            return None
        
        # Get top 5 winners and losers
        winners = equity_df.nlargest(5, 'PnL_AUD')[['Ticker', 'PnL_AUD', 'PnL_Pct']]
        losers = equity_df.nsmallest(5, 'PnL_AUD')[['Ticker', 'PnL_AUD', 'PnL_Pct']]
        
        combined = pd.concat([winners, losers]).sort_values('PnL_AUD')
        
        colors = ['#EC7063' if x < 0 else '#48C9B0' for x in combined['PnL_AUD']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=combined['PnL_AUD'],
                y=combined['Ticker'],
                orientation='h',
                marker_color=colors,
                text=combined['PnL_Pct'],
                texttemplate='%{text:.1f}%',
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>P&L: $%{x:,.0f}<br>Return: %{text:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="Top Performers",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#2E4053')
            ),
            xaxis_title="P&L (AUD)",
            yaxis_title="",
            height=400,
            margin=dict(t=40, b=20, l=0, r=20),
            showlegend=False
        )
        
        return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

class Dashboard:
    """Main dashboard UI orchestration"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.analytics = PortfolioAnalytics()
        self.charts = ChartBuilder()
    
    def render_header(self, stats: dict, fx_rate: float):
        """Render dashboard header with responsive metric cards"""

        CARD = (
            "background:#f8f9fa;border:1px solid #e0e0e0;border-radius:12px;"
            "padding:12px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.07);"
        )
        LABEL = (
            "font-size:0.72rem;font-weight:600;color:#666;"
            "text-transform:uppercase;letter-spacing:0.3px;margin-bottom:4px;"
        )
        VALUE = "font-size:1.15rem;font-weight:700;color:#1a1a1a;line-height:1.2;"
        D_UP  = "font-size:0.78rem;font-weight:600;color:#1a9655;margin-top:3px;"
        D_DN  = "font-size:0.78rem;font-weight:600;color:#dc3545;margin-top:3px;"
        D_NEU = "font-size:0.78rem;font-weight:600;color:#555;margin-top:3px;"

        GRID = (
            "display:grid;grid-template-columns:1fr 1fr;"
            "gap:10px;margin-bottom:10px;"
        )

        def _card(label, value, delta="", delta_dir="neu"):
            d_style = {"up": D_UP, "down": D_DN, "neu": D_NEU}.get(delta_dir, D_NEU)
            arrow   = "▲ " if delta_dir == "up" else ("▼ " if delta_dir == "down" else "")
            d_html  = f'<div style="{d_style}">{arrow}{delta}</div>' if delta else ""
            return (
                f'<div style="{CARD}">'
                f'<div style="{LABEL}">{label}</div>'
                f'<div style="{VALUE}">{value}</div>'
                f'{d_html}'
                f'</div>'
            )

        pnl_dir  = "up"   if stats['total_pnl'] >= 0 else "down"
        spnl_dir = "up"   if stats['stock_pnl'] >= 0 else "down"

        st.title("📊 Portfolio Command Center")
        st.caption(f"Updated: {datetime.now().strftime('%d %b %Y  %H:%M')}")

        # ── Row 1 ──────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="{GRID}">'
            + _card("Total Value",      f"${stats['total_mv']:,.0f}")
            + _card("Total P&L",        f"${stats['total_pnl']:,.0f}",
                    f"{stats['pnl_pct']:.2f}%", pnl_dir)
            + _card("Capital Injected", f"${stats['capital_injected']:,.0f}")
            + _card("Stock Positions",  str(stats['num_positions']),
                    f"${stats['cash_value']:,.0f} cash", "neu")
            + '</div>',
            unsafe_allow_html=True
        )

        # ── Row 2 ──────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="{GRID}">'
            + _card("Stock Performance", f"${stats['stock_pnl']:,.0f}",
                    f"{stats['stock_pnl_pct']:.2f}%", spnl_dir)
            + _card("AUD / USD",         f"{fx_rate:.4f}")
            + _card("Best Performer",    stats['best_performer'],
                    f"+{stats['best_performer_pct']:.1f}%", "up")
            + _card("Worst Performer",   stats['worst_performer'],
                    f"{stats['worst_performer_pct']:.1f}%", "down")
            + '</div>'
            + '<hr style="border:none;border-top:1px solid #e9ecef;margin:16px 0;">',
            unsafe_allow_html=True
        )
    
    def render_charts(self, df: pd.DataFrame):
        """Render portfolio visualization charts"""
        st.markdown("---")
        st.subheader("📈 Portfolio Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["💼 Allocation", "🎯 Strategy & Sectors", "🏆 Performance"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # All assets including cash
                pie_data_all = self.analytics.prepare_pie_data(df)
                fig1 = self.charts.create_pie_chart(
                    pie_data_all,
                    "Asset Allocation (All)"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Equities only
                df_equity = df[df['Ticker'] != 'Cash']
                pie_data_equity = self.analytics.prepare_pie_data(df_equity)
                fig2 = self.charts.create_pie_chart(
                    pie_data_equity,
                    "Equity Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector allocation
                sector_fig = self.charts.create_sector_chart(df)
                if sector_fig:
                    st.plotly_chart(sector_fig, use_container_width=True)
                else:
                    st.info("📊 Sector data not available. Add a 'Sector' column to your spreadsheet.")
            
            with col2:
                # Strategy allocation
                strategy_fig = self.charts.create_strategy_chart(df)
                if strategy_fig:
                    st.plotly_chart(strategy_fig, use_container_width=True)
                else:
                    st.info("🎯 Strategy data not available. Add a 'Strategy_Role' column to your spreadsheet.")
        
        with tab3:
            # Top performers chart
            perf_fig = self.charts.create_performance_chart(df)
            if perf_fig:
                st.plotly_chart(perf_fig, use_container_width=True)
            else:
                st.info("No performance data available")
            
            # Additional performance metrics
            st.markdown("### 📊 Performance Breakdown")
            col1, col2, col3 = st.columns(3)
            
            equity_df = df[df['Ticker'] != 'Cash']
            
            with col1:
                avg_return = equity_df['PnL_Pct'].mean() if not equity_df.empty else 0
                st.metric("Avg Return", f"{avg_return:.2f}%")
            
            with col2:
                median_return = equity_df['PnL_Pct'].median() if not equity_df.empty else 0
                st.metric("Median Return", f"{median_return:.2f}%")
            
            with col3:
                win_rate = (len(equity_df[equity_df['PnL_AUD'] > 0]) / len(equity_df) * 100) if not equity_df.empty else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
    
    def render_holdings_table(self, df: pd.DataFrame, force_mobile: bool = False):
        """Render holdings - auto-adapts to mobile or desktop mode"""
        st.markdown("---")
        st.subheader("📋 Holdings")

        if force_mobile:
            # ═══════════════════════════════════════════════════════════════
            # MOBILE MODE - Clean HTML card list (no choice, just show it)
            # ═══════════════════════════════════════════════════════════════
            equity_df = df[df['Ticker'] != 'Cash'].copy()
            mob = equity_df.groupby('Ticker').agg(
                MV_AUD  =('MV_AUD',  'sum'),
                Cost_AUD=('Cost_AUD','sum'),
                PnL_AUD =('PnL_AUD', 'sum'),
            ).reset_index()
            mob['PnL_%'] = (mob['PnL_AUD'] / mob['Cost_AUD'] * 100).fillna(0)
            mob = mob.sort_values('MV_AUD', ascending=False)

            total_mv   = mob['MV_AUD'].sum()
            total_cost = mob['Cost_AUD'].sum()
            total_pnl  = mob['PnL_AUD'].sum()
            total_pct  = (total_pnl / total_cost * 100) if total_cost else 0

            # Inline styles
            ROW_BASE   = "display:flex;justify-content:space-between;align-items:center;padding:11px 15px;border-bottom:1px solid #f0f0f0;"
            ROW_TOTAL  = ROW_BASE + "background:#f8f9fa;font-weight:700;border-top:2px solid #2E4053;"
            WRAP       = "background:#fff;border:1px solid #e0e0e0;border-radius:12px;margin:12px 0;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);"
            HDR        = "display:flex;justify-content:space-between;padding:11px 15px;background:linear-gradient(135deg,#2E4053 0%,#34495e 100%);color:#fff;"
            HDR_T      = "color:#fff !important;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;"
            TICKER_S   = "font-weight:700;font-size:1.05rem;color:#1a1a1a;min-width:70px;"
            MV_S       = "font-size:1rem;font-weight:700;color:#1a1a1a;margin-bottom:4px;"
            PNL_UP     = "font-size:0.88rem;font-weight:600;color:#1a9655;margin-bottom:3px;"
            PNL_DN     = "font-size:0.88rem;font-weight:600;color:#dc3545;margin-bottom:3px;"
            COST_S     = "font-size:0.78rem;color:#888;"

            rows_html = ""
            for _, r in mob.iterrows():
                pnl_style = PNL_UP if r['PnL_AUD'] >= 0 else PNL_DN
                arrow     = "▲" if r['PnL_AUD'] >= 0 else "▼"
                rows_html += (
                    f'<div style="{ROW_BASE}">'
                    f'  <div style="{TICKER_S}">{r["Ticker"]}</div>'
                    f'  <div style="flex:1;text-align:right;">'
                    f'    <div style="{MV_S}">${r["MV_AUD"]:,.0f}</div>'
                    f'    <div style="{pnl_style}">{arrow} {r["PnL_%"]:+.1f}% (${r["PnL_AUD"]:,.0f})</div>'
                    f'    <div style="{COST_S}">Cost: ${r["Cost_AUD"]:,.0f}</div>'
                    f'  </div>'
                    f'</div>'
                )

            total_pnl_style = PNL_UP if total_pnl >= 0 else PNL_DN
            total_arrow     = "▲" if total_pnl >= 0 else "▼"

            html = (
                f'<div style="{WRAP}">'
                f'  <div style="{HDR}">'
                f'    <span style="{HDR_T}">Stock</span>'
                f'    <span style="{HDR_T}">Value · P&L · Cost</span>'
                f'  </div>'
                + rows_html +
                f'  <div style="{ROW_TOTAL}">'
                f'    <div style="font-size:1.05rem;font-weight:700;">TOTAL</div>'
                f'    <div style="flex:1;text-align:right;">'
                f'      <div style="{MV_S}">${total_mv:,.0f}</div>'
                f'      <div style="{total_pnl_style}">{total_arrow} {total_pct:+.1f}% (${total_pnl:,.0f})</div>'
                f'      <div style="{COST_S}">Cost: ${total_cost:,.0f}</div>'
                f'    </div>'
                f'  </div>'
                f'</div>'
            )

            st.markdown(html, unsafe_allow_html=True)

        else:
            # ═══════════════════════════════════════════════════════════════
            # DESKTOP MODE - Full styled dataframe with options
            # ═══════════════════════════════════════════════════════════════
            view_choice = st.radio(
                "Table Style",
                ["Summary", "Detailed"],
                horizontal=True,
                label_visibility="collapsed"
            )

            df_display = (
                self._prepare_summary_view(df) if view_choice == "Summary"
                else self._prepare_detailed_view(df)
            )

            # Column order
            if view_choice == "Summary":
                col_order = ['Ticker','MV_AUD','PnL_%','PnL_AUD','Cost_AUD','Shares','Current_Price','Avg_Cost']
            else:
                col_order = ['Ticker','Platform','MV_AUD','PnL_%','PnL_AUD','Cost_AUD','Shares','Current_Price','Avg_Cost']

            available  = [c for c in col_order if c in df_display.columns]
            df_display = df_display[available]

            st.dataframe(
                df_display.style
                    .format(self._get_format_dict(view_choice), na_rep="—")
                    .apply(self._highlight_totals, axis=1)
                    .apply(self._color_pnl, subset=['PnL_AUD'] if 'PnL_AUD' in df_display.columns else [], axis=0)
                    .apply(self._color_pnl, subset=['PnL_%']   if 'PnL_%'   in df_display.columns else [], axis=0),
                use_container_width=True,
                height=520,
                column_config={
                    "Ticker":        st.column_config.TextColumn("Stock",    width="small"),
                    "Platform":      st.column_config.TextColumn("Platform", width="small"),
                    "MV_AUD":        st.column_config.TextColumn("Mkt Val",  width="medium"),
                    "PnL_%":         st.column_config.TextColumn("P&L %",   width="small"),
                    "PnL_AUD":       st.column_config.TextColumn("P&L $",   width="medium"),
                    "Cost_AUD":      st.column_config.TextColumn("Cost $",   width="medium"),
                    "Shares":        st.column_config.TextColumn("Shares",   width="small"),
                    "Current_Price": st.column_config.TextColumn("Price",    width="small"),
                    "Avg_Cost":      st.column_config.TextColumn("Avg Cost", width="small"),
                }
            )
    
    def render_insights(self, df: pd.DataFrame, stats: dict):
        """Render portfolio insights - mobile-first layout"""
        st.markdown("---")
        st.subheader("💡 Portfolio Insights")

        equity_df  = df[df['Ticker'] != 'Cash'].copy()
        equity_agg = equity_df.groupby('Ticker').agg(
            MV_AUD=('MV_AUD', 'sum'),
            PnL_AUD=('PnL_AUD', 'sum')
        ).reset_index()

        ev = stats['equity_value'] if stats['equity_value'] > 0 else 1

        # ── Concentration ────────────────────────────────────────────────────
        st.markdown("#### 🎯 Top Holdings")

        top_holdings = equity_agg.nlargest(10, 'MV_AUD').copy()
        top_holdings['% of Stocks'] = (top_holdings['MV_AUD'] / ev * 100).round(1)
        top_holdings['% of Total']  = (top_holdings['MV_AUD'] / stats['total_mv'] * 100).round(1)
        top_holdings['Market Value'] = top_holdings['MV_AUD'].apply(lambda x: f"${x:,.0f}")
        top_holdings['% of Stocks']  = top_holdings['% of Stocks'].apply(lambda x: f"{x:.1f}%")
        top_holdings['% of Total']   = top_holdings['% of Total'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            top_holdings[['Ticker', 'Market Value', '% of Stocks', '% of Total']],
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Ticker":       st.column_config.TextColumn("Stock",        width="small"),
                "Market Value": st.column_config.TextColumn("Value",        width="medium"),
                "% of Stocks":  st.column_config.TextColumn("% Stocks",     width="small"),
                "% of Total":   st.column_config.TextColumn("% Portfolio",  width="small"),
            }
        )

        # ── Concentration metric cards ────────────────────────────────────
        top_3_pct  = equity_agg.nlargest(3,  'MV_AUD')['MV_AUD'].sum() / ev * 100
        top_5_pct  = equity_agg.nlargest(5,  'MV_AUD')['MV_AUD'].sum() / ev * 100
        top_10_pct = equity_agg.nlargest(10, 'MV_AUD')['MV_AUD'].sum() / ev * 100

        CARD  = "background:#f8f9fa;border:1px solid #e0e0e0;border-radius:12px;padding:12px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.07);"
        LABEL = "font-size:0.72rem;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.3px;margin-bottom:4px;"
        VALUE = "font-size:1.15rem;font-weight:700;color:#1a1a1a;line-height:1.2;"
        DELTA = "font-size:0.78rem;font-weight:600;color:#555;margin-top:3px;"
        GRID  = "display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;"

        st.markdown(
            f'<div style="{GRID}">'
            f'<div style="{CARD}"><div style="{LABEL}">Top 3 Holdings</div>'
            f'<div style="{VALUE}">{top_3_pct:.1f}%</div>'
            f'<div style="{DELTA}">of stock portfolio</div></div>'
            f'<div style="{CARD}"><div style="{LABEL}">Top 5 Holdings</div>'
            f'<div style="{VALUE}">{top_5_pct:.1f}%</div>'
            f'<div style="{DELTA}">of stock portfolio</div></div>'
            f'<div style="{CARD}"><div style="{LABEL}">Top 10 Holdings</div>'
            f'<div style="{VALUE}">{top_10_pct:.1f}%</div>'
            f'<div style="{DELTA}">of stock portfolio</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Risk badge
        if top_3_pct > 60:
            st.error("🔴 High concentration — top 3 stocks = over 60% of equities. Consider rebalancing.")
        elif top_3_pct > 50:
            st.warning("🟡 Moderate concentration — top 3 stocks = over 50% of equities.")
        else:
            st.success("🟢 Well diversified — top 3 stocks = under 50% of equities.")

        # ── Cash vs Stocks ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💵 Cash vs Stocks")

        cash_pct   = stats['cash_pct']
        equity_pct = stats['equity_pct']

        st.markdown(
            f'<div style="{GRID}">'
            f'<div style="{CARD}"><div style="{LABEL}">Cash Balance</div>'
            f'<div style="{VALUE}">${stats["cash_value"]:,.0f}</div>'
            f'<div style="{DELTA}">{cash_pct:.1f}% of portfolio</div></div>'
            f'<div style="{CARD}"><div style="{LABEL}">Stock Value</div>'
            f'<div style="{VALUE}">${stats["equity_value"]:,.0f}</div>'
            f'<div style="{DELTA}">{equity_pct:.1f}% of portfolio</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if cash_pct > 30:
            st.warning(f"⚠️ **{cash_pct:.1f}% in cash** — large uninvested balance. Consider deploying into positions.")
        elif cash_pct < 5:
            st.warning(f"⚠️ **Only {cash_pct:.1f}% cash** — very low buying power for new opportunities.")
        else:
            st.success(f"✅ **{cash_pct:.1f}% cash** — healthy reserve with room to invest.")

        # ── Diversification Score ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎲 Diversification Score")

        position_pcts     = equity_agg['MV_AUD'] / ev * 100
        hhi               = (position_pcts ** 2).sum()
        div_score         = ((10000 - hhi) / 10000 * 100)

        if div_score >= 70:
            st.success(f"**{div_score:.0f} / 100 — Excellent 🎯**")
        elif div_score >= 50:
            st.info(f"**{div_score:.0f} / 100 — Good 👍**")
        elif div_score >= 30:
            st.warning(f"**{div_score:.0f} / 100 — Moderate ⚠️**")
        else:
            st.error(f"**{div_score:.0f} / 100 — Concentrated 🔴**")

        st.progress(div_score / 100)
        st.caption("Score based on how evenly spread your equity positions are. 100 = perfectly equal weight.")
    
    def render_strategy_analysis(self, df: pd.DataFrame, stats: dict):
        """Render strategy role allocation vs targets"""
        strategy_data = self.analytics.calculate_strategy_allocation(df, stats)
        
        if not strategy_data:
            return  # No strategy role column in data
        
        st.markdown("---")
        st.subheader("🎯 Strategy Allocation")
        
        role_df = strategy_data['data']
        colors = strategy_data['colors']
        
        # Style constants for inline cards
        CARD_BASE = "background:#f8f9fa;border:2px solid {border};border-radius:12px;padding:14px 16px;margin:8px 0;"
        ROLE_NAME = "font-size:0.85rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;"
        METRIC_ROW = "display:flex;justify-content:space-between;align-items:center;margin:4px 0;"
        LABEL = "font-size:0.78rem;color:#666;font-weight:600;"
        VALUE = "font-size:1.05rem;font-weight:700;color:#1a1a1a;"
        GAP_POS = "font-size:0.85rem;font-weight:600;color:#1a9655;"
        GAP_NEG = "font-size:0.85rem;font-weight:600;color:#dc3545;"
        
        # Render each role as a card
        for role in strategy_data['order']:
            role_row = role_df[role_df.iloc[:, 0] == role]
            
            if role_row.empty:
                continue
            
            current_pct = role_row['Current_%'].values[0]
            target_pct = role_row['Target_%'].values[0]
            gap_pct = role_row['Gap_%'].values[0]
            mv_aud = role_row['MV_AUD'].values[0]
            
            border_color = colors.get(role, '#e0e0e0')
            role_color = colors.get(role, '#666')
            
            gap_style = GAP_POS if gap_pct >= 0 else GAP_NEG
            gap_arrow = "▲" if gap_pct >= 0 else "▼"
            gap_text = f"{gap_arrow} {abs(gap_pct):.1f}% gap"
            
            card_html = (
                f'<div style="{CARD_BASE.format(border=border_color)}">'
                f'  <div style="{ROLE_NAME.format(color=role_color)}">{role}</div>'
                f'  <div style="{METRIC_ROW}">'
                f'    <span style="{LABEL}">Current</span>'
                f'    <span style="{VALUE}">{current_pct:.1f}%</span>'
                f'  </div>'
                f'  <div style="{METRIC_ROW}">'
                f'    <span style="{LABEL}">Target</span>'
                f'    <span style="{VALUE}">{target_pct:.1f}%</span>'
                f'  </div>'
                f'  <div style="{METRIC_ROW}">'
                f'    <span style="{LABEL}">Gap</span>'
                f'    <span style="{gap_style}">{gap_text}</span>'
                f'  </div>'
                f'  <div style="margin-top:8px;padding-top:8px;border-top:1px solid #e0e0e0;">'
                f'    <span style="{LABEL}">Value:</span> '
                f'    <span style="font-size:0.9rem;font-weight:600;color:#1a1a1a;">${mv_aud:,.0f}</span>'
                f'  </div>'
                f'</div>'
            )
            
            st.markdown(card_html, unsafe_allow_html=True)
        
        # Summary at bottom
        total_target = role_df['Target_%'].sum()
        st.caption(f"💡 Total target allocation: {total_target:.1f}% · Unallocated: {100 - total_target:.1f}%")
    
    def _prepare_summary_view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate holdings by ticker"""
        df['Native_Cost_Total'] = df['Shares'] * df['Avg_Cost']
        
        agg = df.groupby('Ticker').agg({
            'Shares': 'sum',
            'Native_Cost_Total': 'sum',
            'Current_Price': 'mean',
            'Cost_AUD': 'sum',
            'MV_AUD': 'sum',
            'PnL_AUD': 'sum'
        }).reset_index()
        
        agg['Avg_Cost'] = agg['Native_Cost_Total'] / agg['Shares']
        agg['PnL_%'] = (agg['PnL_AUD'] / agg['Cost_AUD'] * 100).fillna(0)
        
        agg = agg[['Ticker', 'Shares', 'Avg_Cost', 'Current_Price', 
                   'Cost_AUD', 'MV_AUD', 'PnL_AUD', 'PnL_%']]
        
        return self._add_totals_row(agg)
    
    def _prepare_detailed_view(self, df: pd.DataFrame) -> pd.DataFrame:
        """Show individual positions with platform info"""
        df_view = df[[
            'Ticker', 'Platform', 'Shares', 'Avg_Cost', 
            'Current_Price', 'Cost_AUD', 'MV_AUD', 'PnL_AUD'
        ]].copy()
        
        df_view['PnL_%'] = (df_view['PnL_AUD'] / df_view['Cost_AUD'] * 100).fillna(0)
        
        return self._add_totals_row(df_view)
    
    def _add_totals_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary totals row"""
        df_sorted = df.sort_values('MV_AUD', ascending=False).reset_index(drop=True)
        df_sorted.index += 1

        # Build total row with NaN for non-summable columns (shows "—")
        total_data = {col: [float('nan')] for col in df_sorted.columns}
        total_data['Ticker']  = ['TOTAL']
        total_data['Cost_AUD'] = [df_sorted['Cost_AUD'].sum()]
        total_data['MV_AUD']   = [df_sorted['MV_AUD'].sum()]
        total_data['PnL_AUD']  = [df_sorted['PnL_AUD'].sum()]

        total_row = pd.DataFrame(total_data, index=[''])

        total_cost = total_row['Cost_AUD'].iloc[0]
        if total_cost != 0:
            total_row['PnL_%'] = (total_row['PnL_AUD'] / total_cost * 100)

        return pd.concat([df_sorted, total_row])
    
    @staticmethod
    def _get_format_dict(view_mode: str) -> dict:
        """Get formatting dictionary for dataframe"""
        return {
            'Shares':        "{:,.2f}",
            'Avg_Cost':      "{:,.2f}",
            'Current_Price': "{:,.2f}",
            'Cost_AUD':      "${:,.0f}",
            'MV_AUD':        "${:,.0f}",
            'PnL_AUD':       "${:,.0f}",
            'PnL_%':         "{:+.2f}%",
            'Stop_Loss_Price': "{:,.2f}",
        }
    
    @staticmethod
    def _highlight_totals(row):
        """Highlight totals row"""
        if row.get('Ticker') == 'TOTAL':
            return ['background-color: #f8f9fa; font-weight: 600'] * len(row)
        return [''] * len(row)
    
    @staticmethod
    def _color_pnl(col):
        """Color PnL values (green positive, red negative)"""
        return [
            'color: #28a745' if val > 0 else 'color: #dc3545' if val < 0 else ''
            for val in col
        ]
    
    def render_download(self, df: pd.DataFrame):
        """Render CSV download with complete portfolio details in native currencies"""
        st.markdown("---")
        st.subheader("📥 Download Portfolio")
        
        # Calculate native cost total per ticker (Shares × Avg_Cost in native currency)
        df_with_native = df.copy()
        df_with_native['Native_Cost_Total'] = df_with_native['Shares'] * df_with_native['Avg_Cost']
        
        # Aggregate by ticker
        download_df = df_with_native.groupby('Ticker').agg({
            'Shares': 'sum',
            'Native_Cost_Total': 'sum',
            'Current_Price': 'mean',  # Average current price (should be same across platforms)
            'Currency': 'first',
            'Cost_AUD': 'sum',
            'MV_AUD': 'sum',
            'PnL_AUD': 'sum',
        }).reset_index()
        
        # Calculate weighted average cost in native currency
        download_df['Avg_Cost_Native'] = (download_df['Native_Cost_Total'] / download_df['Shares']).round(2)
        
        # Calculate P&L %
        download_df['PnL_%'] = (download_df['PnL_AUD'] / download_df['Cost_AUD'] * 100).fillna(0).round(2)
        
        # Round numeric columns
        download_df['MV_AUD'] = download_df['MV_AUD'].round(2)
        download_df['Cost_AUD'] = download_df['Cost_AUD'].round(2)
        download_df['PnL_AUD'] = download_df['PnL_AUD'].round(2)
        download_df['Shares'] = download_df['Shares'].round(4)
        download_df['Current_Price'] = download_df['Current_Price'].round(2)
        
        # Create display columns with currency labels
        download_df['Avg_Cost_Display'] = download_df.apply(
            lambda row: f"{row['Avg_Cost_Native']:.2f} {row['Currency']}" 
                        if pd.notna(row['Avg_Cost_Native']) and pd.notna(row['Currency']) 
                        else '',
            axis=1
        )
        download_df['Current_Price_Display'] = download_df.apply(
            lambda row: f"{row['Current_Price']:.2f} {row['Currency']}" 
                        if pd.notna(row['Current_Price']) and pd.notna(row['Currency']) 
                        else '',
            axis=1
        )
        
        # Reorder columns for better readability
        download_df = download_df[[
            'Ticker', 'Currency', 'Shares', 'Avg_Cost_Display', 'Current_Price_Display',
            'Cost_AUD', 'MV_AUD', 'PnL_AUD', 'PnL_%'
        ]]
        
        # Rename for cleaner CSV headers
        download_df.columns = [
            'Ticker', 'Currency', 'Shares', 'Avg Cost', 'Current Price',
            'Cost (AUD)', 'Market Value (AUD)', 'P&L (AUD)', 'P&L %'
        ]
        
        # Sort by market value
        download_df = download_df.sort_values('Market Value (AUD)', ascending=False)
        
        # Add totals row
        total_cost = download_df['Cost (AUD)'].sum().round(2)
        total_mv = download_df['Market Value (AUD)'].sum().round(2)
        total_pnl = download_df['P&L (AUD)'].sum().round(2)
        total_pnl_pct = ((total_pnl / total_cost * 100) if total_cost > 0 else 0).round(2)
        
        total_row = pd.DataFrame({
            'Ticker': ['TOTAL'],
            'Currency': [''],
            'Shares': [''],
            'Avg Cost': [''],
            'Current Price': [''],
            'Cost (AUD)': [total_cost],
            'Market Value (AUD)': [total_mv],
            'P&L (AUD)': [total_pnl],
            'P&L %': [total_pnl_pct]
        })
        
        download_df = pd.concat([download_df, total_row], ignore_index=True)
        
        # Convert to CSV
        csv = download_df.to_csv(index=False).encode('utf-8')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Avg Cost & Current Price in native currency (USD/AUD). All dollar values in AUD."
            )
        
        with col2:
            if st.button("🔄 Refresh Data", use_container_width=True, type="secondary"):
                st.cache_data.clear()
                st.rerun()
        
        st.caption(f"📄 {len(download_df)-1} positions + TOTAL · Data refreshes every 60s · {datetime.now().strftime('%d %b %Y %H:%M')}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    # Setup
    setup_page()
    
    # Initialize dashboard (no authentication)
    dashboard = Dashboard()
    
    try:
        # Load data
        df_raw = dashboard.data_manager.load_portfolio_data()
        
        if df_raw.empty:
            st.error("❌ No portfolio data available")
            st.stop()
        
        # Extract capital and filter
        capital_row = df_raw[df_raw['Ticker'] == 'CAPITAL']
        capital = capital_row['Shares'].sum() if not capital_row.empty else 743564
        
        df_clean = df_raw[df_raw['Ticker'] != 'CAPITAL'].copy()
        
        # Fetch market data
        df_enriched, fx_rate = dashboard.data_manager.fetch_market_data(df_clean)
        
        # Calculate statistics
        stats = dashboard.analytics.calculate_summary_stats(df_enriched, capital)
        
        # ═══════════════════════════════════════════════════════════════════
        # MODE SELECTOR - Let user choose mobile or desktop experience
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("### Display Mode")
        mode = st.radio(
            "Choose your view",
            ["📱 Mobile (Simple & Clean)", "💻 Desktop (Full Dashboard)"],
            horizontal=True,
            help="Mobile mode: Clean cards, simple lists, fast. Desktop mode: Full charts, tables, analytics."
        )
        
        is_mobile = "Mobile" in mode
        
        st.markdown("---")
        
        if is_mobile:
            # ═══════════════════════════════════════════════════════════
            # MOBILE MODE - Simple, clean, essential info only
            # ═══════════════════════════════════════════════════════════
            dashboard.render_header(stats, fx_rate)
            
            # Strategy allocation (simplified for mobile)
            dashboard.render_strategy_analysis(df_enriched, stats)
            
            # Just the essentials
            dashboard.render_holdings_table(df_enriched, force_mobile=True)
            dashboard.render_download(df_enriched)
            
            # Simple insights
            st.markdown("---")
            st.subheader("💡 Quick Insights")
            
            equity_df = df_enriched[df_enriched['Ticker'] != 'Cash'].copy()
            equity_agg = equity_df.groupby('Ticker').agg(
                MV_AUD=('MV_AUD', 'sum')
            ).reset_index()
            
            top_3_pct = equity_agg.nlargest(3, 'MV_AUD')['MV_AUD'].sum() / stats['equity_value'] * 100 if stats['equity_value'] > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💵 Cash", f"${stats['cash_value']:,.0f}", f"{stats['cash_pct']:.1f}%")
            with col2:
                st.metric("🎯 Top 3", f"{top_3_pct:.1f}%", "of stocks")
            
            if top_3_pct > 60:
                st.warning("⚠️ High concentration in top 3 positions")
            elif top_3_pct > 50:
                st.info("ℹ️ Moderate concentration")
            else:
                st.success("✅ Well diversified")
            
            # Refresh button
            st.markdown("---")
            if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.rerun()
        
        else:
            # ═══════════════════════════════════════════════════════════
            # DESKTOP MODE - Full dashboard with all features
            # ═══════════════════════════════════════════════════════════
            dashboard.render_header(stats, fx_rate)
            dashboard.render_charts(df_enriched)
            dashboard.render_strategy_analysis(df_enriched, stats)
            dashboard.render_insights(df_enriched, stats)
            dashboard.render_holdings_table(df_enriched, force_mobile=False)
            dashboard.render_download(df_enriched)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support")

if __name__ == "__main__":
    main()
