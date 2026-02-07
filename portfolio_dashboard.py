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
    CACHE_TTL: int = 600  # 10 minutes
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
    
    # Add viewport meta tag for mobile optimization
    st.markdown("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        /* Modern clean aesthetic */
        .main { 
            background-color: #ffffff !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        h1, h2, h3 { 
            font-weight: 600 !important;
            letter-spacing: -0.5px;
            color: #1a1a1a !important;
        }
        
        /* Fix metric cards - ensure visibility */
        [data-testid="stMetric"] {
            background: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
            border-radius: 12px;
            padding: 1.25rem !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.3s ease;
        }
        
        /* Fix metric values - ensure text is visible */
        [data-testid="stMetric"] label {
            color: #495057 !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1a1a1a !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.875rem !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stMetric"]:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        
        /* Modern buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            border: 1px solid #dee2e6;
            background-color: #ffffff;
            color: #495057 !important;
            transition: all 0.2s ease;
            padding: 0.5rem 1.5rem;
        }
        
        .stButton>button:hover {
            border-color: #2E4053;
            background-color: #2E4053;
            color: white !important;
            box-shadow: 0 4px 12px rgba(46, 64, 83, 0.2);
        }
        
        /* Radio buttons */
        .stRadio > div {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 8px;
        }
        
        /* Fix tab labels */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #495057 !important;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            color: #2E4053 !important;
        }
        
        /* Data tables - fix visibility */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            color: #1a1a1a !important;
        }
        
        .dataframe thead tr th {
            background-color: #f8f9fa !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }
        
        .dataframe tbody tr td {
            color: #1a1a1a !important;
        }
        
        /* Fix text visibility on mobile */
        p, span, div {
            color: #1a1a1a !important;
        }
        
        /* Ensure captions are visible */
        .caption {
            color: #6c757d !important;
        }
        
        /* Hide Streamlit branding */
        footer, #MainMenu { visibility: hidden; }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-live { background-color: #28a745; }
        .status-cached { background-color: #ffc107; }
        
        /* Mobile-specific fixes */
        @media (max-width: 768px) {
            [data-testid="stMetric"] {
                padding: 1rem !important;
            }
            
            [data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 1.25rem !important;
            }
            
            h1 {
                font-size: 1.75rem !important;
            }
            
            h2 {
                font-size: 1.5rem !important;
            }
            
            h3 {
                font-size: 1.25rem !important;
            }
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
        total_pnl = total_mv - total_cost
        pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        cash_value = df[df['Ticker'] == 'Cash']['MV_AUD'].sum()
        equity_value = total_mv - cash_value
        
        # Winner/Loser analysis
        winners = df[(df['PnL_AUD'] > 0) & (df['Ticker'] != 'Cash')]
        losers = df[(df['PnL_AUD'] < 0) & (df['Ticker'] != 'Cash')]
        
        return {
            'total_mv': total_mv,
            'total_cost': total_cost,
            'capital_injected': capital,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'num_positions': len(df[df['Ticker'] != 'Cash']),
            'cash_value': cash_value,
            'cash_pct': (cash_value / total_mv * 100) if total_mv > 0 else 0,
            'equity_value': equity_value,
            'equity_pct': (equity_value / total_mv * 100) if total_mv > 0 else 0,
            'num_winners': len(winners),
            'num_losers': len(losers),
            'winners_value': winners['PnL_AUD'].sum() if not winners.empty else 0,
            'losers_value': losers['PnL_AUD'].sum() if not losers.empty else 0,
            'best_performer': winners.nlargest(1, 'PnL_Pct').iloc[0]['Ticker'] if not winners.empty else 'N/A',
            'best_performer_pct': winners.nlargest(1, 'PnL_Pct').iloc[0]['PnL_Pct'] if not winners.empty else 0,
            'worst_performer': losers.nsmallest(1, 'PnL_Pct').iloc[0]['Ticker'] if not losers.empty else 'N/A',
            'worst_performer_pct': losers.nsmallest(1, 'PnL_Pct').iloc[0]['PnL_Pct'] if not losers.empty else 0,
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
            marker=dict(line=dict(color='white', width=2))
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
            marker=dict(line=dict(color='white', width=2))
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
            marker=dict(line=dict(color='white', width=2))
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
        """Render dashboard header with key metrics"""
        st.title("📊 Portfolio Command Center")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Primary metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Value",
                f"${stats['total_mv']:,.0f}",
                help="Current market value in AUD"
            )
        
        with col2:
            delta_color = "normal" if stats['total_pnl'] >= 0 else "inverse"
            st.metric(
                "Total P&L",
                f"${stats['total_pnl']:,.0f}",
                f"{stats['pnl_pct']:.2f}%",
                delta_color=delta_color,
                help="Total profit or loss"
            )
        
        with col3:
            st.metric(
                "Capital Injected",
                f"${stats['capital_injected']:,.0f}",
                help="Total capital invested"
            )
        
        with col4:
            st.metric(
                "Positions",
                f"{stats['num_positions']}",
                f"{stats['cash_pct']:.1f}% cash",
                help="Number of holdings"
            )
        
        with col5:
            st.metric(
                "AUD/USD",
                f"{fx_rate:.4f}",
                help="Current exchange rate"
            )
        
        # Secondary metrics row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Winners",
                f"{stats['num_winners']}",
                f"+${stats['winners_value']:,.0f}",
                delta_color="normal",
                help="Profitable positions"
            )
        
        with col2:
            st.metric(
                "Losers", 
                f"{stats['num_losers']}",
                f"${stats['losers_value']:,.0f}",
                delta_color="inverse",
                help="Loss-making positions"
            )
        
        with col3:
            st.metric(
                "Best Performer",
                stats['best_performer'],
                f"+{stats['best_performer_pct']:.1f}%",
                delta_color="normal",
                help="Top gainer"
            )
        
        with col4:
            st.metric(
                "Worst Performer",
                stats['worst_performer'],
                f"{stats['worst_performer_pct']:.1f}%",
                delta_color="inverse",
                help="Biggest loser"
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
    
    def render_holdings_table(self, df: pd.DataFrame):
        """Render detailed holdings table with view options"""
        st.markdown("---")
        st.subheader("📋 Holdings Detail")
        
        # View mode selector
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            view_mode = st.radio(
                "View",
                ["Summary", "Detailed"],
                horizontal=True,
                label_visibility="collapsed"
            )
        
        # Prepare data based on view mode
        if view_mode == "Summary":
            df_display = self._prepare_summary_view(df)
        else:
            df_display = self._prepare_detailed_view(df)
        
        # Render table
        st.dataframe(
            df_display.style
            .format(self._get_format_dict(view_mode), na_rep="—")
            .apply(self._highlight_totals, axis=1)
            .apply(self._color_pnl, subset=['PnL_AUD'], axis=0)
            .apply(self._color_pnl, subset=['PnL_%'], axis=0),
            use_container_width=True,
            height=500
        )
    
    def render_insights(self, df: pd.DataFrame, stats: dict):
        """Render portfolio insights and risk metrics"""
        st.markdown("---")
        st.subheader("💡 Portfolio Insights")
        
        equity_df = df[df['Ticker'] != 'Cash'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Concentration Analysis")
            
            # Top 3 holdings concentration
            top_3_value = equity_df.nlargest(3, 'MV_AUD')['MV_AUD'].sum()
            top_3_pct = (top_3_value / stats['equity_value'] * 100) if stats['equity_value'] > 0 else 0
            
            # Top 5 holdings concentration
            top_5_value = equity_df.nlargest(5, 'MV_AUD')['MV_AUD'].sum()
            top_5_pct = (top_5_value / stats['equity_value'] * 100) if stats['equity_value'] > 0 else 0
            
            # Top 10 holdings concentration
            top_10_value = equity_df.nlargest(10, 'MV_AUD')['MV_AUD'].sum()
            top_10_pct = (top_10_value / stats['equity_value'] * 100) if stats['equity_value'] > 0 else 0
            
            concentration_data = pd.DataFrame({
                'Category': ['Top 3 Holdings', 'Top 5 Holdings', 'Top 10 Holdings'],
                'Percentage': [top_3_pct, top_5_pct, top_10_pct]
            })
            
            fig = px.bar(
                concentration_data,
                x='Category',
                y='Percentage',
                text='Percentage',
                color='Percentage',
                color_continuous_scale=['#48C9B0', '#F5B041', '#EC7063']
            )
            
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=20, b=20, l=0, r=0),
                yaxis_title="Portfolio %",
                xaxis_title="",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Concentration warnings
            if top_3_pct > 50:
                st.warning("⚠️ Top 3 positions represent >50% of portfolio - high concentration risk")
            elif top_3_pct > 40:
                st.info("ℹ️ Moderate concentration in top 3 positions")
        
        with col2:
            st.markdown("#### 📊 Risk Metrics")
            
            # Calculate volatility proxy (based on position sizes)
            position_sizes = equity_df['MV_AUD'] / stats['equity_value'] * 100 if stats['equity_value'] > 0 else 0
            
            # Diversification metrics
            num_positions = len(equity_df)
            avg_position_size = position_sizes.mean() if not equity_df.empty else 0
            max_position_size = position_sizes.max() if not equity_df.empty else 0
            
            # Herfindahl-Hirschman Index (HHI) for concentration
            hhi = (position_sizes ** 2).sum() if not equity_df.empty else 0
            
            metrics_df = pd.DataFrame({
                'Metric': ['Number of Positions', 'Avg Position Size', 'Largest Position', 'HHI (Concentration)'],
                'Value': [f"{num_positions}", f"{avg_position_size:.1f}%", f"{max_position_size:.1f}%", f"{hhi:.0f}"]
            })
            
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("**HHI Guide:** <1500 = Diversified | 1500-2500 = Moderate | >2500 = Concentrated")
            
            # Risk assessment
            if hhi > 2500:
                st.error("🔴 High concentration - consider diversifying")
            elif hhi > 1500:
                st.warning("🟡 Moderate concentration")
            else:
                st.success("🟢 Well diversified portfolio")
            
            # Allocation warnings
            if stats['cash_pct'] > 30:
                st.info(f"💰 {stats['cash_pct']:.1f}% cash - consider deploying capital")
            elif stats['cash_pct'] < 5:
                st.warning(f"⚠️ Only {stats['cash_pct']:.1f}% cash - limited dry powder")
    
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
        
        total_row = pd.DataFrame({
            'Ticker': ['TOTAL'],
            'Cost_AUD': [df_sorted['Cost_AUD'].sum()],
            'MV_AUD': [df_sorted['MV_AUD'].sum()],
            'PnL_AUD': [df_sorted['PnL_AUD'].sum()],
        }, index=[''])
        
        total_cost = total_row['Cost_AUD'].iloc[0]
        if total_cost != 0:
            total_row['PnL_%'] = (total_row['PnL_AUD'] / total_cost * 100)
        
        return pd.concat([df_sorted, total_row])
    
    @staticmethod
    def _get_format_dict(view_mode: str) -> dict:
        """Get formatting dictionary for dataframe"""
        base_format = {
            'Avg_Cost': "{:.2f}",
            'Current_Price': "{:.2f}",
            'Cost_AUD': "${:,.0f}",
            'MV_AUD': "${:,.0f}",
            'PnL_AUD': "${:,.0f}",
            'PnL_%': "{:+.2f}%"
        }
        
        if view_mode == "Detailed":
            base_format['Stop_Loss_Price'] = "{:.2f}"
        
        return base_format
    
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
    
    def render_actions(self):
        """Render action buttons"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📥 Export CSV", use_container_width=True):
                st.info("Export feature coming soon")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    # Setup
    setup_page()
    
    # Authentication
    if not AuthManager.check_password():
        st.stop()
    
    # Initialize dashboard
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
        
        # Render UI
        dashboard.render_header(stats, fx_rate)
        dashboard.render_charts(df_enriched)
        dashboard.render_insights(df_enriched, stats)
        dashboard.render_holdings_table(df_enriched)
        dashboard.render_actions()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support")

if __name__ == "__main__":
    main()
