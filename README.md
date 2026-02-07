# Portfolio Command Center 📊

A production-ready, enterprise-grade portfolio tracking dashboard built with Streamlit. Designed with Silicon Valley engineering standards: clean architecture, comprehensive error handling, and modern UX.

## ✨ Key Features

- **Real-time Market Data**: Live price updates via Yahoo Finance API
- **Multi-Currency Support**: Automatic AUD/USD conversion
- **Interactive Visualizations**: Modern charts with Plotly
- **Secure Access**: Password-protected dashboard
- **Responsive Design**: Optimized for desktop and tablet
- **Performance**: Intelligent caching for fast load times
- **Error Resilience**: Graceful degradation when APIs fail

## 🏗️ Architecture

```
portfolio_dashboard.py
├── Configuration Layer (DashboardConfig)
├── Authentication (AuthManager)
├── Data Layer (DataManager)
│   ├── Portfolio data loading
│   └── Market data fetching
├── Analytics (PortfolioAnalytics)
│   ├── Summary statistics
│   └── Data aggregation
├── Visualization (ChartBuilder)
│   ├── Pie charts
│   └── Bar charts
└── UI Layer (Dashboard)
    ├── Header with metrics
    ├── Chart section
    ├── Holdings table
    └── Action buttons
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone or download the repository
cd your-project-folder

# Install dependencies
pip install -r requirements.txt

# Set up secrets
mkdir .streamlit
echo 'PASSWORD = "your-secure-password"' > .streamlit/secrets.toml
```

### Running the Dashboard

```bash
streamlit run portfolio_dashboard.py
```

Navigate to `http://localhost:8501` in your browser.

## 📋 Data Requirements

The dashboard expects a Google Sheets CSV export with these columns:

**Required:**
- `Ticker`: Stock symbol or asset identifier
- `Shares`: Number of shares held
- `Avg_Cost`: Average purchase price

**Optional:**
- `Platform`: Trading platform/broker name
- `Currency`: USD or AUD
- `Stop_Loss_Price`: Risk management level
- `Target_Weight`: Portfolio allocation target
- `Sector`: Asset classification
- `Strategy_Role`: Investment strategy category

**Special Rows:**
- `CAPITAL`: Ticker value indicating total invested capital
- `Cash`: Ticker value for cash positions

## 🔧 Configuration

### Environment Variables

Create `.streamlit/secrets.toml`:

```toml
PASSWORD = "your-secure-password"
```

### Customization

Edit `DashboardConfig` in `portfolio_dashboard.py`:

```python
@dataclass
class DashboardConfig:
    SHEET_URL: str = "your-google-sheets-csv-url"
    DEFAULT_FX_RATE: float = 0.70
    CACHE_TTL: int = 600  # seconds
    PIE_THRESHOLD: float = 0.85  # for "Others" grouping
```

## 📊 Features Deep Dive

### Authentication
- Password-protected access
- Session state management
- Secure credential storage

### Data Pipeline
- **Loading**: Google Sheets CSV import with validation
- **Enrichment**: Live price data from Yahoo Finance
- **Calculation**: Automatic P&L, FX conversion, percentages
- **Caching**: 10-minute cache for performance

### Analytics
- Portfolio-level summary statistics
- Position aggregation (summary vs. detailed)
- Intelligent grouping for visualizations
- P&L tracking with percentage returns

### Visualizations
- **Asset Allocation**: Includes cash positions
- **Equity Distribution**: Stock holdings only
- **Holdings Table**: Summary or detailed view
- Modern color schemes and hover effects

### Error Handling
- Graceful fallbacks for API failures
- Data validation with clear error messages
- Logging for debugging
- User-friendly error notifications

## 🎨 UI/UX Highlights

- **Modern Design**: Clean, minimalist aesthetic
- **Responsive Metrics**: Hover effects and animations
- **Color Coding**: Green/red P&L indicators
- **Smart Grouping**: Small positions aggregated in charts
- **Mobile-Friendly**: Responsive column layouts

## 🔒 Security Best Practices

1. **Never commit secrets**: Use `.gitignore` for `.streamlit/secrets.toml`
2. **Use environment variables**: For production deployments
3. **HTTPS only**: Deploy behind SSL certificate
4. **Regular updates**: Keep dependencies current

## 🚢 Deployment

### Streamlit Cloud

1. Push to GitHub (exclude secrets)
2. Connect repository to Streamlit Cloud
3. Add secrets in dashboard settings
4. Deploy!

### Docker (Alternative)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "portfolio_dashboard.py"]
```

## 📈 Performance

- **Initial Load**: ~2-3 seconds (with caching)
- **Refresh**: <1 second (cached data)
- **Market Data Sync**: ~3-5 seconds (API dependent)
- **Memory**: ~100-200 MB typical

## 🐛 Troubleshooting

### "Failed to load portfolio data"
- Check Google Sheets URL is public/accessible
- Verify CSV export format
- Check required columns exist

### "Using cached prices"
- Yahoo Finance API may be rate-limited
- Check internet connection
- Verify ticker symbols are valid

### Authentication not working
- Ensure `.streamlit/secrets.toml` exists
- Check PASSWORD matches exactly
- Restart Streamlit server

## 🛠️ Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Docstrings for all classes/functions
- Maximum line length: 88 characters

### Testing
```bash
# Run with debug logging
streamlit run portfolio_dashboard.py --logger.level=debug
```

### Adding Features

1. Extend `DashboardConfig` for new settings
2. Add methods to appropriate class (separation of concerns)
3. Update UI in `Dashboard` class
4. Test with various data scenarios

## 📝 Changelog

### Version 2.0 (Current)
- ✅ Complete architecture refactor
- ✅ Comprehensive error handling
- ✅ Modern UI with animations
- ✅ Improved performance (caching)
- ✅ Better code organization
- ✅ Enhanced security
- ✅ Full logging support

### Version 1.0 (Legacy)
- Basic portfolio tracking
- Simple authentication
- Manual calculations

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write clean, documented code
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 💡 Tips

- **Refresh regularly**: Click "Refresh Data" for latest prices
- **Monitor cache**: Data updates every 10 minutes automatically
- **Export data**: Use CSV export for record-keeping
- **Mobile access**: Works on tablets and large phones
- **Keyboard shortcuts**: Streamlit supports `R` to rerun

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review logs in terminal
- Contact system administrator

---

**Built with ❤️ using Streamlit, Plotly, and modern Python practices**
