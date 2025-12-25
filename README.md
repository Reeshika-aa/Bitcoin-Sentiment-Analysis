# Bitcoin-Sentiment-Analysis

#  Bitcoin Sentiment Trading Dashboard

Analyze crypto trading performance by correlating market sentiment (Fear & Greed Index) with profitability.

##  Quick Start

```bash
pip install streamlit pandas plotly numpy
streamlit run crypto_dashboard_bugfix.py
```

##  Required Files

Place in same directory:
- `historical_data.csv` - Trading history (Timestamp IST, Closed PnL, Side, Symbol)
- `fear_greed_index.csv` - Sentiment data (date, value, classification)

##  Features

- **Win Rate & Risk/Reward**: Win%, Profit Factor, Expectancy
- **Time Analysis**: Golden Hour, best day of week, monthly performance
- **Holding Time**: Optimal trade duration (Scalp, Day, Swing, Long-term)
- **Sentiment Correlation**: Which mood makes money? (Fear vs Greed)
- **Top Events**: Top 3 crashes & spikes with verified causes
- **What-If Simulator**: Test hypothetical strategies
- **Strategy Insights**: Long vs Short performance by sentiment

##  Key Sections

1. Problem & Context
2. Trading History Overview
3. US Election Case Study (Nov 2024)
4. Post-Election Crashes & Spikes
5. Strategy Analysis (Profitability, Long/Short, Risk)
6. Win Rate & Success Metrics
7. Time-Based Performance (Hourly, Daily, Monthly, Heatmap)
8. Holding Time Analysis
9. Sentiment Correlation & What-If Scenarios

##  Performance Tips

- Use sidebar filters to reduce data
- Analyze specific date ranges
- Expected load: 5-15 seconds for 10K-50K trades

##  Troubleshooting

**CSV not found?** Check filenames match exactly
**Slow loading?** Use sidebar filters or shorter date range
**Data errors?** Verify date formats: `DD-MM-YYYY HH:MM` (historical) and `YYYY-MM-DD` (sentiment)

##  Key Metrics

- **Win Rate**: % of profitable trades
- **Profit Factor**: Total profits ÷ total losses (>1 = profitable)
- **Expectancy**: Expected $ per trade (>0 = positive edge)
- **Golden Hour**: Most profitable hour of day
- **Volatility**: Risk/unpredictability of results

##  Disclaimer

Educational purposes only. Past performance ≠ future results. 



Built with Streamlit, Plotly, Pandas | Data: Hyperliquid + Alternative.me Fear & Greed Index
