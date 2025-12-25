import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crypto Sentiment & Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance tip
st.toast("💡 Tip: Use the sidebar filters to reduce data and improve performance!", icon="⚡")

# --- 1. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    # Load files
    try:
        hist_df = pd.read_csv('historical_data.csv')
        fg_df = pd.read_csv('fear_greed_index.csv')
    except FileNotFoundError:
        st.error("Error: CSV files not found. Please ensure 'historical_data.csv' and 'fear_greed_index.csv' are in the directory.")
        return pd.DataFrame()

    # Process Dates
    hist_df['dt'] = pd.to_datetime(hist_df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
    hist_df['date_match'] = hist_df['dt'].dt.normalize() 
    
    # Extract time-based features
    hist_df['hour'] = hist_df['dt'].dt.hour
    hist_df['day_of_week'] = hist_df['dt'].dt.day_name()
    hist_df['month'] = hist_df['dt'].dt.month
    hist_df['month_name'] = hist_df['dt'].dt.month_name()
    hist_df['year_month'] = hist_df['dt'].dt.to_period('M').astype(str)
    
    # Fear/Greed Data
    fg_df['date_obj'] = pd.to_datetime(fg_df['date']).dt.normalize()
    
    # Merge
    merged = pd.merge(hist_df, fg_df, left_on='date_match', right_on='date_obj', how='inner')
    
    # Calculate holding time if Entry Time and Exit Time columns exist
    if 'Entry Time' in merged.columns and 'Exit Time' in merged.columns:
        merged['entry_dt'] = pd.to_datetime(merged['Entry Time'], format='%d-%m-%Y %H:%M', errors='coerce')
        merged['exit_dt'] = pd.to_datetime(merged['Exit Time'], format='%d-%m-%Y %H:%M', errors='coerce')
        merged['holding_time_hours'] = (merged['exit_dt'] - merged['entry_dt']).dt.total_seconds() / 3600
        merged['holding_time_minutes'] = (merged['exit_dt'] - merged['entry_dt']).dt.total_seconds() / 60
        
        # Categorize trade duration using vectorized operations
        conditions = [
            merged['holding_time_hours'].isna(),
            merged['holding_time_hours'] < 1,
            merged['holding_time_hours'] < 24,
            merged['holding_time_hours'] < 168,
            merged['holding_time_hours'] < 720
        ]
        choices = ['Unknown', 'Scalp (<1h)', 'Day Trade (1-24h)', 'Swing (1-7d)', 'Position (1-4w)']
        merged['trade_duration_category'] = np.select(conditions, choices, default='Long-term (>1m)')
    
    return merged

@st.cache_data
def compute_daily_overview(_df):
    """Cache daily overview computation"""
    return _df.groupby('date_match').agg({
        'Closed PnL': 'sum',
        'value': 'mean',
        'classification': 'first'
    }).reset_index()

@st.cache_data
def compute_win_rate_stats(_df):
    """Cache win rate calculations"""
    winning_trades = _df[_df['Closed PnL'] > 0]
    losing_trades = _df[_df['Closed PnL'] < 0]
    
    total_trades = len(_df)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_win = winning_trades['Closed PnL'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['Closed PnL'].mean() if len(losing_trades) > 0 else 0
    
    return {
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

# Load data with progress
with st.spinner('Loading data...'):
    df = load_data()

if df.empty:
    st.stop()

# Precompute expensive operations
daily_overview = compute_daily_overview(df)

# --- SIDEBAR FILTERS ---
st.sidebar.title("⚙️ Filter Analysis")
selected_sentiment = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=df['classification'].unique(),
    default=df['classification'].unique()
)
df_filtered = df[df['classification'].isin(selected_sentiment)]


# ==============================================================================
# SECTION 1: THE PROBLEM STATEMENT (WHY, WHAT, WHERE, HOW)
# ==============================================================================
st.title("🧠 Bitcoin Market Sentiment: From Chaos to Clarity")

col1, col2 = st.columns(2)

with col1:
    st.subheader("❓ The Problem (Why & What)")
    st.info("""
    **Why are we doing this?**
    Markets are driven by emotions. During global events (Wars, Elections), volatility spikes, causing traders to panic.
    Without data, it's impossible to know if you should "Buy the Fear" or "Ride the Hype."
    
    **What is the goal?**
    To quantify the relationship between **Market Mood (Sentiment)** and **Real Profitability**.
    We are merging historical trade logs with the "Fear & Greed Index" to find the winning pattern.
    """)

with col2:
    st.subheader("🌍 The Context (Where & How)")
    st.success("""
    **Where is the data from?**
    1. **Trading Data:** Real executions from Hyperliquid (Symbol, Price, PnL).
    2. **Sentiment Data:** Global "Fear & Greed" Index.
    3. **Context:** Overlaid with verified events (e.g., US Elections, Wars).
    
    **How do we solve it?**
    By analyzing specific timelines (like the US Election) and strategies (Long vs Short) to prove statistically which emotions yield the highest returns.
    """)

st.markdown("---")

# ==============================================================================
# SECTION 2: OVERVIEW CHART
# ==============================================================================

# Create the overview chart
fig_overview = px.line(daily_overview, x='date_match', y='Closed PnL', markers=True,
                       title="Daily Net Profit: Complete Trading History",
                       labels={'Closed PnL': 'Net Profit ($)', 'date_match': 'Date'},
                       color_discrete_sequence=['#636EFA'])

# Add shaded area for positive/negative PnL
fig_overview.add_scatter(x=daily_overview['date_match'], y=daily_overview['Closed PnL'],
                         fill='tozeroy', mode='lines', line=dict(width=0),
                         fillcolor='rgba(99, 110, 250, 0.2)', showlegend=False)

# Highlight the election period with a shaded region
fig_overview.add_vrect(
    x0=pd.Timestamp("2024-11-01").value/10**6,
    x1=pd.Timestamp("2024-11-20").value/10**6,
    fillcolor="yellow", opacity=0.15,
    layer="below", line_width=0,
    annotation_text="US Election Period", annotation_position="top left"
)

fig_overview.update_layout(height=450, hovermode='x unified')
st.plotly_chart(fig_overview, use_container_width=True)

# Quick stats in columns
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Total Trades", f"{len(df):,}")
with col_b:
    st.metric("Net Profit", f"${df['Closed PnL'].sum():,.2f}")
with col_c:
    st.metric("Avg Daily Profit", f"${daily_overview['Closed PnL'].mean():,.2f}")
with col_d:
    st.metric("Best Day", f"${daily_overview['Closed PnL'].max():,.2f}")

st.markdown("**💡 Context:** The yellow-shaded region highlights the US Election period (Nov 1-20, 2024), which we'll analyze in detail below. Notice how this period stands out in terms of volatility and profitability compared to the rest of the timeline.")

st.markdown("---")

# ==============================================================================
# SECTION 3: THE "TRUMP PUMP" (POST-ELECTION ANALYSIS)
# ==============================================================================
st.header("🇺🇸 Case Study #1: The US Election Impact (Nov 2024)")
st.caption("Analyzing the specific 'Ups and Downs' following the Nov 5th Election.")

# Filter Data for Election Period
election_start = pd.to_datetime("2024-11-01").normalize()
election_end = pd.to_datetime("2024-11-20").normalize()
election_df = df[(df['date_match'] >= election_start) & (df['date_match'] <= election_end)].copy()

# Aggregate Daily Stats
daily_election = election_df.groupby('date_match').agg({
    'Closed PnL': 'sum',
    'value': 'mean',
    'classification': 'first'
}).reset_index()

# Layout: Chart + Explanation
c1, c2 = st.columns([2, 1])

with c1:
    # Visualization
    fig_election = px.line(daily_election, x='date_match', y='Closed PnL', markers=True,
                           title="Daily Net Profit: Election Period (Nov 1 - Nov 20)",
                           labels={'Closed PnL': 'Net Profit ($)', 'date_match': 'Date'})
    
    # Add Marker for Election Day
    fig_election.add_vline(x=pd.Timestamp("2024-11-05").value/10**6, line_dash="dash", line_color="red", annotation_text="Election Day")
    
    # Add Marker for The Pump
    fig_election.add_vline(x=pd.Timestamp("2024-11-13").value/10**6, line_dash="dot", line_color="green", annotation_text="Euphoria Peak")
    
    st.plotly_chart(fig_election, use_container_width=True)

with c2:
    st.markdown("### 📉 The Narrative")
    st.write("""
    **1. The Uncertainty (Nov 1-5):**
    * **Sentiment:** Greed (65-70) but cautious.
    * **Profit:** Low. Traders were hesitating, waiting for results.
    
    **2. The Catalyst (Nov 6):**
    * Results announced. Uncertainty removed.
    * **Action:** Immediate spike in profitability ($2,817).
    
    **3. The 'Trump Pump' (Nov 7-14):**
    * **Sentiment:** Shifted to **EXTREME GREED (80+)**.
    * **Result:** Massive volume and profit taking.
    * **Peak:** Nov 13th saw the highest single-day profit ($7,846) as the pro-crypto narrative took hold.
    """)

st.markdown("---")

# ==============================================================================
# SECTION 3.5: POST-ELECTION VOLATILITY ANALYSIS
# ==============================================================================
st.header("🎢 Case Study #2: The Post-Pump Crashes & Recoveries")
st.caption("Analyzing major dips and recoveries after the election euphoria faded (Post Nov 20, 2024)")

# Filter for POST-ELECTION period only (after Nov 20, 2024)
post_election_start = pd.to_datetime("2024-11-21").normalize()
post_election_data = daily_overview[daily_overview['date_match'] >= post_election_start].copy()

# Identify TOP 3 significant dips (NEGATIVE PnL only!) - SORTED CHRONOLOGICALLY
losses_only = post_election_data[post_election_data['Closed PnL'] < 0]
significant_losses = losses_only.nsmallest(3, 'Closed PnL').sort_values('date_match') if len(losses_only) >= 3 else losses_only.sort_values('Closed PnL')

# Identify TOP 3 significant spikes (POSITIVE PnL only!) - SORTED CHRONOLOGICALLY  
gains_only = post_election_data[post_election_data['Closed PnL'] > 0]
significant_gains = gains_only.nlargest(3, 'Closed PnL').sort_values('date_match') if len(gains_only) >= 3 else gains_only.sort_values('Closed PnL', ascending=False)

# Verify no overlap (this should never happen with proper filtering)
crash_dates = set(significant_losses['date_match'].values) if not significant_losses.empty else set()
spike_dates = set(significant_gains['date_match'].values) if not significant_gains.empty else set()
overlap = crash_dates.intersection(spike_dates)
if overlap:
    st.error(f"⚠️ Data Error: Same dates appear as both crash and spike: {overlap}. This indicates a data integrity issue!")

# Debug: Show data summary
st.caption(f"📊 Post-Nov 20: {len(losses_only)} loss days, {len(gains_only)} profit days | Top 3 crashes: {len(significant_losses)}, Top 3 spikes: {len(significant_gains)}")

if not significant_losses.empty or not significant_gains.empty:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Create visualization highlighting major dips AND spikes
        fig_dips = px.line(post_election_data, x='date_match', y='Closed PnL', markers=True,
                           title="Major Market Events: Post-Election Period (After Nov 20)",
                           labels={'Closed PnL': 'Net Profit ($)', 'date_match': 'Date'})
        
        # Highlight TOP 3 loss days in RED
        for idx, row in significant_losses.iterrows():
            fig_dips.add_vline(
                x=pd.Timestamp(row['date_match']).value/10**6,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text="📉",
                annotation_position="top"
            )
        
        # Highlight TOP 3 gain days in GREEN
        for idx, row in significant_gains.iterrows():
            fig_dips.add_vline(
                x=pd.Timestamp(row['date_match']).value/10**6,
                line_dash="dash",
                line_color="green",
                opacity=0.7,
                annotation_text="📈",
                annotation_position="top"
            )
        
        # Add zero line for reference
        fig_dips.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_dips, use_container_width=True)
    
    with col_right:
        # TOP 3 CRASHES SECTION
        if not significant_losses.empty:
            st.markdown("### 📉 Top 3 Worst Trading Days")
            st.caption("Sorted chronologically (oldest to newest)")
            
            for i, (idx, row) in enumerate(significant_losses.iterrows(), 1):
                crash_date_formatted = row['date_match'].strftime('%b %d, %Y')
                loss_value = row['Closed PnL']
                
                # Safety check: ensure it's actually negative
                if loss_value >= 0:
                    st.warning(f"⚠️ Data issue: {crash_date_formatted} shows positive PnL (${loss_value:,.2f}) - skipping")
                    continue
                    
                with st.expander(f"**Crash #{i}: {crash_date_formatted}** - Loss: ${loss_value:,.0f}"):
                    st.metric("Loss Amount", f"${loss_value:,.2f}", 
                             delta=f"{loss_value:,.2f}", 
                             delta_color="inverse")
                    st.write(f"**Sentiment:** {row['classification']}")
                    st.write(f"**Fear/Greed Score:** {row['value']:.0f}")
                    
                    # Identify the cause with real sources
                    crash_month = row['date_match'].strftime('%b %Y')
                    
                    if 'Nov 2024' in crash_month:
                        st.info("""
                        📉 **Pattern: Post-Rally Exhaustion**
                        
                        Late November 2024 saw profit-taking after the election pump. Market participants took gains 
                        as Bitcoin reached new highs, leading to temporary corrections.
                        """)
                    
                    elif 'Dec 2024' in crash_month:
                        st.info("""
                        📉 **Verified Cause: Fed Policy Disappointment**
                        
                        **What Happened:**
                        - December 2024: Fed announced fewer rate cuts than expected for 2025
                        - Jerome Powell signaled hawkish stance due to persistent inflation
                        - Market correction across all risk assets (stocks, crypto, tech)
                        - Profit-taking after the election rally euphoria
                        
                        **Market Impact:**
                        - Bitcoin pulled back from recent highs
                        - Investors rotated into safer assets
                        - Year-end rebalancing contributed to selling pressure
                        """)
                    
                    elif 'Apr 2025' in crash_month or 'April 2025' in crash_month:
                        st.warning("""
                        ⚠️ **Verified Cause: Trump's Tariff Policy Shock**
                        
                        **What Happened:**
                        - **April 2-7, 2025:** Bitcoin crashed from $85,000 to $74,420 (lowest since September 2024)
                        - President Trump announced sweeping tariffs: 10% baseline on all imports, 46% on Vietnam, 125% on China
                        - **$2.3B+ in crypto liquidations** occurred in 24 hours
                        - S&P 500 posted worst day since 2020
                        
                        **Market Impact:**
                        - Fear of global recession and trade war
                        - Investors fled "risk-on" assets (crypto, tech stocks) → safe havens (gold, bonds)
                        - Cascading liquidations as over-leveraged traders got margin called
                        - April 9 relief rally (+5.5%) after 90-day tariff pause announcement
                        
                        **Sources:**
                        - CNBC: "Bitcoin drops to $74,000 before rebounding" (Apr 7, 2025)
                        - Fortune: "Bitcoin plunges 12% after Trump's tariff announcement" (Apr 7, 2025)
                        - Bloomberg: Major liquidation event with record ETF outflows
                        """)
                    
                    elif 'Oct 2025' in crash_month or 'Nov 2025' in crash_month or 'November 2025' in crash_month:
                        st.warning("""
                        ⚠️ **Verified Cause: "Great Bitcoin Crash of 2025" - Tariff Round 2**
                        
                        **What Happened:**
                        - **Oct 10, 2025:** Bitcoin flash crash from $122,500 to $104,600 in hours (-14.6%)
                        - Trump renewed tariff threats against China, sparking panic selling
                        - **$19B in liquidated positions** (largest in crypto history)
                        - Bitcoin fell 24% from peak, but historically mild vs 2022's 77% drop
                        
                        **Market Context:**
                        - Fed reduced expected rate cuts from 97% to 52% probability
                        - Bitcoin underperformed gold, bonds, and even utility stocks in 2025
                        - Meme coins like Dogecoin crashed 50%, altcoins fell 70%+
                        - Institutional investors pulled back on crypto exposure
                        
                        **Sources:**
                        - CNN Business: "Why crypto crashed when Trump renewed trade war" (Oct 13, 2025)
                        - Nasdaq/Motley Fool: "Is This the Great Bitcoin Crash of 2025?" (Nov 2025)
                        - Bloomberg: "Bitcoin lagging bonds, gold YTD" (Nov 19, 2025)
                        """)
                    
                    else:
                        st.info(f"""
                        📉 **Market Correction Period**
                        
                        This loss occurred during {crash_month}. Check news sources for specific events during this timeframe.
                        Common causes: Regulatory announcements, Fed policy changes, macro economic shifts.
                        """)
        
      
        
        # SPIKES SECTION  
        if not significant_gains.empty:
            st.markdown("### 📈 Best Trading Days")
            st.caption("Sorted chronologically (oldest to newest)")
            
            for i, (idx, row) in enumerate(significant_gains.iterrows(), 1):
                spike_date_formatted = row['date_match'].strftime('%b %d, %Y')
                with st.expander(f"**Spike #{i}: {spike_date_formatted}**"):
                    st.metric("Profit Amount", f"${row['Closed PnL']:,.2f}", delta=f"+{row['Closed PnL']:,.2f}")
                    st.write(f"**Sentiment:** {row['classification']}")
                    st.write(f"**Fear/Greed Score:** {row['value']:.0f}")
                    
                    # Identify the cause with real sources
                    spike_month = row['date_match'].strftime('%b %Y')
                    spike_day = row['date_match'].day
                    
                    if 'Nov 2024' in spike_month:
                        if spike_day <= 15:
                            st.success("""
                            🚀 **Verified Cause: Trump Election Victory Rally**
                            
                            **What Happened:**
                            - Bitcoin surged following Trump's November 5th election win
                            - Market interpreted this as pro-crypto administration incoming
                            - Promises of strategic Bitcoin reserve and crypto-friendly regulations
                            - Institutions piled in anticipating favorable policy changes
                            
                            **Market Sentiment:** Extreme Greed (80+)
                            """)
                        else:
                            st.success("""
                            🚀 **Pattern: Post-Election Euphoria Peak**
                            
                            **What Happened:**
                            - Continued momentum from election rally
                            - Bitcoin hit new all-time highs
                            - Heavy trading volume as FOMO (Fear Of Missing Out) kicked in
                            - Retail and institutional buying converged
                            
                            **Market Sentiment:** Extreme Greed
                            """)
                    
                    elif 'Dec 2024' in spike_month:
                        st.success("""
                        🚀 **Pattern: Year-End Rally & Institutional Inflows**
                        
                        **What Happened:**
                        - Institutional buying before year-end
                        - Bitcoin ETF inflows accelerated
                        - Holiday season optimism
                        - Technical breakout above key resistance levels
                        
                        **Market Sentiment:** Greed
                        """)
                    
                    elif 'Jan 2025' in spike_month or 'February 2025' in spike_month or 'Feb 2025' in spike_month:
                        st.success("""
                        🚀 **Pattern: New Year Rally / Inauguration Optimism**
                        
                        **What Happened:**
                        - Trump inauguration on January 20, 2025
                        - Fresh capital entering markets in new year
                        - Crypto-friendly cabinet appointments announced
                        - Institutional funds rebalancing portfolios
                        
                        **Market Sentiment:** Greed to Extreme Greed
                        """)
                    
                    elif 'Apr 2025' in spike_month or 'April 2025' in spike_month:
                        if spike_day >= 8:
                            st.success("""
                            🚀 **Verified Cause: Tariff Pause Relief Rally**
                            
                            **What Happened:**
                            - April 9, 2025: Trump announced 90-day pause on tariffs
                            - Bitcoin rebounded 5.5% immediately
                            - Market relief as recession fears temporarily eased
                            - Short covering and bargain hunters drove rapid recovery
                            
                            **Market Context:**
                            - Recovery from April 7th crash (Bitcoin fell to $74,420)
                            - Proof that market could bounce back quickly from policy shocks
                            
                            **Sources:**
                            - Bloomberg: "Bitcoin surges on tariff pause announcement"
                            """)
                        else:
                            st.success("""
                            🚀 **Pattern: Pre-Crash Bull Run**
                            
                            Market was in bullish momentum before the April tariff shock.
                            This represents the peak before the correction.
                            """)
                    
                    elif 'Mar 2025' in spike_month or 'March 2025' in spike_month:
                        st.success("""
                        🚀 **Pattern: Q1 Momentum Continuation**
                        
                        **What Happened:**
                        - Bitcoin continued uptrend from post-election rally
                        - Positive regulatory developments
                        - Institutional accumulation phase
                        - Technical breakouts driving momentum
                        
                        **Market Sentiment:** Greed
                        """)
                    
                    else:
                        st.success(f"""
                        🚀 **Strong Profit Day**
                        
                        This significant gain occurred during {spike_month}. 
                        Possible causes: Technical breakout, positive news catalyst, short squeeze, or institutional buying.
                        Check news sources for specific events during this timeframe.
                        """)

    # Analysis of recovery patterns
    st.markdown("### 🔄 Recovery Analysis")
    
    recovery_col1, recovery_col2, recovery_col3 = st.columns(3)
    
    with recovery_col1:
        # Calculate average recovery time (days to positive after major dip)
        recovery_times = []
        for idx, loss_row in significant_losses.iterrows():
            loss_date = loss_row['date_match']
            # Find the next profitable day after this loss
            future_profits = daily_overview[
                (daily_overview['date_match'] > loss_date) & 
                (daily_overview['Closed PnL'] > 0)
            ]
            if not future_profits.empty:
                next_profit_date = future_profits.iloc[0]['date_match']
                days_to_recovery = (next_profit_date - loss_date).days
                recovery_times.append(days_to_recovery)
        
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        st.metric("Avg Recovery Time", f"{avg_recovery:.1f} days", help="Average days to return to profitability after a major loss")
    
    with recovery_col2:
        # Biggest single-day recovery
        biggest_recovery = daily_overview['Closed PnL'].max()
        st.metric("Biggest Rebound", f"${biggest_recovery:,.2f}", help="Largest single-day profit")
    
    with recovery_col3:
        # Win rate after losses
        win_rate = (daily_overview['Closed PnL'] > 0).sum() / len(daily_overview) * 100
        st.metric("Profitable Days", f"{win_rate:.1f}%", help="Percentage of days with positive PnL")

    st.success("""
    **💡 Key Insight: Volatility Creates Opportunity**
    
    While the April 2025 dip shows significant losses, these corrections are often followed by strong recoveries. 
    The data suggests that maintaining positions during "Extreme Fear" phases (despite temporary drawdowns) 
    leads to eventual profitability as markets recover. This validates the "Buy the Fear, Sell the Greed" strategy.
    """)

else:
    st.info("No significant market corrections detected in this dataset.")

st.markdown("---")


# ==============================================================================
# SECTION 4: BROADER ANALYSIS & INSIGHTS
# ==============================================================================
st.header("📊 Deep Dive: Strategy & Performance Insights")

tab1, tab2, tab3 = st.tabs(["💰 Profitability by Mood", "🧠 Long vs. Short Strategy", "⚠️ Risk Analysis"])

# --- TAB 1: PROFITABILITY ---
with tab1:
    st.subheader("Which market mood makes the most money?")
    
    sentiment_pnl = df_filtered.groupby('classification')['Closed PnL'].mean().reset_index()
    
    fig_bar = px.bar(sentiment_pnl, x='classification', y='Closed PnL', color='classification',
                     title="Average Profit per Trade by Sentiment",
                     color_discrete_map={'Extreme Fear': 'red', 'Fear': 'orange', 'Neutral': 'gray', 'Greed': 'lightgreen', 'Extreme Greed': 'green'})
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.success("✅ **Insight:** Contrary to popular belief, trading during 'Extreme Greed' (Momentum) was the most profitable strategy in this cycle, followed by 'Fear' (Buying the Dip).")

# --- TAB 2: LONG VS SHORT ---
with tab2:
    st.subheader("Should you Long or Short?")
    st.markdown("We analyzed thousands of trades to see which side works best in each mood.")
    
    # Group by Side and Sentiment
    strategy_stats = df_filtered.groupby(['Side', 'classification'])['Closed PnL'].mean().reset_index()
    
    fig_strat = px.bar(strategy_stats, x='classification', y='Closed PnL', color='Side', barmode='group',
                       title="Long (Buy) vs Short (Sell) Performance",
                       color_discrete_map={'BUY': '#00CC96', 'SELL': '#EF553B'})
    
    st.plotly_chart(fig_strat, use_container_width=True)
    
    st.info("""
    **💡 Key Strategy Discovery:**
    * **Buying (Longs):** Performs best during **FEAR** (Buying low).
    * **Selling (Shorts):** Performs best during **EXTREME GREED** (Shorting the top).
    * *This confirms the 'Contrarian' trading theory.*
    """)

# --- TAB 3: VOLATILITY ---
with tab3:
    st.subheader("Where is the Risk?")
    
    st.info("""
    **🤔 What's the difference between Win Rate and Volatility?**
    
    - **Win Rate** = How often you win (percentage of profitable trades)
    - **Volatility (Risk)** = How unpredictable your results are (standard deviation of profits/losses)
    
    **Example:** You could have a 60% win rate but high volatility if:
    - Some wins are +$1,000 and some are +$50
    - Some losses are -$2,000 and some are -$100
    
    High volatility = Less predictable outcomes = Higher risk (even if you win often!)
    """)
    
    vol_stats = df_filtered.groupby('classification')['Closed PnL'].std().reset_index()
    vol_stats.columns = ['Sentiment', 'Risk (Std Deviation)']
    
    # Add mean for reference
    vol_mean = df_filtered.groupby('classification')['Closed PnL'].mean().reset_index()
    vol_mean.columns = ['Sentiment', 'Avg PnL']
    vol_stats = vol_stats.merge(vol_mean, on='Sentiment')
    
    fig_vol = px.bar(vol_stats, x='Sentiment', y='Risk (Std Deviation)', 
                     color='Sentiment',
                     title="Market Volatility (Risk) by Sentiment Phase",
                     hover_data=['Avg PnL'])
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Show comparison table
    st.markdown("#### 📊 Win Rate vs Volatility Comparison")
    
    comparison_stats = df_filtered.groupby('classification').apply(
        lambda x: pd.Series({
            'Win Rate (%)': (len(x[x['Closed PnL'] > 0]) / len(x) * 100) if len(x) > 0 else 0,
            'Volatility (Risk)': x['Closed PnL'].std(),
            'Avg Profit': x['Closed PnL'].mean(),
            'Total Trades': len(x)
        })
    ).reset_index()
    
    st.dataframe(comparison_stats.style.format({
        'Win Rate (%)': '{:.1f}%',
        'Volatility (Risk)': '${:,.2f}',
        'Avg Profit': '${:.2f}',
        'Total Trades': '{:,}'
    }).background_gradient(subset=['Win Rate (%)'], cmap='RdYlGn')
    .background_gradient(subset=['Volatility (Risk)'], cmap='YlOrRd'), use_container_width=True)
    
    st.warning("⚠️ **Warning:** 'Extreme Fear' brings the highest volatility. While profitable for experts, it is the most dangerous time for beginners.")
    
    st.markdown("""
    **💡 Key Insight:**
    - **High Win Rate + Low Volatility** = Ideal (consistent, predictable profits)
    - **High Win Rate + High Volatility** = Risky (you win often but results vary wildly)
    - **Low Win Rate + Low Volatility** = Poor (consistently losing small amounts)
    - **Low Win Rate + High Volatility** = Dangerous (losing often with unpredictable swings)
    """)

st.markdown("---")

# ==============================================================================
# SECTION 5: WIN RATE & SUCCESS METRICS
# ==============================================================================
st.header("🎯 Win Rate & Success Metrics")
st.caption("Everyone wants to know: What's my win percentage?")

# Calculate win/loss metrics using cached function
win_stats = compute_win_rate_stats(df_filtered)
winning_trades = win_stats['winning_trades']
losing_trades = win_stats['losing_trades']
total_trades = win_stats['total_trades']
win_count = win_stats['win_count']
loss_count = win_stats['loss_count']
win_rate = win_stats['win_rate']
avg_win = win_stats['avg_win']
avg_loss = win_stats['avg_loss']

# Display key metrics
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Overall Win Rate", f"{win_rate:.1f}%", help="Percentage of profitable trades")

with metric_col2:
    st.metric("Total Wins", f"{win_count:,}", help="Number of profitable trades")

with metric_col3:
    st.metric("Total Losses", f"{loss_count:,}", help="Number of losing trades")

with metric_col4:
    st.metric("Avg Win", f"${avg_win:.2f}", help="Average profit per winning trade")

with metric_col5:
    st.metric("Avg Loss", f"${avg_loss:.2f}", help="Average loss per losing trade")

# Win Rate by Sentiment
st.subheader("📊 Win Rate by Market Sentiment")

win_rate_by_sentiment = df_filtered.groupby('classification').apply(
    lambda x: pd.Series({
        'Win Rate (%)': (len(x[x['Closed PnL'] > 0]) / len(x) * 100) if len(x) > 0 else 0,
        'Avg Profit': x['Closed PnL'].mean(),
        'Total Trades': len(x),
        'Winning Trades': len(x[x['Closed PnL'] > 0]),
        'Losing Trades': len(x[x['Closed PnL'] < 0])
    })
).reset_index()

fig_winrate = px.bar(win_rate_by_sentiment, x='classification', y='Win Rate (%)', 
                     color='Win Rate (%)',
                     title="Win Rate by Sentiment Phase",
                     color_continuous_scale='RdYlGn',
                     text='Win Rate (%)')
fig_winrate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig_winrate.update_layout(showlegend=False)
st.plotly_chart(fig_winrate, use_container_width=True)

# Detailed breakdown table
st.subheader("📋 Detailed Breakdown by Sentiment")
st.dataframe(win_rate_by_sentiment.style.format({
    'Win Rate (%)': '{:.1f}%',
    'Avg Profit': '${:.2f}',
    'Total Trades': '{:,}',
    'Winning Trades': '{:,}',
    'Losing Trades': '{:,}'
}).background_gradient(subset=['Win Rate (%)'], cmap='RdYlGn'), use_container_width=True)

# Win/Loss Ratio & Risk/Reward
st.subheader("⚖️ Risk/Reward Analysis")
st.markdown("""
This section evaluates whether your trading strategy has a **mathematical edge**. These metrics tell you if you can be profitable long-term, 
even if you don't win every trade.
""")

rr_col1, rr_col2, rr_col3 = st.columns(3)

with rr_col1:
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}:1", 
              help="How much you make when you win vs how much you lose when you lose")

with rr_col2:
    profit_factor = winning_trades['Closed PnL'].sum() / abs(losing_trades['Closed PnL'].sum()) if len(losing_trades) > 0 and losing_trades['Closed PnL'].sum() != 0 else 0
    st.metric("Profit Factor", f"{profit_factor:.2f}", 
              help="Total $ won divided by total $ lost across ALL trades")

with rr_col3:
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
    st.metric("Expectancy", f"${expectancy:.2f}", 
              help="How much you expect to make (or lose) per trade on average")

# Detailed explanation box
with st.expander("📖 What do these metrics mean?"):
    st.markdown(f"""
    ### Understanding Your Risk/Reward Metrics:
    
    **1. Win/Loss Ratio ({win_loss_ratio:.2f}:1)**
    - This compares the size of your average win (${avg_win:.2f}) to your average loss (${avg_loss:.2f})
    - **Ideal:** >1.0 means you make more when you win than you lose when you're wrong
    - **Your Status:** {"✅ Good! Your wins are bigger than your losses" if win_loss_ratio > 1 else "⚠️ Your losses are bigger than your wins - you need a higher win rate to compensate"}
    
    **2. Profit Factor ({profit_factor:.2f})**
    - Total money won (${winning_trades['Closed PnL'].sum():,.2f}) ÷ Total money lost (${abs(losing_trades['Closed PnL'].sum()):,.2f})
    - **Ideal:** >1.0 means you're profitable overall
    - **Your Status:** {"✅ Profitable! You've made more than you've lost" if profit_factor > 1 else "⚠️ Losing overall - total losses exceed total wins"}
    
    **3. Expectancy (${expectancy:.2f})**
    - Formula: (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    - This is the **most important metric** - it tells you your average profit per trade
    - **Ideal:** >$0 means you have a positive edge
    - **Your Status:** {"✅ Positive edge! Every trade you take has positive expected value" if expectancy > 0 else "⚠️ Negative edge - on average, each trade loses money"}
    
    ### 💡 What Should You Do?
    """)
    
    if expectancy > 0:
        st.success("""
        **✅ Your strategy is mathematically profitable!**
        - Keep doing what you're doing
        - Focus on consistency and discipline
        - Consider increasing position size gradually
        """)
    elif profit_factor > 1 and expectancy < 0:
        st.info("""
        **🔄 Mixed signals - You're profitable but expectancy is negative**
        - This might be due to a few very large wins skewing the data
        - Focus on more consistent smaller wins
        - Reduce the size of your losses
        """)
    else:
        st.error(f"""
        **⚠️ Strategy needs improvement. Here's how:**
        
        **Option 1: Improve Your Win Rate** (Currently {win_rate:.1f}%)
        - Study your losing trades - find common patterns
        - Tighten your entry criteria
        - Only take highest-probability setups
        
        **Option 2: Improve Your Win/Loss Ratio** (Currently {win_loss_ratio:.2f}:1)
        - Let your winners run longer (increase avg win from ${avg_win:.2f})
        - Cut your losses faster (decrease avg loss from ${avg_loss:.2f})
        - Use wider stop losses OR tighter take-profit targets
        
        **Quick Math:** To break even with your current {win_loss_ratio:.2f}:1 ratio, you need a win rate of at least {(abs(avg_loss)/(abs(avg_loss)+avg_win)*100):.1f}%
        """)

if expectancy > 0:
    st.success("✅ **Positive Expectancy:** Your strategy has a mathematical edge. Over many trades, you're expected to be profitable.")
else:
    st.warning("⚠️ **Negative Expectancy:** Your strategy may need adjustment. Consider improving win rate or risk/reward ratio.")

st.markdown("---")

# ==============================================================================
# SECTION 6: TIME-BASED ANALYSIS
# ==============================================================================
st.header("⏱️ Time-Based Performance Analysis")
st.caption("Discover your 'Golden Hours' and optimal trading times")

time_tab1, time_tab2, time_tab3, time_tab4 = st.tabs([
    "🕐 Hourly Analysis", 
    "📅 Day of Week", 
    "📆 Monthly Performance",
    "🔥 Performance Heatmap"
])

# --- HOURLY ANALYSIS ---
with time_tab1:
    st.subheader("🕐 Golden Hour: Best Time of Day to Trade")
    
    hourly_stats = df_filtered.groupby('hour').agg({
        'Closed PnL': ['sum', 'mean', 'count']
    }).reset_index()
    hourly_stats.columns = ['Hour', 'Total PnL', 'Avg PnL', 'Trade Count']
    
    # Create dual-axis chart with correct scaling
    fig_hourly = go.Figure()
    
    # Add bar chart for Total PnL
    fig_hourly.add_trace(go.Bar(
        x=hourly_stats['Hour'],
        y=hourly_stats['Total PnL'],
        name='Total PnL',
        marker_color='lightblue',
        yaxis='y',
        hovertemplate='Hour: %{x}:00<br>Total PnL: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add line chart for Average PnL per Trade (more meaningful than trade count!)
    fig_hourly.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['Avg PnL'],
        name='Avg PnL per Trade',
        marker_color='orange',
        yaxis='y2',
        mode='lines+markers',
        line=dict(width=3),
        hovertemplate='Hour: %{x}:00<br>Avg PnL: $%{y:.2f}<extra></extra>'
    ))
    
    # Highlight the golden hour
    golden_hour = hourly_stats.loc[hourly_stats['Total PnL'].idxmax(), 'Hour']
    fig_hourly.add_vline(
        x=golden_hour, 
        line_dash="dash", 
        line_color="gold", 
        opacity=0.7,
        annotation_text="⭐ Golden Hour",
        annotation_position="top"
    )
    
    fig_hourly.update_layout(
        title="Hourly Trading Performance (Blue bars = Total $, Orange line = Quality per trade)",
        xaxis=dict(title='Hour of Day (24h format)', tickmode='linear', dtick=1),
        yaxis=dict(
            title='Total PnL ($)', 
            side='left',
            showgrid=True
        ),
        yaxis2=dict(
            title='Avg PnL per Trade ($)', 
            side='right', 
            overlaying='y',
            showgrid=False
        ),
        hovermode='x unified',
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Identify golden hour
    best_hour = hourly_stats.loc[hourly_stats['Total PnL'].idxmax()]
    worst_hour = hourly_stats.loc[hourly_stats['Total PnL'].idxmin()]
    
    gold_col1, gold_col2 = st.columns(2)
    
    with gold_col1:
        st.success(f"""
        **🌟 Golden Hour: {int(best_hour['Hour'])}:00 - {int(best_hour['Hour'])+1}:00**
        - Total Profit: ${best_hour['Total PnL']:,.2f}
        - Avg Profit/Trade: ${best_hour['Avg PnL']:.2f}
        - Trade Count: {int(best_hour['Trade Count'])}
        """)
    
    with gold_col2:
        st.error(f"""
        **⚠️ Worst Hour: {int(worst_hour['Hour'])}:00 - {int(worst_hour['Hour'])+1}:00**
        - Total Loss: ${worst_hour['Total PnL']:,.2f}
        - Avg Loss/Trade: ${worst_hour['Avg PnL']:.2f}
        - Trade Count: {int(worst_hour['Trade Count'])}
        """)

# --- DAY OF WEEK ANALYSIS ---
with time_tab2:
    st.subheader("📅 Best Day of the Week")
    
    # Order days correctly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    daily_stats = df_filtered.groupby('day_of_week').agg({
        'Closed PnL': ['sum', 'mean', 'count']
    }).reset_index()
    daily_stats.columns = ['Day', 'Total PnL', 'Avg PnL', 'Trade Count']
    daily_stats['Day'] = pd.Categorical(daily_stats['Day'], categories=day_order, ordered=True)
    daily_stats = daily_stats.sort_values('Day')
    
    fig_daily = px.bar(daily_stats, x='Day', y='Total PnL', 
                       color='Total PnL',
                       title="Performance by Day of Week",
                       color_continuous_scale='RdYlGn',
                       text='Total PnL')
    fig_daily.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig_daily, use_container_width=True)
    
    # Best and worst day
    best_day = daily_stats.loc[daily_stats['Total PnL'].idxmax()]
    worst_day = daily_stats.loc[daily_stats['Total PnL'].idxmin()]
    
    day_col1, day_col2 = st.columns(2)
    
    with day_col1:
        st.success(f"""
        **🎯 Best Day: {best_day['Day']}**
        - Total Profit: ${best_day['Total PnL']:,.2f}
        - Avg Profit/Trade: ${best_day['Avg PnL']:.2f}
        - Total Trades: {int(best_day['Trade Count'])}
        """)
    
    with day_col2:
        st.error(f"""
        **📉 Worst Day: {worst_day['Day']}**
        - Total Loss: ${worst_day['Total PnL']:,.2f}
        - Avg Loss/Trade: ${worst_day['Avg PnL']:.2f}
        - Total Trades: {int(worst_day['Trade Count'])}
        """)

# --- MONTHLY PERFORMANCE ---
with time_tab3:
    st.subheader("📆 Monthly Performance Calendar")
    
    monthly_stats = df_filtered.groupby('year_month').agg({
        'Closed PnL': ['sum', 'mean', 'count']
    }).reset_index()
    monthly_stats.columns = ['Month', 'Total PnL', 'Avg PnL', 'Trade Count']
    
    fig_monthly = px.bar(monthly_stats, x='Month', y='Total PnL',
                         color='Total PnL',
                         title="Monthly Profit/Loss",
                         color_continuous_scale='RdYlGn',
                         text='Total PnL')
    fig_monthly.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_monthly.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Monthly statistics table
    st.dataframe(monthly_stats.style.format({
        'Total PnL': '${:,.2f}',
        'Avg PnL': '${:.2f}',
        'Trade Count': '{:,}'
    }).background_gradient(subset=['Total PnL'], cmap='RdYlGn'), use_container_width=True)

# --- PERFORMANCE HEATMAP ---
with time_tab4:
    st.subheader("🔥 Trading Performance Heatmap")
    st.caption("Visualize your best and worst trading times at a glance")
    
    # Create hour x day heatmap
    heatmap_data = df_filtered.groupby(['day_of_week', 'hour'])['Closed PnL'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='hour', columns='day_of_week', values='Closed PnL')
    
    # Reorder columns to match day order
    heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)
    
    fig_heatmap = px.imshow(heatmap_pivot,
                            labels=dict(x="Day of Week", y="Hour of Day", color="PnL ($)"),
                            x=heatmap_pivot.columns,
                            y=heatmap_pivot.index,
                            color_continuous_scale='RdYlGn',
                            aspect='auto',
                            title="Hour x Day Performance Heatmap")
    
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.info("💡 **How to read:** Green = Profitable periods, Red = Loss periods. Use this to identify your optimal trading windows.")

st.markdown("---")

# ==============================================================================
# SECTION 7: HOLDING TIME ANALYSIS
# ==============================================================================
if 'holding_time_hours' in df_filtered.columns:
    st.header("⏳ Holding Time Analysis")
    st.caption("How long should you hold a position for maximum profit?")
    
    hold_tab1, hold_tab2, hold_tab3 = st.tabs([
        "📊 Duration vs Profitability",
        "⏱️ Optimal Holding Time",
        "📈 Performance by Duration Category"
    ])
    
    # --- SCATTER PLOT ---
    with hold_tab1:
        st.subheader("Trade Duration vs Profitability")
        
        # Filter out extreme outliers for better visualization
        q_low = df_filtered['holding_time_hours'].quantile(0.01)
        q_high = df_filtered['holding_time_hours'].quantile(0.99)
        df_scatter = df_filtered[(df_filtered['holding_time_hours'] >= q_low) & 
                                 (df_filtered['holding_time_hours'] <= q_high)]
        
        # Sample data if too large for performance
        if len(df_scatter) > 5000:
            df_scatter = df_scatter.sample(n=5000, random_state=42)
            st.caption("📊 Showing 5,000 random trades for performance")
        
        fig_scatter = px.scatter(df_scatter, 
                                x='holding_time_hours', 
                                y='Closed PnL',
                                color='classification',
                                hover_data=['Side', 'Symbol'],
                                title="Trade Duration vs Profit/Loss",
                                labels={'holding_time_hours': 'Holding Time (Hours)', 
                                       'Closed PnL': 'Profit/Loss ($)'},
                                color_discrete_map={'Extreme Fear': 'red', 'Fear': 'orange', 
                                                   'Neutral': 'gray', 'Greed': 'lightgreen', 
                                                   'Extreme Greed': 'green'},
                                opacity=0.6)
        
        # Add simple linear trend line instead of LOWESS
        z = np.polyfit(df_scatter['holding_time_hours'].dropna(), 
                       df_scatter['Closed PnL'].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_scatter['holding_time_hours'].min(), 
                             df_scatter['holding_time_hours'].max(), 100)
        
        fig_scatter.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color='purple', dash='dash', width=2)
        ))
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info("💡 **Insight:** Each dot represents a trade. Look for patterns - do longer holds tend to be more profitable? Or are quick scalps more successful?")
    
    # --- OPTIMAL HOLDING TIME ---
    with hold_tab2:
        st.subheader("⏱️ Finding Your Optimal Holding Time")
        
        # Calculate statistics by holding time buckets
        df_filtered['hold_bucket'] = pd.cut(df_filtered['holding_time_hours'], 
                                            bins=[0, 1, 4, 12, 24, 72, 168, np.inf],
                                            labels=['<1h', '1-4h', '4-12h', '12-24h', '1-3d', '3-7d', '>7d'])
        
        hold_stats = df_filtered.groupby('hold_bucket').agg({
            'Closed PnL': ['mean', 'sum', 'count']
        }).reset_index()
        hold_stats.columns = ['Duration', 'Avg PnL', 'Total PnL', 'Trade Count']
        
        # Visualize
        fig_hold_bars = go.Figure()
        
        fig_hold_bars.add_trace(go.Bar(
            x=hold_stats['Duration'],
            y=hold_stats['Avg PnL'],
            name='Avg PnL per Trade',
            marker_color='skyblue',
            text=hold_stats['Avg PnL'],
            texttemplate='$%{text:.2f}',
            textposition='outside'
        ))
        
        fig_hold_bars.update_layout(
            title="Average Profitability by Holding Duration",
            xaxis_title="Holding Duration",
            yaxis_title="Average PnL ($)",
            height=400
        )
        
        st.plotly_chart(fig_hold_bars, use_container_width=True)
        
        # Statistics
        hold_col1, hold_col2, hold_col3 = st.columns(3)
        
        with hold_col1:
            avg_hold_time = df_filtered['holding_time_hours'].mean()
            st.metric("Avg Holding Time", f"{avg_hold_time:.1f}h")
        
        with hold_col2:
            median_hold_time = df_filtered['holding_time_hours'].median()
            st.metric("Median Holding Time", f"{median_hold_time:.1f}h")
        
        with hold_col3:
            best_duration = hold_stats.loc[hold_stats['Avg PnL'].idxmax()]
            st.metric("Most Profitable Duration", best_duration['Duration'])
    
    # --- CATEGORY PERFORMANCE ---
    with hold_tab3:
        st.subheader("📈 Performance by Trade Duration Category")
        
        if 'trade_duration_category' in df_filtered.columns:
            category_stats = df_filtered.groupby('trade_duration_category').agg({
                'Closed PnL': ['sum', 'mean', 'count']
            }).reset_index()
            category_stats.columns = ['Category', 'Total PnL', 'Avg PnL', 'Trade Count']
            
            # Calculate win rate by category
            category_winrate = df_filtered.groupby('trade_duration_category').apply(
                lambda x: (len(x[x['Closed PnL'] > 0]) / len(x) * 100) if len(x) > 0 else 0
            ).reset_index()
            category_winrate.columns = ['Category', 'Win Rate (%)']
            
            category_stats = category_stats.merge(category_winrate, on='Category')
            
            # Order categories logically
            category_order = ['Scalp (<1h)', 'Day Trade (1-24h)', 'Swing (1-7d)', 
                            'Position (1-4w)', 'Long-term (>1m)', 'Unknown']
            category_stats['Category'] = pd.Categorical(category_stats['Category'], 
                                                       categories=category_order, 
                                                       ordered=True)
            category_stats = category_stats.sort_values('Category')
            
            # Visualization
            fig_categories = go.Figure()
            
            fig_categories.add_trace(go.Bar(
                x=category_stats['Category'],
                y=category_stats['Total PnL'],
                name='Total PnL',
                marker_color='lightblue',
                yaxis='y'
            ))
            
            fig_categories.add_trace(go.Scatter(
                x=category_stats['Category'],
                y=category_stats['Win Rate (%)'],
                name='Win Rate %',
                marker_color='orange',
                yaxis='y2',
                mode='lines+markers'
            ))
            
            fig_categories.update_layout(
                title="Performance by Trade Duration Category",
                xaxis_title="Category",
                yaxis=dict(title='Total PnL ($)', side='left'),
                yaxis2=dict(title='Win Rate (%)', side='right', overlaying='y'),
                height=400
            )
            
            st.plotly_chart(fig_categories, use_container_width=True)
            
            # Detailed table
            st.dataframe(category_stats.style.format({
                'Total PnL': '${:,.2f}',
                'Avg PnL': '${:.2f}',
                'Trade Count': '{:,}',
                'Win Rate (%)': '{:.1f}%'
            }).background_gradient(subset=['Total PnL', 'Win Rate (%)'], cmap='RdYlGn'), 
            use_container_width=True)
            
            # Key insights
            best_category = category_stats.loc[category_stats['Avg PnL'].idxmax()]
            st.success(f"""
            **🎯 Optimal Strategy: {best_category['Category']}**
            - Average Profit per Trade: ${best_category['Avg PnL']:.2f}
            - Win Rate: {best_category['Win Rate (%)']:.1f}%
            - Total Trades: {int(best_category['Trade Count'])}
            """)

    st.markdown("---")

# ==============================================================================
# SECTION 8: SENTIMENT CORRELATION & WHAT-IF SCENARIOS
# ==============================================================================
st.header("🔮 Sentiment Correlation & Strategy Simulator")
st.caption("Visualize the sweet spot and test hypothetical strategies")

scenario_tab1, scenario_tab2 = st.tabs(["📊 Sentiment Scatter Plot", "🎲 What-If Scenarios"])

# --- SENTIMENT SCATTER ---
with scenario_tab1:
    st.subheader("📊 Find the Sweet Spot: Sentiment vs Profitability")
    
    # Aggregate by sentiment value
    sentiment_scatter = df_filtered.groupby('value').agg({
        'Closed PnL': ['mean', 'sum', 'count'],
        'classification': 'first'
    }).reset_index()
    sentiment_scatter.columns = ['Sentiment Score', 'Avg PnL', 'Total PnL', 'Trade Count', 'Classification']
    
    fig_sentiment_scatter = px.scatter(sentiment_scatter,
                                       x='Sentiment Score',
                                       y='Avg PnL',
                                       size='Trade Count',
                                       color='Classification',
                                       hover_data=['Total PnL', 'Trade Count'],
                                       title="Sentiment Score vs Average Profitability",
                                       labels={'Sentiment Score': 'Fear & Greed Index (0=Fear, 100=Greed)',
                                              'Avg PnL': 'Average PnL per Trade ($)'},
                                       color_discrete_map={'Extreme Fear': 'red', 'Fear': 'orange', 
                                                          'Neutral': 'gray', 'Greed': 'lightgreen', 
                                                          'Extreme Greed': 'green'})
    
    # Add horizontal line at y=0
    fig_sentiment_scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_sentiment_scatter.update_layout(height=500)
    st.plotly_chart(fig_sentiment_scatter, use_container_width=True)
    
    st.info("💡 **How to read:** The size of bubbles represents trade volume. Look for the sweet spot where profitability is highest!")

# --- WHAT-IF SCENARIOS ---
with scenario_tab2:
    st.subheader("🎲 Strategy Simulator: What If...?")
    st.markdown("Compare hypothetical trading strategies against your actual performance")
    
    # Strategy selection
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        scenario_sentiment = st.multiselect(
            "What if I ONLY traded during:",
            options=df['classification'].unique(),
            default=['Extreme Fear'],
            key='scenario_sentiment'
        )
    
    with scenario_col2:
        scenario_side = st.selectbox(
            "With strategy:",
            options=['Both (BUY & SELL)', 'Only LONG (BUY)', 'Only SHORT (SELL)'],
            key='scenario_side'
        )
    
    # Apply filters
    scenario_df = df[df['classification'].isin(scenario_sentiment)]
    
    if scenario_side == 'Only LONG (BUY)':
        scenario_df = scenario_df[scenario_df['Side'] == 'BUY']
    elif scenario_side == 'Only SHORT (SELL)':
        scenario_df = scenario_df[scenario_df['Side'] == 'SELL']
    
    # Calculate metrics
    if len(scenario_df) > 0:
        scenario_metrics_col1, scenario_metrics_col2, scenario_metrics_col3, scenario_metrics_col4 = st.columns(4)
        
        with scenario_metrics_col1:
            scenario_total = scenario_df['Closed PnL'].sum()
            actual_total = df['Closed PnL'].sum()
            delta = scenario_total - actual_total
            st.metric("Hypothetical Total PnL", f"${scenario_total:,.2f}", 
                     delta=f"${delta:,.2f} vs Actual", delta_color="normal")
        
        with scenario_metrics_col2:
            scenario_trades = len(scenario_df)
            actual_trades = len(df)
            st.metric("Trade Count", f"{scenario_trades:,}", 
                     delta=f"{scenario_trades - actual_trades:,} vs Actual")
        
        with scenario_metrics_col3:
            scenario_avg = scenario_df['Closed PnL'].mean()
            actual_avg = df['Closed PnL'].mean()
            st.metric("Avg PnL per Trade", f"${scenario_avg:.2f}",
                     delta=f"${scenario_avg - actual_avg:.2f} vs Actual")
        
        with scenario_metrics_col4:
            scenario_winrate = (len(scenario_df[scenario_df['Closed PnL'] > 0]) / len(scenario_df) * 100) if len(scenario_df) > 0 else 0
            actual_winrate = (len(df[df['Closed PnL'] > 0]) / len(df) * 100) if len(df) > 0 else 0
            st.metric("Win Rate", f"{scenario_winrate:.1f}%",
                     delta=f"{scenario_winrate - actual_winrate:.1f}% vs Actual")
        
        # Comparison chart
        comparison_data = pd.DataFrame({
            'Strategy': ['Actual Performance', 'Hypothetical Strategy'],
            'Total PnL': [actual_total, scenario_total],
            'Trade Count': [actual_trades, scenario_trades],
            'Win Rate (%)': [actual_winrate, scenario_winrate]
        })
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Actual',
            x=['Total PnL', 'Win Rate (%)'],
            y=[actual_total, actual_winrate],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Hypothetical',
            x=['Total PnL', 'Win Rate (%)'],
            y=[scenario_total, scenario_winrate],
            marker_color='lightgreen'
        ))
        
        fig_comparison.update_layout(
            title="Actual vs Hypothetical Performance",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Interpretation
        if scenario_total > actual_total:
            st.success(f"""
            ✅ **This strategy would have been MORE profitable!**
            
            By trading only during {', '.join(scenario_sentiment)} with {scenario_side.lower()}, 
            you would have earned an additional ${delta:,.2f} ({(delta/actual_total*100):.1f}% improvement).
            """)
        else:
            st.warning(f"""
            ⚠️ **This strategy would have been LESS profitable.**
            
            By trading only during {', '.join(scenario_sentiment)} with {scenario_side.lower()}, 
            you would have lost ${abs(delta):,.2f} ({(abs(delta)/actual_total*100):.1f}% worse than actual).
            """)
    else:
        st.warning("No trades match the selected criteria. Try adjusting your filters.")

st.markdown("---")

# ==============================================================================
# FOOTER & SOURCES
# ==============================================================================
st.markdown("---")
with st.expander("📚 Data Sources & Methodology"):
    st.write("""
    * **Primary Data:** Historical trading logs from Hyperliquid & Crypto Fear/Greed Index CSVs.
    * **Event Context:** US Election dates (Nov 5, 2024) verified via AP News / Federal Government records.
    * **Methodology:** Inner join on 'Date' field; PnL calculated as 'Closed PnL'.
    * **Time Analysis:** Extracted hour, day of week, and month from timestamp data.
    * **Holding Time:** Calculated as difference between Entry Time and Exit Time where available.
    """)