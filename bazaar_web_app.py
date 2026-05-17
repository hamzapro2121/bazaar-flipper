import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import requests
import time

# --- WEB PAGE CONFIGURATION ---
st.set_page_config(page_title="Hypixel Bazaar AI Sandbox", page_icon="📈", layout="wide")

st.title("📈 Hypixel SkyBlock Unrestricted Bazaar AI Trading Sandbox")
st.markdown("This web application uses an advanced **Genetic Optimization Algorithm** capable of dynamically evaluating any asset in the Hypixel economy.")

# --- STEP 1: LIVE HYPIXEL API DATA ENGINE (ALL ITEMS) ---
@st.cache_data(ttl=60)
def fetch_all_bazaar_products():
    """Fetches real-time price profiles for every item on the Hypixel Bazaar."""
    url = "https://hypixel.net"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data_json = response.json()
            if data_json.get("success"):
                return data_json["products"]
    except Exception:
        pass
    return None

# Pull entire catalog from live API
raw_products = fetch_all_bazaar_products()

if raw_products:
    ALL_AVAILABLE_ITEMS = sorted(list(raw_products.keys()))
else:
    # Safe structural fallbacks if API is offline during loading
    ALL_AVAILABLE_ITEMS = ['ENCHANTED_DIAMOND', 'PURPLE_CANDY', 'BOOSTER_COOKIE', 'STOCK_OF_STONKS', 'ENCHANTED_GOLD']
    raw_products = {}

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🛒 Asset Portfolio Selection")
# Multi-select dropdown packed with every item in the game
SELECTED_ITEMS = st.sidebar.multiselect(
    "Choose Items to Optimize", 
    options=ALL_AVAILABLE_ITEMS, 
    default=ALL_AVAILABLE_ITEMS[:3] # Starts with the first 3 items as default
)

st.sidebar.header("⚙️ Simulation Settings")
STARTING_BUDGET = st.sidebar.slider("Starting Capital (Coins)", 10000000, 500000000, 100000000, step=10000000)
TIMESTEPS = st.sidebar.slider("Trading Duration (Minutes / Timesteps)", 100, 1440, 400, step=50)
GENERATIONS = st.sidebar.slider("Genetic Generations", 5, 50, 15, step=5)
POPULATION_SIZE = st.sidebar.slider("Population Size (Diversity Pool)", 10, 50, 20, step=5)
INITIAL_MUTATION_RATE = st.sidebar.slider("Initial Mutation Probability Rate", 0.10, 0.80, 0.50, step=0.05)
RISK_PENALTY_WEIGHT = st.sidebar.slider("Drawdown Penalty Weight", 0.0, 5.0, 1.5, step=0.5)

# --- MAP GENETIC BOUNDS DYNAMICALLY ---
def generate_market_data(steps, selected_items):
    all_market_data = []
    time_axis = np.arange(steps)
    
    for item in selected_items:
        # Dynamic extraction of fallback points directly based on real-time value scales
        live_buy = 100.0
        live_sell = 103.0
        sell_moving_vol = 5000
        
        if item in raw_products:
            status = raw_products[item]["quick_status"]
            live_buy = status.get("buyPrice", 100.0)
            live_sell = status.get("sellPrice", 103.0)
            sell_moving_vol = max(10, status.get("sellMovingWeek", 500000) // 10080)
            
        # Avoid mathematical divide-by-zero crashes on dead items
        if live_buy <= 0: live_buy = 1.0
            
        # Simulate price pathways relative to actual market pricing
        random_noise = np.random.normal(0, live_buy * 0.012, steps)
        vol_noise = np.random.poisson(sell_moving_vol, steps)
        
        item_df = pd.DataFrame({
            'minute': time_axis,
            'item_id': item,
            'buy_price': np.maximum(0.1, live_buy + random_noise),
            'market_volume': np.maximum(1, vol_noise)
        })
        real_spread = live_sell / live_buy if live_buy > 0 else 1.03
        item_df['sell_price'] = item_df['buy_price'] * real_spread
        item_df['ma'] = item_df['buy_price'].rolling(window=10, min_periods=1).mean()
        
        all_market_data.append(item_df)
            
    return pd.concat(all_market_data).reset_index(drop=True)

TAX_RATE = 0.0125

# --- STEP 2: BULK-BUYING FITNESS SIMULATION ---
def run_trading_simulation(dna, selected_items, return_history=False):
    wallet = STARTING_BUDGET
    inventory = {item: 0 for item in selected_items}
    bought_at = {item: 0.0 for item in selected_items}
    time_held = {item: 0 for item in selected_items}
    item_profits = {item: 0.0 for item in selected_items}
    
    wallet_history = []
    minute_axis = []
    
    if not selected_items:
        return 0
        
    master_df = st.session_state['market_dataframe']
    
    for minute in range(TIMESTEPS):
        current_market = master_df[master_df['minute'] == minute]
        
        for _, row in current_market.iterrows():
            item = row['item_id']
            current_buy = row['buy_price']
            current_sell = row['sell_price']
            ma_price = row['ma']
            market_available_vol = row['market_volume']
            
            buy_dip_percent, sell_target_percent, max_hold_time = dna[item]
            
            if inventory[item] == 0:
                if current_buy < ma_price * (1 - buy_dip_percent) and wallet >= current_buy:
                    # Dynamically scales batch orders safely based on basic unit costs
                    max_allowed = 5000 if current_buy < 10000 else (100 if current_buy < 500000 else 5)
                    affordable = int(wallet // current_buy)
                    units_to_buy = min(max_allowed, affordable, int(market_available_vol))
                    
                    if units_to_buy > 0:
                        inventory[item] = units_to_buy
                        bought_at[item] = current_buy
                        cost = current_buy * units_to_buy
                        wallet -= cost
                        item_profits[item] -= cost
                        time_held[item] = 0
            
            elif inventory[item] > 0:
                time_held[item] += 1
                net_payout_per_unit = current_sell * (1 - TAX_RATE)
                
                if net_payout_per_unit > bought_at[item] * (1 + sell_target_percent) or time_held[item] >= max_hold_time:
                    units_to_sell = min(inventory[item], int(market_available_vol))
                    if units_to_sell > 0:
                        payout = net_payout_per_unit * units_to_sell
                        wallet += payout
                        item_profits[item] += payout
                        inventory[item] -= units_to_sell
                        if inventory[item] == 0:
                            bought_at[item] = 0.0
                    
        wallet_history.append(wallet)
        minute_axis.append(minute)
                    
    for item in selected_items:
        if inventory[item] > 0:
            final_rows = master_df[(master_df['minute'] == TIMESTEPS - 1) & (master_df['item_id'] == item)]
            if not final_rows.empty:
                emergency_payout = final_rows.iloc[0]['buy_price'] * (1 - TAX_RATE) * inventory[item]
                wallet += emergency_payout
                item_profits[item] += emergency_payout
                
    net_profit = wallet - STARTING_BUDGET
    
    wallet_series = pd.Series(wallet_history)
    pct_changes = wallet_series.pct_change().dropna()
    downside_returns = pct_changes[pct_changes < 0]
    downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0.0
    risk_penalty = downside_deviation * RISK_PENALTY_WEIGHT * STARTING_BUDGET
    adjusted_fitness = net_profit - risk_penalty

    if return_history:
        return adjusted_fitness, net_profit, item_profits, minute_axis, wallet_history
    return adjusted_fitness

# --- TRIGGER THE ALGORITHM RUN ---
if not SELECTED_ITEMS:
    st.info("👈 Please select at least one item from the sidebar portfolio tool to start.")
else:
    if st.button("🚀 Run Unrestricted Genetic Engine"):
        st.session_state['market_dataframe'] = generate_market_data(TIMESTEPS, SELECTED_ITEMS)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        population = []
        for _ in range(POPULATION_SIZE):
            bot_dna = {}
            for item in SELECTED_ITEMS:
                bot_dna[item] = [
                    np.random.uniform(0.00, 0.02),
                    np.random.uniform(0.005, 0.05),
                    np.random.randint(10, 100)
                ]
            population.append(bot_dna)

        for gen in range(GENERATIONS):
            current_mutation_rate = INITIAL_MUTATION_RATE * (1.0 - (gen / GENERATIONS))
            
            fitness_scores = [run_trading_simulation(bot_dna, SELECTED_ITEMS) for bot_dna in population]
            fitness_array = np.array(fitness_scores)
            sorted_indices = np.argsort(fitness_array)[::-1]
            
            population = [population[i] for i in sorted_indices]
            best_fitness = fitness_array[sorted_indices][0]
            best_dna = population[0]
            
            pct = int(((gen + 1) / GENERATIONS) * 100)
            progress_bar.progress(pct)
            status_text.text(f"Generation {gen+1}/{GENERATIONS} | Elite Score: {best_fitness:,.0f} | Mut Rate: {current_mutation_rate:.2f}")
            
            cutoff = max(2, POPULATION_SIZE // 2)
            survivors = population[:cutoff]
            
            new_population = [best_dna]  
            while len(new_population) < POPULATION_SIZE:
                p1, p2 = random.sample(survivors, 2)
                child_dna = {}
                for item in SELECTED_ITEMS:
                    child_dna[item] = []
                    for param_idx in range(3):
                        gene = p1[item][param_idx] if random.random() > 0.5 else p2[item][param_idx]
                        
                        if random.random() < current_mutation_rate:
                            if param_idx == 0: gene = max(0.0, gene + np.random.normal(0, 0.004))
                            elif param_idx == 1: gene = max(0.002, gene + np.random.normal(0, 0.008))
                            elif param_idx == 2: gene = int(max(5, gene + np.random.randint(-20, 21)))
                        child_dna[item].append(gene)
                new_population.append(child_dna)
            population = new_population

        adj_fit, net_gain, item_profits, time_history, wealth_history = run_trading_simulation(best_dna, SELECTED_ITEMS, return_history=True)

        # --- UI RESULTS DISPLAY ---
        st.success(f"🏁 Evolution Complete! Net Raw Yield: **{net_gain:+,.2f} Coins** (Risk-Adjusted Score: {adj_fit:,.2f})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧬 Top Strategic Configurations")
            # Displays optimized strategy settings for selected portfolio entities
            display_limit = min(5, len(SELECTED_ITEMS))
            st.write(f"Showing up to {display_limit} items:")
            for item in SELECTED_ITEMS[:display_limit]:
                st.write(f"**{item}:**")
                st.text(f"  ↳ Buy Dip target: {best_dna[item][0]*100:.3f}% below Moving Avg")
                st.text(f"  ↳ Sell Profit Target: {best_dna[item][1]*100:.2f}% markup threshold")
                st.text(f"  ↳ Max Hold Lifecycle: {best_dna[item][2]} ticks")

        with col2:
            st.subheader("📦 Asset Yield Allocations")
            for item, prof in list(item_profits.items())[:10]: # Caps UI output list to avoid layout clutter
                st.metric(label=f"{item} Yield", value=f"{prof:,.2f} Coins")

        st.subheader("📊 Portfolio Liquidity Tracking Window")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_history, y=wealth_history, mode='lines', name='Total Liquidity Pool', line=dict(color='#00ffcc', width=3)))
        fig.update_layout(template="plotly_dark", xaxis_title="Simulation Timeline (Ticks)", yaxis_title="Liquid Capital Pool Value")
        st.plotly_chart(fig, use_container_width=True)
