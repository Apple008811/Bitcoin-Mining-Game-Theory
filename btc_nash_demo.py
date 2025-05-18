import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
import math
import datetime

# 添加成本假设
ASSUMPTIONS = {
    'rnd_efficiency': 0.05,
    'spillover_rate': 0.2,
    'rnd_cost_factor': 0.1,
    'electricity_cost': 0.035,      # 美元/kWh，行业平均
    'maintenance_cost': 0.03,       # 设备成本的年维护费用
    'cooling_cost': 0.02,           # 设备成本的年冷却费用
    'labor_cost': 0.02,             # 设备成本的年人工费用
    'hardware_lifetime': 3,         # 设备使用寿命（年）
    'efficiency_improvement': 0.1,
    'depreciation_rate': 0.33,      # 年折旧率
    'hardware_cost_per_th': 20,     # 每TH/s的设备成本（美元）
}

class MiningCostCalculator:
    def __init__(self, assumptions):
        self.assumptions = assumptions
        
    def calculate_hardware_cost(self, hashrate_ths):
        """计算硬件成本（每区块）"""
        hardware_cost = hashrate_ths * self.assumptions['hardware_cost_per_th']
        blocks_per_year = 52560  # 比特币每年区块数
        hardware_cost_per_block = hardware_cost / (blocks_per_year * self.assumptions['hardware_lifetime'])
        return hardware_cost_per_block
        
    def calculate_electricity_cost(self, hashrate_ths, power_per_th_watt=30):
        """计算电费（每区块）"""
        total_power_watt = hashrate_ths * power_per_th_watt
        block_time_sec = 600  # 比特币平均出块时间
        kwh_per_block = (total_power_watt * block_time_sec) / (3600 * 1000)
        electricity_cost = kwh_per_block * self.assumptions['electricity_cost']
        return electricity_cost
        
    def calculate_operational_cost(self, hashrate_ths):
        """计算运营成本（每区块）"""
        hardware_cost = hashrate_ths * self.assumptions['hardware_cost_per_th']
        blocks_per_year = 52560
        maintenance_cost = (hardware_cost * self.assumptions['maintenance_cost']) / blocks_per_year
        cooling_cost = (hardware_cost * self.assumptions['cooling_cost']) / blocks_per_year
        labor_cost = (hardware_cost * self.assumptions['labor_cost']) / blocks_per_year
        return maintenance_cost + cooling_cost + labor_cost
        
    def calculate_total_cost_per_block(self, hashrate_ths, power_per_th_watt=30):
        """计算总成本（每区块）"""
        hardware_cost = self.calculate_hardware_cost(hashrate_ths)
        electricity_cost = self.calculate_electricity_cost(hashrate_ths, power_per_th_watt)
        operational_cost = self.calculate_operational_cost(hashrate_ths)
        return hardware_cost + electricity_cost + operational_cost

def fetch_btc_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    resp = requests.get(url)
    data = resp.json()
    return data['bitcoin']['usd']

def fetch_block_height():
    url = "https://blockchain.info/q/getblockcount"
    resp = requests.get(url)
    return int(resp.text)

def get_block_reward(block_height):
    # 比特币每210,000区块减半
    halvings = block_height // 210000
    initial_reward = 50
    reward = initial_reward / (2 ** halvings)
    return reward

st.title("Bitcoin Mining Pools: Hashrate, Nash Equilibrium, and Profit Analysis")

@st.cache_data
def fetch_btc_hashrate():
    url = "https://mempool.space/api/v1/mining/pools/2h"
    resp = requests.get(url)
    data = resp.json()
    if isinstance(data, dict) and 'pools' in data:
        data = data['pools']
    pools = []
    block_counts = []
    for pool in data:
        pools.append(pool.get('name', str(pool)))
        block_counts.append(pool.get('blockCount', 0))
    df = pd.DataFrame({'Pool': pools, 'Blocks': block_counts})
    df['Total Blocks'] = df['Blocks'].sum()
    df['Hashrate (%)'] = df['Blocks'] / df['Total Blocks'] * 100
    return df

# Initialize the mining cost calculator with default assumptions
cost_calculator = MiningCostCalculator(ASSUMPTIONS)

df = fetch_btc_hashrate()
st.dataframe(df)

block_reward = st.number_input("Block Reward (BTC)", value=3.125, step=0.001)
avg_fee_per_block = st.number_input("Average Fee per Block (BTC)", value=0.2, step=0.01)

# theoretical model
df['Total Hashrate'] = df['Hashrate (%)'].sum()
df['Prob'] = df['Hashrate (%)'] / df['Total Hashrate']
df['Expected Reward'] = df['Prob'] * (block_reward + avg_fee_per_block)

# Use MiningCostCalculator to compute actual cost for each pool
# Assume Hashrate (%) is proportional to hash_rate input for the calculator
# (If needed, you can scale or normalize as appropriate)
def fetch_network_hashrate_ths():
    # Use blockchain.com API, returns hashrate in TH/s
    url = "https://api.blockchain.info/charts/hash-rate?timespan=7days&format=json"
    resp = requests.get(url)
    data = resp.json()
    try:
        values = data.get('values', [])
        if not values:
            st.error("No hashrate data returned from API.")
            return 0
        # 取最后一个非空数据点
        for v in reversed(values):
            if v.get('y', 0) > 0:
                hashrate_ths = v['y']  # 直接就是TH/s
                return hashrate_ths
        st.error("No valid hashrate value found in API data.")
        return 0
    except Exception as e:
        st.error(f"Failed to fetch network hashrate: {e}")
        return 0

network_hashrate_ths = fetch_network_hashrate_ths()
st.write(f"Current Network Hashrate: {network_hashrate_ths:,.0f} TH/s")

# For each pool, calculate its absolute hashrate in TH/s
# and use it for cost calculation

df['Pool Hashrate (TH/s)'] = df['Hashrate (%)'] / 100 * network_hashrate_ths
df['Cost'] = df['Pool Hashrate (TH/s)'].apply(lambda x: cost_calculator.calculate_total_cost_per_block(x))
df['Net Profit'] = df['Expected Reward'] - df['Cost']

st.dataframe(df[['Pool', 'Hashrate (%)', 'Pool Hashrate (TH/s)', 'Prob', 'Expected Reward', 'Cost', 'Net Profit']])

# 可视化
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(df['Pool'], df['Hashrate (%)'], color='skyblue', label='Hashrate %')
ax1.set_ylabel('Hashrate (%)', color='blue')
ax1.set_ylim(0, max(df['Hashrate (%)']) + 10)
ax1.set_title('Bitcoin Mining Pools: Hashrate & Nash Equilibrium Expected Reward')

ax2 = ax1.twinx()
ax2.plot(df['Pool'], df['Expected Reward'], color='orange', marker='o', label='Expected Reward (BTC)')
ax2.plot(df['Pool'], df['Net Profit'], color='green', marker='x', label='Net Profit (BTC)')
ax2.set_ylabel('BTC', color='orange')
ax2.set_ylim(min(df['Net Profit'].min(), 0), max(df['Expected Reward'].max(), 0.5))

plt.xticks(rotation=30)
ax2.legend(loc='upper right')
st.pyplot(fig)

# HHI index
hhi = sum((h/100)**2 for h in df['Hashrate (%)'])
st.write(f"Herfindahl-Hirschman Index (HHI): {hhi:.4f} (lower is more decentralized)")

st.title("Bitcoin Mining Profitability: 2025 vs 2028-2032 Forecast")

# 当前参数
st.header("Current (2025) Parameters")
btc_price = fetch_btc_price()
block_height = fetch_block_height()
block_reward = get_block_reward(block_height)
mining_cost_per_btc = 40000  # 美元/枚，建议定期手动更新

st.write(f"Current BTC Price: ${btc_price:,}")
st.write(f"Current Block Reward: {block_reward} BTC (Block Height: {block_height})")
st.write(f"Estimated Mining Cost per BTC: ${mining_cost_per_btc:,}")

# 计算每区块成本
block_cost = mining_cost_per_btc * block_reward
block_revenue = btc_price * block_reward
profit_per_block = block_revenue - block_cost
st.write(f"Block Cost: ${block_cost:,.0f}")
st.write(f"Block Revenue: ${block_revenue:,.0f}")
st.write(f"Block Profit: ${profit_per_block:,.0f}")

# 预测参数
st.header("Forecast (2028-2032)")
btc_price_2028 = st.number_input("BTC Price (2028-2032, forecast)", value=80000)
block_reward_2028 = 1.5625
cost_per_btc_2028 = st.number_input("Mining Cost per BTC (2028-2032, forecast)", value=45000)
block_cost_2028 = cost_per_btc_2028 * block_reward_2028
block_revenue_2028 = btc_price_2028 * block_reward_2028
profit_2028 = block_revenue_2028 - block_cost_2028

st.write(f"Block Reward: {block_reward_2028} BTC")
st.write(f"Block Cost: ${block_cost_2028:,.0f}")
st.write(f"Block Revenue: ${block_revenue_2028:,.0f}")
st.write(f"Block Profit: ${profit_2028:,.0f}")

# 可视化对比
labels = ['2025', '2028-2032']
profits = [profit_per_block, profit_2028]
costs = [block_cost, block_cost_2028]
revenues = [block_revenue, block_revenue_2028]

fig, ax = plt.subplots()
ax.bar(labels, revenues, label='Block Revenue', color='orange')
ax.bar(labels, costs, label='Block Cost', color='blue', alpha=0.5)
ax.plot(labels, profits, label='Block Profit', color='green', marker='o')
ax.set_ylabel('USD')
ax.set_title('Block Revenue, Cost, and Profit Comparison')
ax.legend()
st.pyplot(fig)

# 假设一年有52560个区块
num_blocks = st.number_input("Number of blocks (e.g. per year)", value=52560)
df['Blocks Mined'] = df['Prob'] * num_blocks
df['Total Profit'] = df['Blocks Mined'] * (block_reward - df['Cost'].iloc[0])
st.dataframe(df[['Pool', 'Blocks Mined', 'Total Profit']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

# 流程节点文本
steps = [
    "Theoretical Model\n(Nash Equilibrium, Mining Contest)",
    "Input Real-World Data\n(Hashrate, Cost, Price, Reward)",
    "Plug Data into Model",
    "Empirical Calculation\n(Expected Reward, Net Profit)",
    "Visualization & Sensitivity Analysis",
    "Practical Insights\n(Miners, Investors, Policymakers)"
]

# 节点坐标
y = list(reversed(range(len(steps))))
x = [0]*len(steps)

# 绘制节点
for i, (xi, yi, text) in enumerate(zip(x, y, steps)):
    box = FancyBboxPatch((xi-0.5, yi-0.25), 1, 0.5,
                         boxstyle="round,pad=0.1", fc="#e3f2fd", ec="#1565c0", lw=2)
    ax.add_patch(box)
    ax.text(xi, yi, text, ha='center', va='center', fontsize=12, color='#1565c0')

# 绘制箭头
for i in range(len(steps)-1):
    ax.annotate('', xy=(x[i+1], y[i+1]+0.25), xytext=(x[i], y[i]-0.25),
                arrowprops=dict(arrowstyle="->", lw=2, color='#1565c0'))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, len(steps))
plt.title("Bridging Theory and Real-World Data in Bitcoin Mining Analysis", fontsize=14, color='#1565c0')
plt.tight_layout()
plt.show()

# 假设df['hashrate']为历史算力数据
data = df['Hashrate (%)'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 构造滑动窗口
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

look_back = 30
X, y = create_dataset(data_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# # 构建LSTM模型
# model = Sequential([
#     LSTM(50, input_shape=(look_back, 1)),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# # 预测未来
# last_seq = data_scaled[-look_back:]
# future = []
# current_seq = last_seq
# for _ in range(30):  # 预测未来30天
#     pred = model.predict(current_seq.reshape(1, look_back, 1))
#     future.append(pred[0, 0])
#     current_seq = np.append(current_seq[1:], pred, axis=0)
# future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

# # 可视化
# plt.plot(df['Hashrate (%)'], label='Historical')
# plt.plot(range(len(df), len(df)+30), future, label='LSTM Forecast')
# plt.legend()
# plt.show()

# 获取当前区块高度
block_height = fetch_block_height()
block_reward = get_block_reward(block_height)
st.write(f"Current Block Reward: {block_reward} BTC (Block Height: {block_height})")

btc_price = fetch_btc_price()
st.write(f"Current BTC Price: ${btc_price:,}")

years = [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
# 用实际数据或简单生成假数据，确保长度一致
revenues = [100000, 120000, 110000, 130000, 125000, 140000, 135000, 138000, 137000, 136000, 134000]
costs =    [ 80000,  90000,  95000, 100000, 105000, 110000, 115000, 117000, 118000, 119000, 120000]
profits =  [r-c for r, c in zip(revenues, costs)]

plt.figure(figsize=(10,6))
plt.plot(years, revenues, label='Block Revenue', color='orange', marker='o')
plt.plot(years, costs, label='Block Cost', color='blue', marker='o')
plt.plot(years, profits, label='Block Profit', color='green', marker='o')
plt.fill_between(years, costs, revenues, color='orange', alpha=0.2, label='Gross Margin')
plt.title('Block Revenue, Cost, and Profit Trend (2018-2032)')
plt.xlabel('Year')
plt.ylabel('USD')
plt.legend()
plt.grid(True)
st.pyplot(plt)

def analyze_hash_rate_centralization(hash_rates):
    """
    Calculate the Herfindahl index for hash rate distribution.
    The Herfindahl index is a measure of concentration, calculated as the sum of the squares of the market shares.
    """
    total_hash_rate = sum(hash_rates)
    if total_hash_rate == 0:
        return 0
    market_shares = [h / total_hash_rate for h in hash_rates]
    herfindahl_index = sum(share ** 2 for share in market_shares)
    return herfindahl_index

# Calculate and print the Herfindahl index for hash rate centralization
herfindahl = analyze_hash_rate_centralization(df['Hashrate (%)'].values)
print(f"Herfindahl Index for Hash Rate Centralization: {herfindahl:.4f}")

if __name__ == "__main__":
    # ... existing code ...
    
    # Calculate and print the Herfindahl index for hash rate centralization
    herfindahl = analyze_hash_rate_centralization(df['Hashrate (%)'].values)
    print(f"Herfindahl Index for Hash Rate Centralization: {herfindahl:.4f}")

    # ... existing code ... 

st.header("Single Miner Profitability: CoinWarz Parameter Alignment")

# CoinWarz参数
single_miner_hashrate = 390  # TH/s
power_watts = 7215
electricity_cost_per_kwh = 0.05
btc_price = 104606.73  # 或用API获取的btc_price
block_reward = 3.125  # 或用API获取的block_reward
blocks_per_day = 144
network_hashrate_ths = network_hashrate_ths  # 用API获取的

btc_per_day = (single_miner_hashrate / network_hashrate_ths) * block_reward * blocks_per_day
kwh_per_day = power_watts * 24 / 1000
electricity_cost_per_day = kwh_per_day * electricity_cost_per_kwh
revenue_per_day = btc_per_day * btc_price
profit_per_day = revenue_per_day - electricity_cost_per_day

df['Single Miner BTC/day'] = btc_per_day
df['Single Miner Revenue/day'] = revenue_per_day
df['Single Miner Electricity Cost/day'] = electricity_cost_per_day
df['Single Miner Profit/day'] = profit_per_day

st.dataframe(df[['Pool', 'Single Miner BTC/day', 'Single Miner Revenue/day', 'Single Miner Electricity Cost/day', 'Single Miner Profit/day']])

# --- The Longest Chain安全性与攻击成本分析 ---
st.header("The Longest Chain安全性与攻击成本分析")

# 用户输入参数
default_network_hashrate_ths = network_hashrate_ths if 'network_hashrate_ths' in locals() else 864_509_979
default_power_per_th_watt = 30
default_electricity_cost_per_kwh = 0.05

aq = st.slider("攻击者算力占比（%）", 1, 99, 40, step=1) / 100
z = st.slider("受害者等待确认区块数", 1, 20, 6)
attack_hours = st.number_input("攻击持续时间（小时）", value=1, key="attack_hours")
network_hashrate_ths = st.number_input("全网算力 (TH/s)", value=int(default_network_hashrate_ths), key="network_hashrate_ths_attack")
power_per_th_watt = st.number_input("单位算力功耗 (W/TH)", value=default_power_per_th_watt, key="power_per_th_watt_attack")
electricity_cost_per_kwh = st.number_input("电价 ($/kWh)", value=default_electricity_cost_per_kwh, key="electricity_cost_per_kwh_attack")

# 51%攻击成功概率
def double_spend_success_prob(q, z):
    p = 1 - q
    if q >= p:
        return 1.0
    lambda_ = z * (q / p)
    prob = 0.0
    for k in range(z+1):
        poisson = math.exp(-lambda_) * lambda_**k / math.factorial(k)
        prob += poisson * (1 - (q/p)**(z-k))
    return prob

prob = double_spend_success_prob(aq, z)
st.write(f"攻击者算力 {aq*100:.1f}%，等待 {z} 个确认，攻击成功概率：{prob:.4%}")

# 攻击成本
def attack_cost(q, network_hashrate_ths, power_per_th_watt, electricity_cost_per_kwh, attack_hours):
    attacker_hashrate = q * network_hashrate_ths  # TH/s
    total_power_watt = attacker_hashrate * power_per_th_watt
    total_power_kwh = total_power_watt * attack_hours / 1000
    cost = total_power_kwh * electricity_cost_per_kwh
    return cost

cost = attack_cost(aq, network_hashrate_ths, power_per_th_watt, electricity_cost_per_kwh, attack_hours)
st.write(f"{aq*100:.1f}%攻击者发动{attack_hours}小时攻击的电费成本：${cost:,.2f}")

# 单台矿机每日利润是否为正
df_profit = df['Single Miner Profit/day'].iloc[0] if 'Single Miner Profit/day' in df.columns else None
if df_profit is not None:
    if df_profit > 0:
        st.success(f"当前单台矿机每日利润为正：${df_profit:.2f}")
    else:
        st.error(f"当前单台矿机每日利润为负：${df_profit:.2f}")

st.header("动态仿真：积分/期望视角下的矿工利润")

# --- 动态仿真参数 ---
current_year = datetime.datetime.now().year
T_days = st.number_input("仿真总天数", value=3650, key="T_days_sim")
dt = 1  # 步长1天
steps = int(T_days / dt)

# 算力变化模型
st.subheader("算力变化模型")
hashrate_start = st.number_input("初始算力 (TH/s)", value=390, key="hashrate_start_sim")
hashrate_growth = st.number_input("每日算力增长 (TH/s)", value=0, key="hashrate_growth_sim")
hashrate_noise = st.number_input("每日算力波动幅度 (TH/s)", value=0, key="hashrate_noise_sim")

# 币价变化模型
st.subheader("币价变化模型")
btc_price_start = st.number_input("初始BTC价格 ($)", value=104606.73, key="btc_price_start_sim")
btc_price_growth = st.number_input("每日BTC价格增长率 (如0.001=0.1%)", value=0.0, key="btc_price_growth_sim")
btc_price_noise = st.number_input("每日BTC价格波动幅度 ($)", value=0.0, key="btc_price_noise_sim")

# 成本变化模型
st.subheader("成本变化模型")
power_per_th_watt_sim = st.number_input("单位算力功耗 (W/TH)", value=30, key="power_per_th_watt_sim")
electricity_cost_per_kwh_start = st.number_input("初始电价 ($/kWh)", value=0.05, key="electricity_cost_per_kwh_sim")
electricity_cost_growth = st.number_input("每日电价增长率 (如0.001=0.1%)", value=0.0, key="electricity_cost_growth_sim")
electricity_cost_noise = st.number_input("每日电价波动幅度 ($)", value=0.0, key="electricity_cost_noise_sim")
other_cost_per_day = st.number_input("其他每日运营成本 ($)", value=0, key="other_cost_per_day_sim")

# --- 区块奖励与手续费动态 ---
st.subheader("区块奖励与手续费动态")
block_reward_start = st.number_input("初始区块奖励 (BTC)", value=3.125, key="block_reward_start_sim")
halving_interval = st.number_input("减半周期（天）", value=1460, key="halving_interval_sim")  # 约4年=1460天
block_height_start = st.number_input("初始区块高度", value=840000, key="block_height_start_sim")
blocks_per_day_sim = 144

# 手续费动态
avg_fee_per_block = st.number_input("初始每区块手续费 (BTC)", value=0.2, key="avg_fee_per_block_sim")
fee_growth = st.number_input("每日手续费增长率 (如0.001=0.1%)", value=0.0, key="fee_growth_sim")
fee_noise = st.number_input("每日手续费波动幅度 (BTC)", value=0.0, key="fee_noise_sim")

# --- 仿真主循环 ---
hashrate = []
costs = []
rewards = []
profits = []
btc_prices = []
electricity_costs = []
block_rewards = []
avg_fees = []

cur_hashrate = hashrate_start
cur_btc_price = btc_price_start
cur_electricity_cost = electricity_cost_per_kwh_start
cur_block_reward = block_reward_start
cur_fee = avg_fee_per_block
cur_block_height = block_height_start

for t in range(steps):
    # 算力随时间变化
    cur_hashrate += hashrate_growth + np.random.normal(0, hashrate_noise)
    cur_hashrate = max(cur_hashrate, 0)
    hashrate.append(cur_hashrate)
    
    # 币价随时间变化
    cur_btc_price *= (1 + btc_price_growth)
    cur_btc_price += np.random.normal(0, btc_price_noise)
    cur_btc_price = max(cur_btc_price, 0)
    btc_prices.append(cur_btc_price)
    
    # 电价随时间变化
    cur_electricity_cost *= (1 + electricity_cost_growth)
    cur_electricity_cost += np.random.normal(0, electricity_cost_noise)
    cur_electricity_cost = max(cur_electricity_cost, 0)
    electricity_costs.append(cur_electricity_cost)
    
    # 区块奖励减半逻辑
    if t > 0 and (t % halving_interval == 0):
        cur_block_reward /= 2
    block_rewards.append(cur_block_reward)
    
    # 手续费动态
    cur_fee *= (1 + fee_growth)
    cur_fee += np.random.normal(0, fee_noise)
    cur_fee = max(cur_fee, 0)
    avg_fees.append(cur_fee)
    
    # 成本
    kwh_per_day = cur_hashrate * power_per_th_watt_sim * 24 / 1000
    electricity_cost = kwh_per_day * cur_electricity_cost
    total_cost = electricity_cost + other_cost_per_day
    costs.append(total_cost)
    
    # 预期收益（含手续费）
    btc_per_day = (cur_hashrate / network_hashrate_ths) * blocks_per_day_sim
    reward_btc = btc_per_day * cur_block_reward
    fee_btc = btc_per_day * cur_fee
    revenue = (reward_btc + fee_btc) * cur_btc_price
    rewards.append(revenue)
    
    # 利润
    profits.append(revenue - total_cost)
    
    # 区块高度推进
    cur_block_height += blocks_per_day_sim

# 数值积分
total_profit = np.sum(profits)
total_cost = np.sum(costs)
total_reward = np.sum(rewards)

st.write(f"仿真总收益: ${total_reward:,.2f}")
st.write(f"仿真总成本: ${total_cost:,.2f}")
st.write(f"仿真总利润: ${total_profit:,.2f}")

# 可视化
import matplotlib.pyplot as plt
df_sim = pd.DataFrame({
    "Day": np.arange(steps),
    "Hashrate (TH/s)": hashrate,
    "BTC Price ($)": btc_prices,
    "Electricity Cost ($/kWh)": electricity_costs,
    "Block Reward (BTC)": block_rewards,
    "Avg Fee (BTC)": avg_fees,
    "Cost ($)": costs,
    "Reward ($)": rewards,
    "Profit ($)": profits
})
st.line_chart(df_sim.set_index("Day")["Profit ($)"])
st.line_chart(df_sim.set_index("Day")["Reward ($)"])
st.line_chart(df_sim.set_index("Day")[["BTC Price ($)", "Electricity Cost ($/kWh)", "Hashrate (TH/s)", "Block Reward (BTC)", "Avg Fee (BTC)"]])

# --- 仿真结果年度聚合与可视化 ---
df_sim["Year"] = (df_sim["Day"] // 365) + current_year
df_year = df_sim.groupby("Year").agg({
    "Reward ($)": "sum",
    "Cost ($)": "sum",
    "Profit ($)": "sum"
}).reset_index()

plt.figure(figsize=(10,6))
plt.plot(df_year["Year"], df_year["Reward ($)"], label='Block Revenue', color='orange', marker='o')
plt.plot(df_year["Year"], df_year["Cost ($)"], label='Block Cost', color='blue', marker='o')
plt.plot(df_year["Year"], df_year["Profit ($)"], label='Block Profit', color='green', marker='o')
plt.fill_between(df_year["Year"], df_year["Cost ($)"], df_year["Reward ($)"], color='orange', alpha=0.2, label='Gross Margin')
plt.title('Block Revenue, Cost, and Profit Trend (仿真年度趋势)')
plt.xlabel('Year')
plt.ylabel('USD')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# --- Nash Equilibrium Explanation and Game Theory Matrix (for Section 6.3) ---
st.subheader("Nash Equilibrium in the Mining Contest")
st.latex(r"""
h_i^* = \arg\max_{h_i \geq 0} \left\{ \frac{R h_i}{h_i + \sum_{j \neq i} h_j^*} - c_i h_i \right\}
""")
st.markdown("""
In Nash equilibrium, each miner chooses their optimal hashrate (or whether to participate) given the strategies of all other miners. No miner can unilaterally improve their expected profit by changing their own strategy.
""")

st.subheader("Game Theory Payoff Matrix: Two Miners")
import pandas as pd
R = 100  # example total reward
hA, hB = 60, 40  # example hashrates
cA, cB = 0.8, 1.2  # example per-unit costs

# Both mine
pi_A = R * hA / (hA + hB) - cA * hA
pi_B = R * hB / (hA + hB) - cB * hB
# Only A mines
pi_A_prime = R - cA * hA
# Only B mines
pi_B_prime = R - cB * hB

payoff_matrix = pd.DataFrame({
    "Miner B: Mine": [f"({pi_A:.1f}, {pi_B:.1f})", f"(0, {pi_B_prime:.1f})"],
    "Miner B: Not Mine": [f"({pi_A_prime:.1f}, 0)", "(0, 0)"]
}, index=["Miner A: Mine", "Miner A: Not Mine"])
st.table(payoff_matrix)
st.caption("Payoff matrix for two miners. Each cell shows (Miner A payoff, Miner B payoff). Parameters: R=100, hA=60, hB=40, cA=0.8, cB=1.2.")

# --- Game Theory Payoff Matrix: Two Identical Miners (Daily Profit, for Section 6.2) ---
st.subheader("Game Theory Payoff Matrix: Two Identical Miners (Daily Profit)")
daily_profit = 13.92  # from Section 6.2 calculation
payoff_matrix_identical = pd.DataFrame({
    "Miner B: Mine": [f"({daily_profit:.2f}, {daily_profit:.2f})", f"(0, {daily_profit:.2f})"],
    "Miner B: Not Mine": [f"({daily_profit:.2f}, 0)", "(0, 0)"]
}, index=["Miner A: Mine", "Miner A: Not Mine"])
st.table(payoff_matrix_identical)
st.caption("Payoff matrix for two identical miners. Each cell shows (Miner A payoff, Miner B payoff) in USD per day. Daily profit is based on the calculation in Section 6.2.")

