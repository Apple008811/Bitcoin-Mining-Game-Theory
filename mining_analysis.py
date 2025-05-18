import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="Bitcoin Mining Analysis",
    page_icon="⛏️",
    layout="wide"
)

# 常量定义
ASSUMPTIONS = {
    # R&D相关假设
    'rnd_efficiency': 0.05,  # 每单位R&D投入带来的效率提升
    'spillover_rate': 0.2,   # 技术溢出率
    'rnd_cost_factor': 0.1,  # R&D成本系数
    
    # 运营成本假设
    'electricity_cost': 0.05,  # 美元/kWh
    'maintenance_cost': 0.1,   # 设备成本的年维护费用
    'cooling_cost': 0.05,      # 设备成本的年冷却费用
    'labor_cost': 0.15,        # 设备成本的年人工费用
    
    # 设备相关假设
    'hardware_lifetime': 2,    # 设备使用寿命（年）
    'efficiency_improvement': 0.1,  # 每年能效提升
    'depreciation_rate': 0.5,  # 年折旧率
    'hardware_cost_per_th': 100,  # 每TH/s的设备成本（美元）
}

class MiningDataCollector:
    def __init__(self):
        self.btc_price = None
        self.block_height = None
        self.block_reward = None
        self.pool_data = None
        
    def fetch_btc_price(self):
        """获取BTC价格"""
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        resp = requests.get(url)
        try:
            data = resp.json()
            st.write("BTC Price API 返回内容：", data)  # 调试输出
            self.btc_price = data['bitcoin']['usd']
        except Exception as e:
            st.error(f"获取BTC价格失败: {e}")
            self.btc_price = None
        return self.btc_price
    
    def fetch_block_height(self):
        """获取当前区块高度"""
        url = "https://blockchain.info/q/getblockcount"
        resp = requests.get(url)
        self.block_height = int(resp.text)
        return self.block_height
    
    def get_block_reward(self):
        """计算当前区块奖励"""
        halvings = self.block_height // 210000
        initial_reward = 50
        self.block_reward = initial_reward / (2 ** halvings)
        return self.block_reward
    
    def fetch_pool_data(self):
        """获取矿池数据"""
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
        
        df = pd.DataFrame({
            'Pool': pools,
            'Blocks': block_counts
        })
        df['Total Blocks'] = df['Blocks'].sum()
        df['Hashrate (%)'] = df['Blocks'] / df['Total Blocks'] * 100
        self.pool_data = df
        return df

class MiningCostCalculator:
    def __init__(self, assumptions=ASSUMPTIONS):
        self.assumptions = assumptions
    
    def calculate_mining_costs(self, hash_rate):
        """计算挖矿总成本"""
        # 设备成本
        hardware_cost = hash_rate * self.assumptions['hardware_cost_per_th']
        
        # 电费
        electricity_cost = hash_rate * self.assumptions['electricity_cost']
        
        # 运营成本
        maintenance = hardware_cost * self.assumptions['maintenance_cost']
        cooling = hardware_cost * self.assumptions['cooling_cost']
        labor = hardware_cost * self.assumptions['labor_cost']
        
        return {
            'hardware': hardware_cost,
            'electricity': electricity_cost,
            'maintenance': maintenance,
            'cooling': cooling,
            'labor': labor,
            'total': hardware_cost + electricity_cost + maintenance + cooling + labor
        }
    
    def calculate_rnd_impact(self, rnd_investment):
        """计算R&D投入的影响"""
        efficiency_gain = rnd_investment * self.assumptions['rnd_efficiency']
        spillover = rnd_investment * self.assumptions['spillover_rate']
        return {
            'efficiency_gain': efficiency_gain,
            'spillover': spillover,
            'cost': rnd_investment * self.assumptions['rnd_cost_factor']
        }

def main():
    st.title("⛏️ Bitcoin Mining Analysis Dashboard")
    
    # 创建数据收集器和成本计算器
    data_collector = MiningDataCollector()
    cost_calculator = MiningCostCalculator()
    
    # 侧边栏 - 参数设置
    st.sidebar.header("Parameters")
    
    # 成本参数
    st.sidebar.subheader("Cost Parameters")
    electricity_cost = st.sidebar.number_input("Electricity Cost ($/kWh)", 
                                             value=ASSUMPTIONS['electricity_cost'],
                                             step=0.01)
    hardware_cost = st.sidebar.number_input("Hardware Cost ($/TH/s)", 
                                          value=ASSUMPTIONS['hardware_cost_per_th'],
                                          step=10)
    
    # R&D参数
    st.sidebar.subheader("R&D Parameters")
    rnd_efficiency = st.sidebar.slider("R&D Efficiency", 
                                     min_value=0.01, 
                                     max_value=0.2, 
                                     value=ASSUMPTIONS['rnd_efficiency'],
                                     step=0.01)
    spillover_rate = st.sidebar.slider("Spillover Rate", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     value=ASSUMPTIONS['spillover_rate'],
                                     step=0.05)
    
    # 更新假设值
    ASSUMPTIONS.update({
        'electricity_cost': electricity_cost,
        'hardware_cost_per_th': hardware_cost,
        'rnd_efficiency': rnd_efficiency,
        'spillover_rate': spillover_rate
    })
    
    # 获取数据
    with st.spinner('Fetching data...'):
        btc_price = data_collector.fetch_btc_price()
        block_height = data_collector.fetch_block_height()
        block_reward = data_collector.get_block_reward()
        pool_data = data_collector.fetch_pool_data()
    
    # 显示基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BTC Price", f"${btc_price:,.2f}")
    with col2:
        st.metric("Block Height", f"{block_height:,}")
    with col3:
        st.metric("Block Reward", f"{block_reward:.8f} BTC")
    
    # 计算成本和收益
    pool_data['Costs'] = pool_data['Hashrate (%)'].apply(
        lambda x: cost_calculator.calculate_mining_costs(x)
    )
    pool_data['Revenue'] = pool_data['Hashrate (%)'] * block_reward * btc_price / 100
    pool_data['Profit'] = pool_data['Revenue'] - pool_data['Costs'].apply(lambda x: x['total'])
    
    # 显示主要矿池数据
    st.subheader("Top Mining Pools")
    top_pools = pool_data.nlargest(10, 'Hashrate (%)')
    st.dataframe(top_pools[['Pool', 'Hashrate (%)', 'Revenue', 'Profit']].style.format({
        'Hashrate (%)': '{:.2f}%',
        'Revenue': '${:,.2f}',
        'Profit': '${:,.2f}'
    }))
    
    # 可视化
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hashrate Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_pools, x='Pool', y='Hashrate (%)', ax=ax1)
        plt.xticks(rotation=45)
        plt.title('Top 10 Mining Pools by Hashrate')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Profit Analysis")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_pools, x='Pool', y='Profit', ax=ax2)
        plt.xticks(rotation=45)
        plt.title('Top 10 Mining Pools by Profit')
        st.pyplot(fig2)
    
    # 中心化分析
    st.subheader("Centralization Analysis")
    hhi = sum((h/100)**2 for h in pool_data['Hashrate (%)'])
    st.metric("Herfindahl-Hirschman Index (HHI)", f"{hhi:.4f}")
    
    # 添加解释
    st.info("""
    **HHI Interpretation:**
    - HHI < 0.15: Highly decentralized
    - 0.15 ≤ HHI < 0.25: Moderately decentralized
    - 0.25 ≤ HHI < 0.5: Moderately centralized
    - HHI ≥ 0.5: Highly centralized
    """)
    
    # 成本构成分析
    st.subheader("Cost Structure Analysis")
    top_pool_costs = pd.DataFrame([cost for cost in top_pools['Costs']])
    top_pool_costs['Pool'] = top_pools['Pool']
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    top_pool_costs.set_index('Pool').plot(kind='bar', stacked=True, ax=ax3)
    plt.title('Cost Structure of Top Mining Pools')
    plt.xticks(rotation=45)
    plt.legend(title='Cost Type')
    st.pyplot(fig3)

if __name__ == "__main__":
    main() 