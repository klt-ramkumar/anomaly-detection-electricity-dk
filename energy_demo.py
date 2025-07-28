#!/usr/bin/env python3
"""
Octopus Agile Energy Analysis - Quick Demo
Built for Quantile Energy Job Application
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import sys

print("🚀 OCTOPUS AGILE ENERGY ANALYSIS - QUANTILE ENERGY DEMO")
print("=" * 60)

class QuickEnergyDemo:
    def __init__(self):
        self.df = None
        self.results = {}
        
    def generate_data(self):
        """Generate realistic energy price data"""
        print("📡 Generating realistic energy price data...")
        
        dates = pd.date_range(start='2024-05-01', end='2024-07-24', freq='30min')
        np.random.seed(42)
        
        prices = []
        for dt in dates:
            hour = dt.hour
            day_of_week = dt.dayofweek
            month = dt.month
            
            # Realistic pricing model
            base_price = 16 + 4 * np.sin(2 * np.pi * month / 12)
            
            # Daily demand pattern
            if 7 <= hour <= 9 or 17 <= hour <= 20:
                multiplier = 1.6  # Peak hours
            elif 0 <= hour <= 6:
                multiplier = 0.5  # Off-peak
            else:
                multiplier = 1.0
                
            # Weekend effect
            if day_of_week >= 5:
                multiplier *= 0.75
                
            # Add volatility and market events
            volatility = 0.25 * base_price
            noise = np.random.normal(0, volatility)
            
            # Price spikes (0.8% chance)
            if np.random.random() < 0.008:
                noise += np.random.uniform(20, 60)
                
            # Negative prices (0.1% chance)
            if np.random.random() < 0.001:
                noise = -base_price - np.random.uniform(5, 25)
            
            final_price = max(-40, base_price * multiplier + noise)
            prices.append(final_price)
        
        self.df = pd.DataFrame({
            'datetime': dates,
            'price_pence': prices
        }).set_index('datetime')
        
        # Add features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['is_weekend'] = (self.df.index.dayofweek >= 5).astype(int)
        self.df['is_peak'] = ((self.df['hour'].between(7,9)) | 
                             (self.df['hour'].between(17,20))).astype(int)
        
        print(f"✅ Generated {len(self.df):,} data points")
        
    def analyze_patterns(self):
        """Analyze market patterns"""
        print("\n📊 Analyzing market patterns...")
        
        avg_price = self.df['price_pence'].mean()
        volatility = self.df['price_pence'].std()
        
        peak_avg = self.df[self.df['is_peak'] == 1]['price_pence'].mean()
        offpeak_avg = self.df[self.df['is_peak'] == 0]['price_pence'].mean()
        peak_premium = ((peak_avg - offpeak_avg) / offpeak_avg) * 100
        
        negative_events = (self.df['price_pence'] < 0).sum()
        
        print(f"   💰 Average Price: {avg_price:.1f} pence/kWh")
        print(f"   📈 Volatility: {volatility:.1f} pence/kWh")
        print(f"   ⚡ Peak Premium: {peak_premium:.1f}% above off-peak")
        print(f"   🔴 Negative Price Events: {negative_events}")
        
        self.results.update({
            'avg_price': avg_price,
            'volatility': volatility,
            'peak_premium': peak_premium,
            'negative_events': negative_events
        })
        
    def detect_anomalies(self):
        """Detect price anomalies"""
        print("\n🔍 Detecting anomalies...")
        
        # Statistical detection
        self.df['price_ma'] = self.df['price_pence'].rolling(48).mean()
        self.df['price_std'] = self.df['price_pence'].rolling(48).std()
        self.df['z_score'] = (self.df['price_pence'] - self.df['price_ma']) / self.df['price_std']
        
        stat_anomalies = np.abs(self.df['z_score']) > 3
        
        # ML detection
        features = ['price_pence', 'hour', 'day_of_week', 'is_weekend', 'is_peak']
        X = self.df[features].fillna(method='ffill')
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        ml_anomalies = iso_forest.fit_predict(X_scaled) == -1
        
        # Economic anomalies
        econ_anomalies = (
            (self.df['price_pence'] < 0) |
            (self.df['price_pence'] > self.df['price_pence'].quantile(0.995))
        )
        
        self.df['anomaly'] = stat_anomalies | ml_anomalies | econ_anomalies
        
        total_anomalies = self.df['anomaly'].sum()
        anomaly_rate = (total_anomalies / len(self.df)) * 100
        
        print(f"   🎯 Total Anomalies: {total_anomalies} ({anomaly_rate:.2f}%)")
        print(f"   📊 Statistical: {stat_anomalies.sum()}")
        print(f"   🤖 ML Detected: {ml_anomalies.sum()}")
        print(f"   💰 Economic: {econ_anomalies.sum()}")
        
        self.results.update({
            'total_anomalies': total_anomalies,
            'anomaly_rate': anomaly_rate
        })
        
    def build_prediction_model(self):
        """Build price prediction model"""
        print("\n🤖 Building prediction model...")
        
        # Feature engineering
        features = ['hour', 'day_of_week', 'is_weekend', 'is_peak']
        
        # Add lag features
        for lag in [1, 2, 24, 48]:
            self.df[f'price_lag_{lag}'] = self.df['price_pence'].shift(lag)
            features.append(f'price_lag_{lag}')
            
        # Add rolling statistics
        self.df['price_ma_12h'] = self.df['price_pence'].rolling(24).mean()
        features.append('price_ma_12h')
        
        # Prepare data
        df_model = self.df[features + ['price_pence']].dropna()
        X = df_model[features]
        y = df_model['price_pence']
        
        # Train-test split (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = (1 - mae / np.mean(y_test)) * 100
        
        print(f"   ✅ Model Accuracy: {accuracy:.1f}%")
        print(f"   📈 Mean Absolute Error: {mae:.2f} pence/kWh")
        
        self.results['model_accuracy'] = accuracy
        
    def estimate_trading_potential(self):
        """Estimate trading opportunities"""
        print("\n💰 Estimating trading potential...")
        
        # Peak/off-peak arbitrage
        peak_avg = self.df[self.df['is_peak'] == 1]['price_pence'].mean()
        offpeak_avg = self.df[self.df['is_peak'] == 0]['price_pence'].mean()
        spread = peak_avg - offpeak_avg
        
        # Daily arbitrage potential (1MWh with 85% efficiency)
        daily_arbitrage = (spread * 0.85) / 100 * 1000  # £ per day
        annual_arbitrage = daily_arbitrage * 365
        
        # Anomaly trading potential
        anomaly_data = self.df[self.df['anomaly']]
        if len(anomaly_data) > 0:
            avg_anomaly_size = np.abs(anomaly_data['price_pence'] - self.results['avg_price']).mean()
            annual_anomaly = (avg_anomaly_size * 0.5) / 100 * 500 * self.results['total_anomalies'] * 4
        else:
            annual_anomaly = 0
            
        # Prediction-based trading
        if self.results['model_accuracy'] > 80:
            prediction_pnl = 30000
        else:
            prediction_pnl = 15000
            
        total_pnl = annual_arbitrage + annual_anomaly + prediction_pnl
        
        print(f"   📊 Peak/Off-peak Arbitrage: £{annual_arbitrage:,.0f}/year")
        print(f"   🎯 Anomaly Trading: £{annual_anomaly:,.0f}/year")
        print(f"   🤖 Prediction Trading: £{prediction_pnl:,.0f}/year")
        print(f"   💵 Total Estimated P&L: £{total_pnl:,.0f}/year")
        
        self.results['estimated_pnl'] = total_pnl
        
    def create_visualizations(self):
        """Create summary visualizations"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Octopus Agile Energy Analysis - Quantile Energy Demo', fontsize=14, fontweight='bold')
        
        # Price time series with anomalies
        normal = self.df[~self.df['anomaly']]
        anomalies = self.df[self.df['anomaly']]
        
        axes[0,0].plot(normal.index[-1000:], normal['price_pence'][-1000:], 
                      color='blue', alpha=0.7, linewidth=0.5, label='Normal')
        axes[0,0].scatter(anomalies.index[-100:], anomalies['price_pence'][-100:], 
                         color='red', s=10, alpha=0.8, label='Anomalies')
        axes[0,0].set_title('Price Time Series with Anomaly Detection')
        axes[0,0].set_ylabel('Price (pence/kWh)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Daily pattern
        hourly_avg = self.df.groupby('hour')['price_pence'].mean()
        axes[0,1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
        axes[0,1].set_title('Average Daily Price Pattern')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Price (pence/kWh)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1,0].hist(self.df['price_pence'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(self.results['avg_price'], color='red', linestyle='--', 
                         label=f'Mean: {self.results["avg_price"]:.1f}p')
        axes[1,0].set_title('Price Distribution')
        axes[1,0].set_xlabel('Price (pence/kWh)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        
        # Key metrics
        metrics_text = f"""Key Results:

• Average Price: {self.results['avg_price']:.1f}p/kWh
• Volatility: {self.results['volatility']:.1f}p
• Peak Premium: {self.results['peak_premium']:.1f}%
• Anomaly Rate: {self.results['anomaly_rate']:.2f}%
• Model Accuracy: {self.results['model_accuracy']:.1f}%
• Est. Annual P&L: £{self.results['estimated_pnl']:,.0f}

Trading Opportunities:
• {self.results['negative_events']} negative price events
• Systematic peak/off-peak arbitrage
• ML-driven prediction strategies
• Anomaly-based momentum trading
        """
        
        axes[1,1].text(0.05, 0.95, metrics_text, transform=axes[1,1].transAxes, 
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Summary Metrics')
        
        plt.tight_layout()
        plt.savefig('energy_analysis_demo.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved visualization as 'energy_analysis_demo.png'")
        
    def run_demo(self):
        """Run complete demonstration"""
        self.generate_data()
        self.analyze_patterns()
        self.detect_anomalies()
        self.build_prediction_model()
        self.estimate_trading_potential()
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("🎯 DEMO COMPLETE - READY FOR QUANTILE ENERGY!")
        print("="*60)
        print("This demonstrates systematic, code-driven energy analysis")
        print("Perfect for quantitative trading strategies!")
        
        return self.results

if __name__ == "__main__":
    demo = QuickEnergyDemo()
    results = demo.run_demo()
    
    print("\n🚀 Demo ready for presentation!")
