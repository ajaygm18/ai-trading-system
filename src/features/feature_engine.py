import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class SwingPoint:
    index: int
    price: float
    swing_type: str  # 'high' or 'low'
    
@dataclass
class OrderBlock:
    start_idx: int
    end_idx: int
    high: float
    low: float
    ob_type: str  # 'bullish' or 'bearish'
    is_mitigated: bool = False
    
class FeatureEngine:
    """
    Core feature engineering pipeline that transforms raw price data
    into structured features for ML model consumption
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def identify_swing_points(self, df: pd.DataFrame, n: int = 5) -> List[SwingPoint]:
        """
        Identify swing highs and lows using the N-parameter algorithm
        """
        swing_points = []
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(n, len(df) - n):
            # Check for swing high
            if all(highs[i] > highs[j] for j in range(i-n, i)) and \
               all(highs[i] > highs[j] for j in range(i+1, i+n+1)):
                swing_points.append(SwingPoint(i, highs[i], 'high'))
                
            # Check for swing low
            if all(lows[i] < lows[j] for j in range(i-n, i)) and \
               all(lows[i] < lows[j] for j in range(i+1, i+n+1)):
                swing_points.append(SwingPoint(i, lows[i], 'low'))
                
        return swing_points
    
    def detect_market_structure(self, swing_points: List[SwingPoint]) -> Dict:
        """
        Analyze market structure from swing points
        Returns trend direction and structure breaks
        """
        highs = [sp for sp in swing_points if sp.swing_type == 'high']
        lows = [sp for sp in swing_points if sp.swing_type == 'low']
        
        structure = {
            'trend': 'sideways',
            'bos_bullish': False,
            'bos_bearish': False,
            'mss_bullish': False,
            'mss_bearish': False
        }
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Check for higher highs and higher lows (uptrend)
            recent_highs = highs[-2:]
            recent_lows = lows[-2:]
            
            if (recent_highs[-1].price > recent_highs[-2].price and 
                recent_lows[-1].price > recent_lows[-2].price):
                structure['trend'] = 'uptrend'
                
            # Check for lower highs and lower lows (downtrend)
            elif (recent_highs[-1].price < recent_highs[-2].price and 
                  recent_lows[-1].price < recent_lows[-2].price):
                structure['trend'] = 'downtrend'
                
        return structure
    
    def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) - 3-candle imbalance patterns
        """
        fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG: low[i] > high[i-2]
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                fvg_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                momentum_ratio = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open']) / \
                               (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) if df.iloc[i-1]['high'] != df.iloc[i-1]['low'] else 0
                
                fvgs.append({
                    'index': i,
                    'type': 'bullish',
                    'top': df.iloc[i]['low'],
                    'bottom': df.iloc[i-2]['high'],
                    'size': fvg_size,
                    'momentum_ratio': momentum_ratio,
                    'is_mitigated': False
                })
                
            # Bearish FVG: high[i] < low[i-2]
            elif df.iloc[i]['high'] < df.iloc[i-2]['low']:
                fvg_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
                momentum_ratio = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open']) / \
                               (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) if df.iloc[i-1]['high'] != df.iloc[i-1]['low'] else 0
                
                fvgs.append({
                    'index': i,
                    'type': 'bearish',
                    'top': df.iloc[i-2]['low'],
                    'bottom': df.iloc[i]['high'],
                    'size': fvg_size,
                    'momentum_ratio': momentum_ratio,
                    'is_mitigated': False
                })
                
        return fvgs
    
    def detect_order_blocks(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[OrderBlock]:
        """
        Detect Order Blocks - last opposing candle before strong moves
        """
        order_blocks = []
        
        for sp in swing_points:
            if sp.swing_type == 'high':
                # Look for bullish order block (last bearish candle before move up)
                for i in range(max(0, sp.index - 10), sp.index):
                    if (df.iloc[i]['close'] < df.iloc[i]['open'] and  # bearish candle
                        df.iloc[i+1:sp.index+1]['high'].max() > df.iloc[i]['high'] * 1.02):  # significant move up
                        
                        order_blocks.append(OrderBlock(
                            start_idx=i,
                            end_idx=i,
                            high=df.iloc[i]['high'],
                            low=df.iloc[i]['low'],
                            ob_type='bullish'
                        ))
                        break
                        
            else:  # swing low
                # Look for bearish order block (last bullish candle before move down)
                for i in range(max(0, sp.index - 10), sp.index):
                    if (df.iloc[i]['close'] > df.iloc[i]['open'] and  # bullish candle
                        df.iloc[i+1:sp.index+1]['low'].min() < df.iloc[i]['low'] * 0.98):  # significant move down
                        
                        order_blocks.append(OrderBlock(
                            start_idx=i,
                            end_idx=i,
                            high=df.iloc[i]['high'],
                            low=df.iloc[i]['low'],
                            ob_type='bearish'
                        ))
                        break
                        
        return order_blocks
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        """
        df_features = df.copy()
        
        # Price-based features
        df_features['price_change'] = df_features['close'].pct_change()
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df_features[f'sma_{period}'] = ta.SMA(df_features['close'], timeperiod=period)
            df_features[f'ema_{period}'] = ta.EMA(df_features['close'], timeperiod=period)
            
        # Momentum Indicators
        df_features['rsi'] = ta.RSI(df_features['close'], timeperiod=14)
        df_features['macd'], df_features['macd_signal'], df_features['macd_hist'] = ta.MACD(df_features['close'])
        
        # Volatility Indicators
        df_features['atr'] = ta.ATR(df_features['high'], df_features['low'], df_features['close'], timeperiod=14)
        df_features['bb_upper'], df_features['bb_middle'], df_features['bb_lower'] = ta.BBANDS(
            df_features['close'], timeperiod=20
        )
        
        # Volume Indicators
        df_features['obv'] = ta.OBV(df_features['close'], df_features['volume'])
        
        return df_features
    
    def create_feature_vector(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to create comprehensive feature vector
        """
        self.logger.info("Starting feature engineering pipeline")
        
        # Calculate technical indicators
        df_features = self.calculate_technical_indicators(df)
        
        # Identify swing points
        swing_points = self.identify_swing_points(df)
        
        # Detect market structure
        market_structure = self.detect_market_structure(swing_points)
        
        # Add market structure features
        for key, value in market_structure.items():
            df_features[f'ms_{key}'] = value
            
        # Detect Fair Value Gaps
        fvgs = self.detect_fair_value_gaps(df)
        
        # Add FVG features
        df_features['fvg_bullish_count'] = 0
        df_features['fvg_bearish_count'] = 0
        df_features['fvg_total_size'] = 0
        
        for fvg in fvgs:
            if fvg['type'] == 'bullish':
                if fvg['index'] < len(df_features):
                    df_features.iloc[fvg['index']]['fvg_bullish_count'] += 1
            else:
                if fvg['index'] < len(df_features):
                    df_features.iloc[fvg['index']]['fvg_bearish_count'] += 1
            if fvg['index'] < len(df_features):
                df_features.iloc[fvg['index']]['fvg_total_size'] += fvg['size']
            
        # Detect Order Blocks
        order_blocks = self.detect_order_blocks(df, swing_points)
        
        # Add Order Block features
        df_features['ob_bullish_nearby'] = 0
        df_features['ob_bearish_nearby'] = 0
        
        for ob in order_blocks:
            if ob.ob_type == 'bullish':
                df_features.iloc[ob.start_idx:]['ob_bullish_nearby'] = 1
            else:
                df_features.iloc[ob.start_idx:]['ob_bearish_nearby'] = 1
                
        # Fill NaN values
        df_features.fillna(method='ffill', inplace=True)
        df_features.fillna(0, inplace=True)
        
        self.logger.info(f"Feature engineering complete. Generated {len(df_features.columns)} features")
        
        return df_features
