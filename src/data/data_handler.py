import pandas as pd
import numpy as np
import websocket
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import threading
import time
from dataclasses import dataclass

@dataclass
class MarketTick:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class DataHandler:
    """
    Core data handler for market data ingestion and management
    Supports both real-time WebSocket feeds and historical data retrieval
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config.get('database_path', 'data/market_data.db')
        self.symbols = config.get('symbols', ['NIFTY', 'BANKNIFTY'])
        self.callbacks = []
        self.ws = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing market data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON market_data(symbol, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
    def add_callback(self, callback: Callable[[MarketTick], None]):
        """Add callback function to be called when new data arrives"""
        self.callbacks.append(callback)
        
    def store_tick(self, tick: MarketTick):
        """Store market tick in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tick.symbol, tick.timestamp, tick.open, tick.high, 
              tick.low, tick.close, tick.volume))
        
        conn.commit()
        conn.close()
        
    def get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Retrieve historical data from database"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM market_data 
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        return df
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Parse market data (format depends on data provider)
            if 'symbol' in data and 'ltp' in data:
                tick = MarketTick(
                    symbol=data['symbol'],
                    timestamp=datetime.now(),
                    open=data.get('open', 0),
                    high=data.get('high', 0),
                    low=data.get('low', 0),
                    close=data['ltp'],
                    volume=data.get('volume', 0)
                )
                
                # Store in database
                self.store_tick(tick)
                
                # Notify callbacks
                for callback in self.callbacks:
                    callback(tick)
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.info("WebSocket connection closed")
        self.is_connected = False
        
    def on_open(self, ws):
        """Handle WebSocket open"""
        self.logger.info("WebSocket connection opened")
        self.is_connected = True
        
        # Subscribe to symbols
        for symbol in self.symbols:
            subscribe_msg = {
                "action": "subscribe",
                "symbol": symbol
            }
            ws.send(json.dumps(subscribe_msg))
            
    def start_streaming(self):
        """Start real-time data streaming"""
        if self.config.get('data_source') == 'websocket':
            ws_url = self.config.get('websocket_url')
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
    def stop_streaming(self):
        """Stop real-time data streaming"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
