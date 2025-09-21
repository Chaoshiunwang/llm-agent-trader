 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/backend/app/backtesting/engine.py b/backend/app/backtesting/engine.py
index 633cad779153ab3e01d0fedc223c6e4a82ac58d4..b1668e4346df09bd1ba708d0766f2a35c458b5d7 100644
--- a/backend/app/backtesting/engine.py
+++ b/backend/app/backtesting/engine.py
@@ -713,59 +713,57 @@ class CustomBacktestEngine:
             data: 股價數據
             trades: 交易記錄
             signals: 原始交易信號
             symbol: 股票代碼
             strategy_name: 策略名稱
 
         Returns:
             績效指標字典
         """
         if not portfolio_history:
             raise ValueError("無投資組合歷史數據")
 
         # 基本資訊
         start_date = portfolio_history[0]["date"]
         end_date = portfolio_history[-1]["date"]
         total_days = len(portfolio_history)
 
         # 最終數值
         final_value = portfolio_history[-1]["total_value"]
         final_return = portfolio_history[-1]["cumulative_return"]
 
         # 計算年化報酬率
         days_in_year = 365.25
         years = total_days / days_in_year
         annual_return = (
-            (final_value / self.config.initial_capital) ** (1 / years) - 1
-            if years > 0
-            else 0
+            (final_value / initial_cash) ** (1 / years) - 1 if years > 0 else 0
         )
 
         # 計算波動率
-        returns = [ph["cumulative_return"] for ph in portfolio_history]
-        returns_series = pd.Series(returns)
-        daily_returns = returns_series.pct_change().dropna()
+        total_values = [ph["total_value"] for ph in portfolio_history]
+        values_series = pd.Series(total_values, dtype="float")
+        daily_returns = values_series.pct_change().dropna()
         volatility = daily_returns.std() * np.sqrt(252)  # 年化波動率
 
         # 處理 NaN 值
         if pd.isna(volatility):
             volatility = 0.0
 
         # 計算最大回撤
         values = [ph["total_value"] for ph in portfolio_history]
         cummax = pd.Series(values).cummax()
         drawdown = (pd.Series(values) - cummax) / cummax
         max_drawdown = drawdown.min()
 
         # 處理 NaN 值
         if pd.isna(max_drawdown):
             max_drawdown = 0.0
 
         # 交易統計
         num_trades = len(trades)
         buy_trades = [t for t in trades if t.order_type == OrderType.BUY]
         sell_trades = [t for t in trades if t.order_type == OrderType.SELL]
 
         # 計算勝率（需要配對買賣交易）
         win_rate = 0.0
         if len(buy_trades) > 0 and len(sell_trades) > 0:
             # 簡化勝率計算：比較買入和賣出價格
 
EOF
)
