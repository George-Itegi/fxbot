from database.db_manager import get_connection
conn = get_connection()
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT COUNT(*) as total FROM backtest_trades")
r = cursor.fetchone()
print(f"backtest_trades total: {r['total']}")

cursor.execute("SELECT source, COUNT(*) as cnt FROM backtest_trades GROUP BY source")
for row in cursor.fetchall():
    print(f"  source={row['source']}: {row['cnt']}")

cursor.execute("""
    SELECT strategy, COUNT(*) as cnt, SUM(win) as wins,
    ROUND(AVG(profit_r),3) as avg_r
    FROM backtest_trades WHERE profit_r IS NOT NULL
    GROUP BY strategy ORDER BY cnt DESC
""")
print("Strategy breakdown:")
for row in cursor.fetchall():
    print(f"  {row['strategy']}: {row['cnt']} trades | wins={row['wins']} | avg_r={row['avg_r']}")

cursor.execute("""
    SELECT MIN(profit_r) as min_r, MAX(profit_r) as max_r,
    AVG(profit_r) as avg_r, STDDEV(profit_r) as std_r
    FROM backtest_trades WHERE profit_r IS NOT NULL
""")
r = cursor.fetchone()
print(f"profit_r dist: min={r['min_r']:.3f} max={r['max_r']:.3f} avg={float(r['avg_r']):.3f} std={float(r['std_r']):.3f}")

cursor.execute("""
    SELECT market_state, COUNT(*) as cnt, ROUND(AVG(profit_r),3) as avg_r
    FROM backtest_trades WHERE profit_r IS NOT NULL
    GROUP BY market_state ORDER BY avg_r DESC
""")
print("Market state breakdown:")
for row in cursor.fetchall():
    print(f"  {row['market_state']}: {row['cnt']} trades | avg_r={row['avg_r']}")

cursor.execute("""
    SELECT session, COUNT(*) as cnt, ROUND(AVG(profit_r),3) as avg_r
    FROM backtest_trades WHERE profit_r IS NOT NULL
    GROUP BY session ORDER BY avg_r DESC
""")
print("Session breakdown:")
for row in cursor.fetchall():
    print(f"  {row['session']}: {row['cnt']} trades | avg_r={row['avg_r']}")

try:
    cursor.execute("SELECT COUNT(*) as total FROM signals")
    r = cursor.fetchone()
    print(f"signals table total: {r['total']}")
except Exception as e:
    print(f"signals table: {e}")

conn.close()
print("DONE")
