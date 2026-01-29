import yfinance as yf
import pandas as pd

symbol = "FPADX"
data = yf.download(symbol, period="5d", progress=False)

print(f"Index type: {type(data.index)}")
if not data.empty:
    print(f"First index value: {data.index[0]}")
    print(f"TZ Info: {data.index[0].tzinfo}")
    
# Check comparison with naive
naive_date = pd.Timestamp.now().normalize()
print(f"Naive Date: {naive_date}")
print(f"Comparison (Naive in Index?): {naive_date in data.index}") 
# Note: this comparison will error if mixed tz, or just act weird.
