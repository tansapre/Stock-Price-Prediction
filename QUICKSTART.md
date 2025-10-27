# Quick Start Guide 🚀

Get started with Stock Price Prediction AI in 5 minutes!

## Step 1: Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Step 2: Launch the App

```bash
# Start the Streamlit web application
streamlit run stock_prediction_app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 3: Authenticate with Zerodha

### Your Credentials:
- **API Key:** di924ht7h955col3
- **API Secret:** 4988yz4r2tg1n5relibczd8g98fsja44

### Authentication Steps:
1. In the app sidebar, your credentials are pre-filled
2. Click **"Get Login URL"** button
3. A login URL will appear - click it to open Zerodha login
4. Log in with your Zerodha credentials
5. After authorization, you'll be redirected to a URL like:
   ```
   http://127.0.0.1/?request_token=XXXXXXXXXX&action=login&status=success
   ```
6. Copy the `request_token` value (everything after `request_token=` and before `&action`)
7. Paste it in the "Request Token" field in the app
8. Click **"Authenticate"**

## Step 4: Train the Model

1. Go to the **"Model Training"** tab
2. Configure training parameters:
   - **Stock Symbol:** RELIANCE (or any NSE stock)
   - **Training Days:** 730 (2 years recommended)
   - **Epochs:** 100
   - **Batch Size:** 32
3. Click **"Start Training"**
4. Wait for training to complete (5-15 minutes)
5. Check that RMSE < 1% ✓

## Step 5: Make Predictions

1. Go to the **"Predictions"** tab
2. Enter stock symbol (e.g., RELIANCE, TCS, INFY)
3. Click **"Predict Next Day Prices"**
4. View predictions with confidence intervals

## Alternative: Command Line Usage

### Train Model
```bash
python train_model.py --symbol RELIANCE --days 730 --epochs 100
```

### Make Predictions
After authentication, create a script:

```python
from predict import predict_for_symbol

prediction = predict_for_symbol(
    symbol='RELIANCE',
    api_key='di924ht7h955col3',
    api_secret='4988yz4r2tg1n5relibczd8g98fsja44',
    access_token='your_access_token',
    exchange='NSE'
)

print(f"Next Day Open: {prediction['next_day_open']:.2f}")
print(f"Next Day Close: {prediction['next_day_close']:.2f}")
```

## Troubleshooting

### Issue: Authentication fails
- **Solution:** Make sure you're using correct API credentials
- Verify that your Kite Connect app is active
- Check if you copied the complete request_token

### Issue: No data returned
- **Solution:** Check if the stock symbol is correct
- Verify the exchange (NSE/BSE)
- Ensure the stock has historical data available

### Issue: Model training fails
- **Solution:** Ensure you have enough RAM (4GB+ recommended)
- Reduce batch size if out of memory
- Check that TensorFlow is properly installed

### Issue: RMSE > 1%
- **Solution:** Train with more data (1000+ days)
- Increase epochs (150-200)
- Try different stocks (some are more predictable)

## Tips for Best Results

1. **Data Quality:**
   - Use at least 2 years of historical data
   - More data = better predictions

2. **Stock Selection:**
   - Large-cap stocks (RELIANCE, TCS, INFY) work best
   - High trading volume stocks are more predictable

3. **Model Maintenance:**
   - Retrain model monthly
   - Monitor prediction accuracy
   - Adjust hyperparameters as needed

4. **Risk Management:**
   - Never rely solely on AI predictions
   - Use stop-loss orders
   - Diversify your portfolio

## Sample Stocks to Try

### Large Cap (Recommended for beginners)
- RELIANCE
- TCS
- INFY
- HDFCBANK
- ICICIBANK

### Mid Cap
- ADANIPORTS
- BAJFINANCE
- MARUTI

### High Volume
- SBIN
- TATASTEEL
- AXISBANK

## Next Steps

1. **Explore Features:**
   - View historical data with candlestick charts
   - Check training metrics and plots
   - Compare predictions with actual prices

2. **Customize:**
   - Adjust lookback period (30-90 days)
   - Modify LSTM architecture in `lstm_model.py`
   - Add more technical indicators in `data_preprocessing.py`

3. **Production Use:**
   - Set up automated training pipeline
   - Create alerts for prediction changes
   - Build trading strategy based on predictions

## Support

Need help? Check:
- **README.md** - Full documentation
- **Code comments** - Detailed explanations
- **GitHub Issues** - Report bugs or ask questions

---

**Happy Trading! 📈**

Remember: This is an educational tool. Always do your own research before making investment decisions.
