import streamlit as st
import pandas as pd
import numpy as np

st.title('Stock Data')
@st.cache
@st.cache(suppress_st_warning=True)
def dtf():
    #DATE_COLUMN = 'date/time'
    DATA_URL = ('https://preferred.kotaksecurities.com/security/production/'
                'TradeApiInstruments_FNO_20_02_2022.txt')

    
    dt = pd.read_csv(DATA_URL, sep="|")
    data = dt

        #lowercase = lambda x: str(x).lower()
        #data.rename(lowercase, axis='columns', inplace=True)
        #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        




    data_load_state = st.text('Loading data...')

    data_load_state.text("Done! (using st.cache)")
    return data


def names():

    stock_name = data["instrumentName"]
    #stock_name.sort_values()
    stock_name.drop_duplicates(keep='first',inplace=True)
    #st.write(stock_name)
    option1 = st.selectbox('Select a stock', (stock_name))

    return option1

def expiry(option1):

    option1 = [option1]
    #st.write(option1)
    result_df = data.loc[data['instrumentName'].isin(option1)]
    #st.write(result_df)
    expiry_dates = result_df['expiry']
    

    return expiry_dates

def sort_expiry(expiry_dates ):
    expiry_dates = expiry_dates.drop_duplicates(keep='first', inplace=False)
    #st.write(expiry_dates)
    stock_expiry = st.selectbox('Select an expiry date', (expiry_dates))

    return stock_expiry



def strike_price(stock_expiry, instument_name ):
    stock_expiry = [stock_expiry]
    instument_name = [instument_name]
    stk = data[(data['instrumentName'].isin(instument_name)) & (data['expiry'].isin(stock_expiry))]
    #st.write(stk)
    stock_strike = st.selectbox('Select a strike price', (stk['strike']))

    return stock_strike

def token(instument_name,stock_expiry, strike,opt):

    instument_name = [instument_name]
    stock_expiry = [stock_expiry]
    strike  = [strike]
    opt = [opt]
    stk = data[(data['instrumentName'].isin(instument_name)) & (data['expiry'].isin(stock_expiry)) &(data['strike'].isin(strike)) & (data['optionType'].isin(opt)) ]
    st.write(stk)
    return stk['instrumentToken']

def option_type():
    optype = data['optionType']
    optype = optype.drop_duplicates(keep='first', inplace=False)
    optype = optype.drop(0)
    opt_type = st.selectbox('Select a option type', optype)

    return opt_type




data = dtf()
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
instument_name = names()
exp = expiry(instument_name)
stock_expiry = sort_expiry(exp)
opt_type = option_type()
strike = strike_price(stock_expiry,instument_name)
instru = token(instument_name,stock_expiry,strike,opt_type)
instru = int(instru)
st.write(instru)



from ks_api_client import ks_api
# Defining the host is optional and defaults to https://sbx.kotaksecurities.com/apim
# See configuration.py for a list of all supported configuration parameters.
client = ks_api.KSTradeApi(access_token = "782db5a8-f03d-381b-a148-68c29dbb3037", userid = "VS27011966", consumer_key = "5JLKFQulFfF25gR_Ychw6lA3Geoa",ip = "127.0.0.1", app_id = "test", \
                        host = "https://tradeapi.kotaksecurities.com/apim", consumer_secret = "uFVN2VYgcj7sz6desNbhpbVwHIka")
                     

client.login(password = "Tanay@1699")
client.session_2fa()
ltp = client.quote(instrument_token = instru)
ltp = float(ltp['success'][0]['ltp'])

st.metric(label="Price", value=ltp)





