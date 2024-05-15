import streamlit as st
from utils.prediction_process_manager import Model
from utils.stock_analyzer import Analyzer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

model = Model()
stock_names = ['S&P 100', 'Apple','AbbVie','Abbott','Accenture','Adobe','American International Group','AMD','Amgen','American Tower','Amazon','Broadcom','American Express','Boeing','Bank of America','BNY Mellon','Booking Holdings','BlackRock','Bristol Myers Squibb','Berkshire Hathaway','Citigroup','Caterpillar','Charter Communications','Colgate-Palmolive','Comcast','Capital One','ConocoPhillips','Costco','Salesforce','Cisco','CVS Health','Chevron','Danaher','Disney','Dow','Duke Energy','Emerson','Exelon','Ford','FedEx','General Dynamics','GE','Gilead','GM','GOOGLE','Goldman Sachs','Home Depot','Honeywell','IBM','Intel','Johnson & Johnson','JPMorgan Chase','Kraft Heinz','Coca-Cola','Linde','Lilly','Lockheed Martin',"Lowe's",'Mastercard',"McDonald's",'Mondelēz International','Medtronic','MetLife','Meta','3M','Altria','Merck','Morgan Stanley','Microsoft','NextEra Energy','Netflix','Nike','Nvidia','Oracle','PepsiCo','Pfizer','Procter & Gamble','Philip Morris International','PayPal','Qualcomm','Raytheon Technologies','Starbucks','Charles Schwab','Southern Company','Simon','AT&T','Target','Thermo Fisher Scientific','T-Mobile','Tesla','Texas Instruments','UnitedHealth Group','Union Pacific','United Parcel Service','U.S. Bank','Visa','Verizon','Walgreens Boots Alliance','Wells Fargo','Walmart','ExxonMobil']
stock_symbols = ['^OEX','AAPL','ABBV','ABT','ACN','ADBE','AIG','AMD','AMGN','AMT','AMZN','AVGO','AXP','BA','BAC','BK','BKNG','BLK','BMY','BRK.B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CRM','CSCO','CVS','CVX','DHR','DIS','DOW','DUK','EMR','EXC','F','FDX','GD','GE','GILD','GM','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KHC','KO','LIN','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SCHW','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']

analyzer = Analyzer() 

st.set_page_config(
    page_title = "Stock Price Prediction App",
    menu_items={
     'Report a Bug': "mailto:ozguraslank@gmail.com",    
     'About': "# Stock Price Prediction Web App\n This project is made by Ozgur Aslan. You can reach me out from the following e-mail:  ozguraslank@gmail.com"
       })

tabs = ["Future Prediction", "Data Analysis"]

page = st.sidebar.radio("Tabs", tabs)

if page == "Future Prediction":
    st.markdown("<h2 style = 'text-align:center;'> Stock Price Prediction </h2>", unsafe_allow_html = True)
    st.write("In this page, you can see the next 90 days price predictions of the stock you select")

    selected_stock = st.selectbox("*Select a Stock*", stock_names)

    # Find the stock symbol of the selected stock
    selected_stock_symbol = stock_symbols[stock_names.index(selected_stock)]   
    
    button = st.button("Predict")

    if button:
        try:
            with st.spinner("Please wait..."):
                pred_df = model.main(selected_stock_symbol)
                st.success("Prediction is done!")
            fig = model.show_actual_pred(selected_stock)
            st.plotly_chart(fig)
        except Exception as e:
            print(f"Error during the prediction for the stock! {e}")
            
if page == "Data Analysis":
    st.markdown("<h2 style = 'text-align:center;'> Stock Analysis </h2>", unsafe_allow_html = True)
    st.write("In this page, you can see brief analysis of the stock you select.")
    st.write("Please select the start date and end date from the menu in left ")

    selected_stock = st.selectbox("*Select a Stock*", stock_names)
    selected_start_date = str(st.sidebar.date_input(label = "*Start Date*", value = dt.date.today() - dt.timedelta(days = 30), max_value = dt.date.today()))
    selected_end_date = str(st.sidebar.date_input(label = "*End Date*", value = dt.date.today(), max_value = dt.date.today()))

    # Find the stock symbol of the selected stock
    selected_stock_symbol = stock_symbols[stock_names.index(selected_stock)]
    
    
    button = st.button("Analyze")

    if button:
        with st.spinner("Please wait..."):
            df = analyzer.download_data(
                selected_stock_symbol,
                selected_start_date,
                selected_end_date)
            
        if (df.empty):
            st.write("No data found, please try another date range")

        else:
            st.write(df.describe().T.drop("count", axis = 1))
            close_price_in_time_fig = analyzer.show_close_price_in_time(selected_stock)
            close_volume_in_time_fig = analyzer.show_volume_close_in_time(selected_stock)
            st.plotly_chart(close_price_in_time_fig)
            st.plotly_chart(close_volume_in_time_fig)   


