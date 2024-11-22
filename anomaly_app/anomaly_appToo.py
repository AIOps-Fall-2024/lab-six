# libraries 
import time
import requests
import pandas as pd

import logging
import json
import os

from datetime import datetime, timezone
from prophet import Prophet
from prometheus_client import Gauge, start_http_server

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# reset training data to 0 origin before HMS conversion
def process_df(df):
    # df['ds'] = df['ds'] - df['ds'].iloc[0]
    
    # total_seconds() converts the timedelta to seconds
    df['ds'] = (df['ds'] - df['ds'].iloc[0]).dt.total_seconds()
    # then do the HMS conversion as usual
    
    # df['ds'] = df['ds'].apply(
    # lambda sec: datetime.fromtimestamp(sec))
    df['ds'] = df['ds'].apply(lambda sec: datetime.fromtimestamp(sec))
    return df

# configure logging to display messages and write them to a file
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('results.log'),
                        logging.StreamHandler()  # display to console
                    ])

# prometheus metrics
anomaly_gauge = Gauge('anomaly_gauge', 'These are the Anomaly Detection numbers')
MAE_gauge = Gauge('MAE', 'Mean Absolute Error')
MAPE_gauge = Gauge('MAPE', 'Mean Absolute Percentage Error')

# start Prometheus server
start_http_server(8002)

# get our metrics from Prometheus
def get_metric(metric):
    url = f"http://35.234.136.57:9090/api/v1/query"
    params = {
        'query': metric,
    }
    response = requests.get(url, params=params)
    data = response.json()
    if not data['data']['result']:
        logging.info(f"No data found for metric: {metric}.")
        return pd.DataFrame()
    metric_data = pd.DataFrame(data['data']['result'][0]['values'], columns=['ds', 'y'])
    metric_data['ds'] = pd.to_datetime(metric_data['ds'], unit='s')
    metric_data['y'] = metric_data['y'].astype(float)
    return metric_data

# we try to get metrics from a json file instead
file_path = "./boutique_training_finalCopy.json"
def get_training_metrics():
    # does the file exist?
    if not os.path.exists(file_path):
        logging.info(f"The file {file_path} does not exist.")
        return pd.DataFrame()
    try:
       # load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # parse the relevant metric data
        results = data.get("data", {}).get("result", [])
        if not results:
            logging.info(f"No data found.")
            return pd.DataFrame()  
        
        # get the values and convert to a DataFrame
        metric_data = pd.DataFrame(results[0]['values'], columns=['ds', 'y'])
        metric_data['ds'] = pd.to_datetime(metric_data['ds'], unit='s') 
        metric_data['y'] = metric_data['y'].astype(float)  
        return metric_data
    
    except Exception as e:
        logging.error(f"Error reading or parsing the JSON file: {e}")
        return pd.DataFrame()

# using the timeseries forecasting model
def prophet_process(df_train):
    model = Prophet(interval_width=0.99, growth='flat', yearly_seasonality=False, 
                    weekly_seasonality=False, daily_seasonality=True)
    model.fit(df_train)
    return model

# now we anomaly detect
def anomaly_detection(df_test, forecast):
    performance = pd.merge(df_test, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    performance['anomaly'] = performance.apply(
        lambda row: 1 if (row['y'] < row['yhat_lower'] or row['y'] > row['yhat_upper']) else 0, axis=1)
    anomalies = performance[performance['anomaly'] == 1]
    anomaly_count = len(anomalies)

    mae = mean_absolute_error(performance['y'], performance['yhat'])
    mape = mean_absolute_percentage_error(performance['y'], performance['yhat'])

    # update Prometheus metrics
    anomaly_gauge.set(anomaly_count)
    MAE_gauge.set(mae)
    MAPE_gauge.set(mape)

    # log results
    current_time = datetime.now()
    results = pd.DataFrame({
        'current_time': [current_time],
        'anomaly_count': [anomaly_count],
        'MAE': [mae],
        'MAPE': [mape]
    })

    separator = '+' * 50
    logging.info(separator)
    logging.info('Training and Forecast Time Results')
    logging.info(results.to_string(index=False))
    logging.info(separator)

    anomaly_status = "Anomalies found:" if not anomalies.empty else "No anomalies detected."
    logging.info(anomaly_status)

    if not anomalies.empty:
        logging.info(anomalies[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))

# main function
if __name__ == "__main__":
     # fetch the training data (last 5 minutes) for 'train_gauge'
    df_train = get_training_metrics()
    if df_train.empty:
        logging.info("The Train Data is empty. We will Retry in 60 seconds.")
        time.sleep(60)
    df_train = process_df(df_train)    

    # now we build the Prophet model
    model = prophet_process(df_train)

    time.sleep(60)
    while True:
        try:
            test_query = "histogram_quantile(0.5, rate(istio_request_duration_milliseconds_bucket{source_app=\"frontend\", destination_app=\"shippingservice\", reporter=\"source\"}[]))"
            df_test = get_metric(test_query)
            if df_test.empty:
                logging.info("The Test data is empty. Retrying...")
                continue
            
           df_test = process_df(df_test)
            
            # make predictions
            forecast = model.predict(df_test)

            # run anomaly detection and log results
            anomaly_detection(df_test, forecast)

            # sleep before the next loop
            # thus running infinetly
            time.sleep(60)

        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(60)
