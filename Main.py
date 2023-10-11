import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv(‘perrin-freres-monthly-champagne-.csv')

df.head()

df.tail()

df.columns=["Month","Sales"]
df.head()

df.drop(106,axis=0,inplace=True)

df.tail()

df.drop(105,axis=0,inplace=True)

df.tail()

df['Month']=pd.to_datetime(df['Month'])

df.head()

df.set_index('Month',inplace=True)

df.head()

df.describe()


Step 2: Visualize the Data

df.plot()
from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df[‘Sales'])
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    

adfuller_test(df[‘Sales'])

df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

df[‘Sales'].shift(1)

df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

df.head(14)

adfuller_test(df['Seasonal First Difference’].dropna())
df['Seasonal First Difference’].plot()

from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()

model_fit.summary()

df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[[‘Sales','forecast']].plot(figsize=(12,8))

import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()


future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 


