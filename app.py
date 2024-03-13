# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

# %%
stock = st.radio("Choose stock", ['AAPL', 'GOOG', 'MSFT'], captions=['Apple', 'Google', 'Microsoft'])


# %%
from dateutil.relativedelta import relativedelta

# stock = "AAPL"
end = datetime.today()
# go back 12 months
go_back = st.slider('Number of months of historic data', 3, 36, 6)
start = end - relativedelta(months=go_back)

# %%
close = pd.DataFrame()
close['Price'] = yf.download(stock, start=start, end=end)['Close']

# %%
close.head()

# %%
close.describe()

# %%
# check for missing values
close.isna().sum()

# %%
# plot the data
close.plot(figsize=(10, 7), title=f"{stock} Price")
plt.ylabel("Price")
plt.show()

# %%
from pandas.plotting import lag_plot
plt.figure()
lag_plot(close['Price'], lag=1)

# %% [markdown]
# # Normality Tests

# %% [markdown]
# ## Visual Normality Checks

# %% [markdown]
# ### Histogram

# %%
plt.hist(close['Price'], bins=50)

# %% [markdown]
# The distrubition does not display a classic bell curve shape. We can try to transform the data to a nepreian log scale to see if that helps.

# %%
# log of the price
close['lPrice'] = np.log(close['Price'])
close.head()

# %%
plt.hist(close['lPrice'], bins=50)

# %% [markdown]
# lPrice seems to be more normally distributed than Price. However, it is skewed to the right.

# %% [markdown]
# ### QQ Plot

# %%
## qq plot of price
from scipy import stats

stats.probplot(close['Price'], dist="norm", plot=plt)
plt.show()

# %%
from scipy import stats

stats.probplot(close['lPrice'], dist="norm", plot=plt)
plt.show()

# %% [markdown]
# From the QQ plot, we can see that the data can be considered normal for the most part. However, there are some outliers that are skewing the data. 
# We also conclude that the lPrice and Price are pretty much the same.

# %% [markdown]
# ## Statistical Normality Tests

# %% [markdown]
# ### Shapiro-Wilk Test

# %%
from scipy.stats import shapiro

stat, p = shapiro(close['Price'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# H0: The data is normally distributed
# p-value: 0.0 < 0.05 => reject H0
# The Shapiro-Wilk test rejects the null hypothesis that the data is normally distributed.

# %%
from scipy.stats import shapiro

stat, p = shapiro(close['lPrice'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# We find that also lPrice is not normally distributed through the Shapiro-Wilk test.

# %% [markdown]
# ### Skewness and Kurtosis

# %%
# skewness and kurtosis
print("Skewness: %f" % close['Price'].skew())
print("Kurtosis: %f" % close['Price'].kurt())

# %% [markdown]
# Skewness is positive, which means that the data is skewed to the right. Kurtosis is negative, which means that the data is platykurtic.

# %% [markdown]
# ### Kolmogorov-Smirnov Test

# %%
# kolmogorov-smirnov test
from scipy.stats import kstest

stat, p = kstest(close['Price'], 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %%
from scipy.stats import kstest

stat, p = kstest(close['lPrice'], 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# Kolmorogov-Smirnov test also rejects the null hypothesis that the data is normally distributed.

# %% [markdown]
# We will now transform the data using Square Root Transformation to see if that helps.

# %%
# transform the data using Square root and test for normality using visual and statistical methods
close['sqrtPrice'] = np.sqrt(close['Price'])

close['sqrtPrice'].plot(figsize=(10, 7), title=f"{stock} Square Root Price")
plt.ylabel("Price")
plt.show()

plt.hist(close['sqrtPrice'], bins=50)
plt.show()

stats.probplot(close['sqrtPrice'], dist="norm", plot=plt)
plt.show()

from scipy.stats import shapiro

stat, p = shapiro(close['sqrtPrice'])

print('Statistics=%.3f, p=%.3f' % (stat, p))

from scipy.stats import kstest
stat, p = kstest(close['sqrtPrice'], 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))



# %%
close['cbrtPrice'] = np.cbrt(close['Price'])

close['cbrtPrice'].plot(figsize=(10, 7), title=f"{stock} Cube Root Price")
plt.ylabel("Price")
plt.show()

plt.hist(close['cbrtPrice'], bins=50)
plt.show()

stats.probplot(close['cbrtPrice'], dist="norm", plot=plt)
plt.show()

stat, p = shapiro(close['cbrtPrice'])
print('Statistics=%.3f, p=%.3f' % (stat, p))

stat, p = kstest(close['cbrtPrice'], 'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# <font color='red'>All transformations failed to make the data normally distributed. We won't be using the transformed data.</font>

# %% [markdown]
# # Stationarity Tests

# %% [markdown]
# KPSS Test
# H0: The data is stationary
# p-value: 0.0 < 0.05 => reject H0

# %%
# KPSS test
from statsmodels.tsa.stattools import kpss

stat, p, lags, crit = kpss(close['Price'], 'c')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# Data is not stationary. We will need to transform the data to make it stationary.

# %% [markdown]
# Differencing the data once to make it stationary.

# %%
# difference the data
close['diff'] = close['Price'].diff().dropna()
close['diff'].plot(figsize=(10, 7), title=f"{stock} Price Difference")
plt.ylabel("Price")
plt.show()

# %%
close['diff'].hist(bins=50)

# %%
close.head()

# %%
close = close.dropna()
stat, p, lags, crit = kpss(close['diff'], 'c')
print('Statistics=%.3f, p=%.3f' % (stat, p))

# %% [markdown]
# Data is now stationary and normally distributed.

# %%
# ADF test
from statsmodels.tsa.stattools import adfuller

result = adfuller(close["Price"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')    

# %% [markdown]
# The ADF test is non conclusive. The ADF statistic is greater than the critical values, however the p-value is greater than 0.05. Therefore, we cannot reject the null hypothesis that the data is non-stationary.

# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(close["diff"], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'n_lags: {result[1]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')    

# %% [markdown]
# The ADF test for the differenced data is conclusive. The ADF statistic is less than the critical values, and the p-value is less than 0.05. Therefore, we can reject the null hypothesis that the data is non-stationary.

# %% [markdown]
# # Model Selection

# %% [markdown]
# ## ACF and PACF

# %%
# ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(close['diff'], lags=50)
plt.show()

plot_pacf(close['diff'], lags=50)
plt.show()

# %% [markdown]
# The ACF and PACF plots suggest an ARIMA(1,1,0) model. The ACF plot shows a tailing off pattern, which suggests an AR model. The PACF plot displays a sharp cut off at lag 1, which suggests an AR(1) model.

# %%
close_.head()

# %%
# ARIMA model
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(close_['Price'], order=(1, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# %%
p = 22
d=1
q=21

# %%
# ARIMA model
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(close_['Price'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# %%
# auto arima
from pmdarima.arima import auto_arima

auto_model = auto_arima(close_['Price'], trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, scoring='mse')
auto_model_fit=auto_model.fit(close_['Price'])
print(auto_model.summary())

# %% [markdown]
# Auto ARIMA suggests an ARIMA(0,1,0) model, however information indicatators show that there is no significant difference between the ARIMA(0,1,0) and ARIMA(1,1,0) models. The ar coefficient is also significant in the ARIMA(1,1,0) model, therefore we will use the ARIMA(1,1,0) model.
# We also have to test the normality of the residuals to make sure that the model is a good fit.

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# ### Residuals

# %%
model_fit.plot_diagnostics()
plt.show()

# %%

test = model_fit.test_normality('jarquebera')
print('Test Statistic: %.2f, p-value: %.2f, Skew: %.2f, Kurtosis: %.2f' % (test[0][0], test[0][1], test[0][2], test[0][3]))

# %% [markdown]
# The p value of the Jarque-Bera test is less than 0.05, therefore we reject the null hypothesis that the residuals are normally distributed.

# %% [markdown]
# # Forecasting

# %%
close_ = close.copy()
close_.head()

# %%
# split the data into train and test
close_ = close_.reset_index()
close_ = close_.drop(columns=['Date', 'lPrice', 'sqrtPrice', 'cbrtPrice', 'diff'])
train = close_.iloc[:int(close_.shape[0]*0.8)]
test = close_.iloc[int(close_.shape[0]*0.8):]

# %%
close_.shape[0], train.shape[0], test.shape[0]

# %%
model = ARIMA(train['Price'], order=(p, d, q))
model_fit = model.fit()

# %%
steps = len(test) -1 + 100
prediction_results= model_fit.get_forecast(steps, alpha=0.05)  # 95% conf
fc = prediction_results.predicted_mean
conf = prediction_results.conf_int()
se = prediction_results.se_mean


# %%
conf

# %%
test.head()

# %%
fc_series = pd.Series(fc, index=test.index)
lower_series = conf['lower Price']
upper_series = conf['upper Price']

# %%
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# %%
train.tail()

# %%
fc_series.head()

# %%
fc_series.tail()

# %%
test.index[0]

# %%
from statsmodels.graphics.tsaplots import plot_predict

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(pd.DataFrame(train), '-g', label='training')
ax.plot(pd.DataFrame(test), '-b', label='actual')


plot_predict(model_fit, start=test.index[0], end=test.index[-1]+20, ax=ax);

# %%
model_fit.params

# %% [markdown]
# ### Evaluation

# %%
# evaluate the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test['Price'], fc_series)
print('MSE: '+str(mse))

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(test['Price'], fc_series)
print('MAE: '+str(mae))

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(test['Price'], fc_series)
print('MAPE: '+str(mape))


# %%
from pmdarima.arima import auto_arima

auto_model = auto_arima(close_['Price'], trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, scoring='mse', seasonal=True)
auto_model_fit=auto_model.fit(close_['Price'])
print(auto_model.summary())

# %%
# plot auto arima forecast
nb_periods = 30
fc, confint = auto_model.predict(n_periods=nb_periods, return_conf_int=True)
index_of_fc = np.arange(len(close_['Price']), len(close_['Price'])+nb_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(close_['Price'], label='training')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# %%
model_fit.summary()

# %%
auto_model_fit.summary()

# %% [markdown]
# Auto arima suggests an ARIMA(0,1,0) model. This result indicates a random walk model. The AIC and BIC values are also very high, which indicates that the model is not a good fit. The residuals are also not normally distributed. Therefore, we can conclude that forecasting using ARIMA is not a good idea. The stock price does not display a trend, and is very volatile.

# %% [markdown]
# 

# %%



