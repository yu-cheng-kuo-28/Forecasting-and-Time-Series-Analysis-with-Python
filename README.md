# Forecasting-and-Time-Series-Analysis-with-Python
**Holt-Winters, SARIMA, auto ARIMA, ACF, PACF & differencing**

- **Keywords**: Holt-Winters, Exponential smoothing, SARIMA, Auto ARIMA, Pmdarima, ACF, PACF, Differencing, Seasonal decomposition
- Complete Python code with results: on Colab https://bit.ly/371pUN5 / on Kaggle https://bit.ly/2OHZPwL

- We assume the reader is already familiar with time series theories including SARIMA & Holt-Winters; if not, check reference [3][5][7][9][13] for more details. Hence, we put emphasis primarily on how to conduct forecasting & time series analysis with Python. Let’s get started!
- In traditional time series area (cf. cutting edge forecasting approaches like RNN, LSTM, GRU), Python is still like a teenager and R is like an adult already. Fortunately, there are some emerging Python modules like pmdarima, starting from 2017, developed by Taylor G Smith et al., help convert R’s time series code into Python code.

## Outline
(1) A New Module: pmdarima
(2) A Toy Dataset: Australian Total Wine Sales
(3) Seasonal Decomposition using Moving Averages
(4) Stationarity: First and Second Order Differencing
(5) AR and MA: ACF & PACF
(6) SARIMA using Auto ARIMA function from pmdarima
(7) Forecasting with SARIMA & Holt-Winters
(8) Reference


## (1) A New Module: pmdarima
pmdarima brings R’s beloved auto.arima to Python, making an even stronger case for why you don’t need R for data science. pmdarima is 100% Python + Cython and does not leverage any R code, but is implemented in a powerful, yet easy-to-use set of functions & classes that will be familiar to scikit-learn users. [11]

## (2) A Toy Dataset: Australian Total Wine Sales
Australian total wine sales by wine makers in bottles <= 1 liter. This time-series records monthly wine sales by Australian wine makers between Jan 1980 — Aug 1994. This dataset is found in the R forecast package. [12]

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
y = pm.datasets.load_wineind()
datetime_1 = pd.period_range('1980-01', periods=176, freq='M')
dataset_wine = pd.DataFrame(data={'sales': y}, index=datetime_1)
```

## (3) Seasonal Decomposition using Moving Averages [1][6]
A time series is said to be comprised of the following three major components:
1. Seasonality
2. Trend
3. Residual
In the following snippet, we utilize statsmodels to decompose the Australian total wine sales time series into its three constituents and then plot them.

Now since we get a bird’s-eye view of time series analysis, we then break down the time series analysis of Australian total wine sales.


## (4) Stationarity: First and Second Order Differencing
A SARIMA model looks like ARIMA(1,1,2)(0,0,0)[12], which can be expressed in a general form ARIMA(p,d,q)(P,D,Q)[s].
1. Stationarity term: d
2. AR term: p
3. MA term: q
First, we need to decide the value of d in the model above by checking whether the series is stationary or non-stationary. To do so, the first step comes in our mind is first and second order differencing. Nonetheless, it’s hard and subjective to tell that at which chart does the series convert from non-stationary to stationary since many people actually determine d = 0 or 1 or 2 simply by merely inspecting the following figure.

Hence, there’re easy yet precise ways to determine the value of d. Here are the snippet and what we get.
```
# Stationarity
from pmdarima.arima import ndiffs as ndiffs
# test =  (‘kpss’, ‘adf’, ‘pp’)
print('KPSS: d =', ndiffs(dataset_wine_array, alpha=0.05, test='kpss', max_d=2)) # d = 1. Indicating non-stationary sequence
print('ADF: d =', ndiffs(dataset_wine_array, alpha=0.05, test='adf', max_d=2)) # d = 0. Indicating stationary sequence
print('PP: d =', ndiffs(dataset_wine_array, alpha=0.05, test='pp', max_d=2)) # d = 0. Indicating stationary sequence
```
KPSS: d = 1
ADF: d = 0
PP: d = 0
Then we choose KPSS's result, d=1, as KPSS is a comparably more advanced technique. However, you will know later that analysis here doesn’t really matter once we leverage the auto.arima function in the new Python module pmdarima.


## (5) AR and MA: ACF & PACF

Having d = 1 at hand, we then move on to finding p (AR) & q (MA).
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
fig, ax = plt.subplots(2,1,figsize=(22,6), sharex=False)
sm.graphics.tsa.plot_acf(dataset_wine_array, lags=50, ax=ax[0])
sm.graphics.tsa.plot_pacf(dataset_wine_array, lags=50, ax=ax[1])
plt.show()

We can determine the value of p (AR) & q (MA) by the figure above as you can see from some time series articles, but again, it’s a bit subjective. Thus, the next paragraph comes the solution — auto.arima function in the new Python module pmdarima derived from R.
(6) SARIMA using Auto ARIMA function from pmdarima [11][13]

```
# Fit the model
model = pm.auto_arima(dataset_wine_array, seasonal=True, m=12, 
                      information_criterion='aic', test='kpss',
                      suppress_warnings=True, trace=True)
# The best model
model.set_params()
model.summary()
```
Finally, by pm.auto_arima() we get the best model ARIMA(0,1,2)(0,1,1)[12] quite effortless with p(AR)=0, d=1, q(MA)=2.


## (7) Forecasting with SARIMA & Holt-Winters
### 7–1 Forecasting with SARIMA [3][11][13]
```
## SARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pmdarima as pm

# Fit the model
model = pm.auto_arima(train, seasonal=True, m=12, 
                      information_criterion='aic', test='kpss',  
                      maxiter=150,
                      suppress_warnings=True, trace=True, verbose=1)
pred_SARIMA_conf_int = model_SARIMA.predict(test.shape[0], return_conf_int=True)[1]
# Make forecasts
pred_SARIMA = model_SARIMA.predict(test.shape[0])  # predict N steps into the future
# Confidence interval
pred_SARIMA_conf_int = model_SARIMA.predict(test.shape[0], return_conf_int=True)[1]
lower_limits = [k[0] for k in pred_SARIMA_conf_int]
upper_limits = [k[1] for k in pred_SARIMA_conf_int]
```


## 7–2 Forecasting with Holt-Winters [3][7]
```
## Holt-Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model_HW = ExponentialSmoothing(train,  trend='add', seasonal='add', seasonal_periods=12, damped_trend=True).fit(optimized=True, use_boxcox=False, remove_bias=False)
pred_HW = model_HW.predict(start=train.shape[0], end=dataset_wine_array.shape[0]-1)
```

To date, ExponentialSmoothing doesn’t have parameter to generate 95% CI as pm.auto_arima does. [2]


## 7–3 Model Evaluation — RMSE, MAE, MAPE

Holt-Winters outperforms SARIMA in terms of RMSE.


## (8) Reference
[1] Brownlee, J. (2020). How to Decompose Time Series Data into Trend and Seasonality. Retrieved from https://bit.ly/2N9yRgi
[2] Ryan Boch (2020). Prediction intervals exponential smoothing statsmodels. Retrieved from https://bit.ly/3rEMqmT
[3] tutorialspoint (2019). Time Series. Retrieved from https://bit.ly/39WaDiw
[4] QuantStats (2019). displaying statsmodels plot_acf and plot_pacf side by side in a jupyter notebook. Retrieved from https://bit.ly/3cTEfz5
[5] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice (2nd ed.). OTexts: Melbourne, Australia. Retrieved from https://otexts.com/fpp2/
[6] Sarkar, D., Bali, R., & Sharma, T. (2018). Practical Machine Learning with Python: A problem-solver’s guide to building real-world intelligent systems. Karnataka, India: Apress.
[7] ayhan (2018). Holt-Winters time series forecasting with statsmodels. Retrieved from https://bit.ly/3cTKP8K
[8] cel (2015). Changing fig size with statsmodel. Retrieved from https://bit.ly/3pWB2SW
[9] McKinney, W., Perktold, J., & Seabold, S. (2011). Time Series Analysis in Python with statsmodels. Retrieved from https://bit.ly/3utCUW1
[10] pypi.org (Unidentified). pmdarima. Retrieved from https://bit.ly/3aQrdjk
[11] Smith, T.G. et al. (Unidentified). pmdarima: ARIMA estimators for Python. Retrieved from https://bit.ly/2N5RgKO
[12] Smith, T.G. et al. (Unidentified). pmdarima.datasets.load_wineind. Retrieved from https://bit.ly/2N5XZVc
[13] Smith, T.G. et al. (Unidentified). Tips to using auto_arima. Retrieved from https://bit.ly/3cZXreE
[14] statsmodels (Unidentified). statsmodels v0.12.1. Retrieved from https://bit.ly/2NaDzKD
