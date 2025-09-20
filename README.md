# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 20-9-2025

#### NAME:GEDIPUDI DARSHANI
#### REGISTER NUMBER:212223230062

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
```
data = pd.read_csv('INDIA.csv',parse_dates=['open'],index_col='open')
data.head()
# Example: load your CSV
data = pd.read_csv("INDIA.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Set as index
data.set_index('date', inplace=True)

# Keep only the numeric column you want to analyze, e.g. 'open'
ts = data['open']

```
```
result = adfuller(ts.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
```
```
x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]
```
```
plot_acf(ts.dropna(), lags=30)
plot_pacf(ts.dropna(), lags=30)
plt.show()

```
```
# Split into train and test
train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

# Fit AR model (let’s say with lag=5)
model = AutoReg(train, lags=5).fit()
print(model.summary())

# Forecast
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

```
```
error = mean_squared_error(test, preds)
print("MSE:", error)

# Plot
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.show()

```
### OUTPUT:

GIVEN DATA

<img width="693" height="296" alt="image" src="https://github.com/user-attachments/assets/0d3335f4-2d3c-4c5b-9791-64a1441bd7b8" />

PACF - ACF

<img width="914" height="546" alt="image" src="https://github.com/user-attachments/assets/fda4734e-4625-46e9-9099-d636c5ac9319" />


<img width="855" height="543" alt="image" src="https://github.com/user-attachments/assets/f95e4414-bd2a-4bd4-b921-b045fce8f19e" />

PREDICTION

<img width="1098" height="36" alt="image" src="https://github.com/user-attachments/assets/72101e13-84ae-429f-9b4a-100c24376c61" />

FINIAL PREDICTION

<img width="1072" height="529" alt="image" src="https://github.com/user-attachments/assets/c8ca2874-d479-4db4-89a9-379c3d1a120e" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
