from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import xgboost as xgb
import numpy as np
#from tqdm import tqdm



df = pd.read_csv("/Users/hugolebelzic/Documents/Productivity/Project/Stock_analyzer/data/3140.HK-3.csv")

Date = pd.to_datetime(df['Date'])
y_init = df['Open']


model_arima = ARIMA(y_init, order=(1, 1, 1)) 
pred_model = model_arima.fit()
pred = pred_model.predict(1,300)

pred_model.plot_predict(1,300)


def plot_pred1(y, y_pred, model_name=None):
    y_pred = pd.DataFrame(y_pred).set_index([pd.Index([len(y)])])
    y = pd.DataFrame(y).set_index([pd.Index(np.arange(2,len(y)+2))])
    
    plt.figure()
    plt.suptitle(model_name)
    plt.subplot(1,2,1)
    plt.title("initial time series")
    plt.plot(y)
    plt.xlabel("Time")
    plt.ylabel("Value $")
    plt.subplot(1,2,2)
    plt.title("zoom in prediction")
    plt.plot(y[-10:])
    plt.scatter(253, y_pred, c="coral", label='pred')
    plt.legend()
    plt.show()


def plot_pred(y, y_pred, model_name=None):
    index_pred = [len(y)+i+3 for i in range(len(y_pred))]
    y_pred = pd.DataFrame(y_pred).set_index([pd.Index(index_pred)])
    y = pd.DataFrame(y).set_index([pd.Index(np.arange(2,len(y)+2))])
    
    plt.figure()
    plt.suptitle(model_name)
    plt.subplot(1,2,1)
    plt.title("initial time series")
    plt.plot(y)
    plt.xlabel("Time")
    plt.ylabel("Value $")
    plt.subplot(1,2,2)
    plt.title("zoom in prediction")
    plt.plot(y[-10:])
    #for i in range(len(y_pred)):
    #    print(i, index_pred[i], y_pred[i])
    plt.scatter(index_pred, y_pred, c="coral", label='pred')
    plt.plot(index_pred, y_pred)
    plt.legend()
    plt.show()
    
    
    
y = y_init.copy()
X = {"Year": list(Date.dt.year[3:]),
     "Month":list(Date.dt.month[3:]),
     "Day":  list(Date.dt.day[3:]),
     "DayofWeek": list(Date.dt.dayofweek[3:]), 
     "Xt_1": list(y[2:-1]), 
     "Xt_2": list(y[1:-2]),
     "Xt_3": list(y[:-3])} 

y = list(y[3:])
X = pd.DataFrame(X)

X_pred = {"Year": [2020],
          "Month": [5],
          "Day": [20],
          "DayofWeek": 2, 
          "Xt_1": y[-1:],
          "Xt_2": y[-2:-1],
          "Xt_3": y[-3:-2]}



X_pred = pd.DataFrame(X_pred)


model_Forest = RandomForestRegressor(n_estimators=200) 
model_Forest.fit(X, y)

y_pred = model_Forest.predict(X_pred)
plot_pred(y, y_pred, "Random Forest")




model_xgb = xgb.XGBRegressor()
model_xgb.fit(X, y)
y_pred = model_xgb.predict(X_pred)
plot_pred(y, y_pred, "xgboost")




model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred = model_linear.predict(X_pred)
plot_pred(y, y_pred, "Linear regression")


model_ridge = Ridge()
model_ridge.fit(X, y)
y_pred = model_ridge.predict(X_pred)
plot_pred(y, y_pred, "Ridge regression")


class ensemble_model:
    
    def __init__(self):
        
        self.models = [RandomForestRegressor(),
                       xgb.XGBRegressor(),
                       LinearRegression(),
                       Ridge()]
        
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X_pred):
        y_pred = []
        for model in self.models:
            y_pred.append(model.predict(X_pred)[0])
        return np.array([np.mean(y_pred)])
        
        
model_ensemble = ensemble_model()
model_ensemble.fit(X, y)
y_pred = model_ensemble.predict(X_pred)
plot_pred(y, y_pred, "Ensemble")


def cross_val_time_series(model, X, y):
    rmse = [] 
    rmspe = [] 
    
    #for i in tqdm(range(200, 249)):
    for i in range(200, 249):
        X_temp = X[:i]
        y_temp = y[:i]
        model.fit(X_temp,
                  y_temp)
        
        X_pred = X[i:(i+1)]
        y_pred_real = y[i:(i+1)]
        y_pred = model.predict(X_pred)
        #rmse.append(np.sqrt(np.sum((y_pred_real-y_pred)**2))) 
        rmse.append(np.abs(y_pred_real-y_pred))
        rmspe.append(np.abs((y_pred_real-y_pred)/y_pred_real))

    print("rmse error: ", np.mean(rmse))
    print("rmspe error: ", np.mean(rmspe))
    
"""
print("Performance cross validation: ")
print("\nRandom forest:")
cross_val_time_series(model_Forest, X, y)
print("\nxgb:")
cross_val_time_series(model_xgb, X, y)
print("\nLinear Regression:")
cross_val_time_series(model_linear, X, y)
print("\nRidge Regression:")
cross_val_time_series(model_ridge, X, y)
print("\nEnsemble:")
cross_val_time_series(model_ensemble, X, y)
"""


def recursive_prediction(model, X, y, day_ahead=5):
    y_temp = y.copy()
    if day_ahead<=0:
        raise Exception("day_ahead should be a positive integer")
    
    start_date =    "{}/{}/{}".format(int(X["Day"][-1:]), int(X["Month"][-1:]),
                                      int(X["Year"][-1:]))
     
    day_done = 0
     
    i = 1
    while day_done<day_ahead:
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=i)
        if end_date.dayofweek<6: #pour éviter le week end
            X_pred = {"Year": [end_date.year],
                      "Month": [end_date.month],
                      "Day": [end_date.day],
                      "DayofWeek": end_date.dayofweek,
                      "Xt_1": y_temp[-1:],
                      "Xt_2": y_temp[-2:-1],
                      "Xt_3": y_temp[-3:-2]}
            X_pred = pd.DataFrame(X_pred)
            y_pred = model.predict(X_pred)[0]
            y_temp.append(y_pred)
             
            day_done+=1
        
        
        i+=1
        
        
    return y_temp[-day_ahead:]


y_pred = recursive_prediction(model_Forest, X, y)
plot_pred(y, y_pred)


y_2 = y_init.copy()
X_2 = {"Year": list(Date.dt.year[4:]), 
     "Month":list(Date.dt.month[4:]),
     "Day":  list(Date.dt.day[4:]),
     "DayofWeek": list(Date.dt.dayofweek[4:]), 
     "Xt_2": list(y_2[2:-2]),
     "Xt_3": list(y_2[1:-3]),  
     "Xt_4": list(y_2[:-4])}  

y_2 = list(y_2[4:])
X_2 = pd.DataFrame(X_2)

model_Forest_2 = RandomForestRegressor(n_estimators=200) 
model_Forest_2.fit(X_2, y_2)




y_3 = y_init.copy()
X_3 = {"Year": list(Date.dt.year[5:]), 
     "Month":list(Date.dt.month[5:]),
     "Day":  list(Date.dt.day[5:]),
     "DayofWeek": list(Date.dt.dayofweek[5:]), 
     "Xt_3": list(y_3[2:-3]), 
     "Xt_4": list(y_3[1:-4]),
     "Xt_5": list(y_3[:-5])}  

y_3 = list(y_3[5:])
X_3 = pd.DataFrame(X_3)

model_Forest_3 = RandomForestRegressor(n_estimators=200)
model_Forest_3.fit(X_3, y_3)


X_pred = {"Year": [2020],
          "Month": [5],
          "Day": [26],
          "DayofWeek": 1, 
          "Xt_1": y[-1:],
          "Xt_2": y[-2:-1],
          "Xt_3": y[-3:-2]}
X_pred = pd.DataFrame(X_pred)


X_pred_2 = {"Year": [2020],
          "Month": [5],
          "Day": [27],
          "DayofWeek": 2, 
          "Xt_2": y[-1:],
          "Xt_3": y[-2:-1],
          "Xt_4": y[-3:-2]}



X_pred_2 = pd.DataFrame(X_pred_2)

X_pred_3 = {"Year": [2020],
          "Month": [5],
          "Day": [28],
          "DayofWeek": 3, 
          "Xt_3": y[-1:],
          "Xt_4": y[-2:-1],
          "Xt_5": y[-3:-2]}


X_pred_3 = pd.DataFrame(X_pred_3)


y_pred = model_Forest.predict(X_pred)
y_pred_2 = model_Forest_2.predict(X_pred_2)
y_pred_3 = model_Forest_3.predict(X_pred_3)
final_pred = [y_pred[0], y_pred_2[0], y_pred_3[0]]
plot_pred(y, final_pred, "One model per day")
print(final_pred)


