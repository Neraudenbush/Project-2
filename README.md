# Project-2
Project 2


```Python
data = pd.read_csv('cars.csv')
```
I chose to use the cars data set
```Python
X = np.array(data['WGT']).reshape(-1,1)
y = np.array(data['MPG']).reshape(-1,1)

plt.scatter(X,y)

#Define all the functions we will neeed to use

def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
  
 def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
  
 def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 
 


def lowess_kern(x, y, kern, tau):



    n = len(x)
    yest = np.zeros(n)

   
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest
    
    
dat = np.concatenate([X_train,y_train.reshape(-1,1)], axis=1)

dat = dat[np.argsort(dat[:, 0])]

dat_test = np.concatenate([X_test,y_test.reshape(-1,1)], axis=1)
dat_test = dat_test[np.argsort(dat_test[:, 0])]

#The Qaurtic kernal provides the lowest MSE and a Tau of .6
Yhat_lowess = lowess_kern(dat[:,0],dat[:,1],Quartic,.6)
    
    
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=410)
mse_lowess = []

for idxtrain, idxtest in kf.split(dat):
  X_train = datl[idxtrain,0]
  y_train = datl[idxtrain,1]
  X_test  = datl[idxtest,0]
  y_test = datl[idxtest,1]
  lm.fit(X_train.reshape(-1,1),y_train)
  yhat_lm = lm.predict(X_test.reshape(-1,1))
  mse_lowess.append(mse(y_test, yhat_lm))
print("Validated MSE Weighted Linear Model = {:,.5f}".format(np.mean(mse_lowess)))
```
Cross Validated MSE of 15.61564 for Locally Weighted Linear Regression

```Python
mse_rf = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0]
  y_train = dat[idxtrain,1]
  X_test  = dat[idxtest,0]
  y_test = dat[idxtest,1]
  rf.fit(X_train.reshape(-1,1),y_train)
  yhat_rf = rf.predict(X_test.reshape(-1,1))
  mse_rf.append(mse(y_test, yhat_rf))
print("Validated MSE RandomForest = {:,.5f}".format(np.mean(mse_rf)))
```
Cross Validated MSE of 16.84321 for Random Forest Regression

The chart below represents the graphs of the linear regression(Red) and the random forest(Green) over the cars data

<img src="Chart Data.png" width="600" height="400" alt="hi" class="inline"/>
