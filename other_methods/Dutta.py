from sklearn.linear_model import BayesianRidge
import pandas as pd
import warnings
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")

df = pd.read_json('/dataset/benchmark.json')
data = pd.DataFrame(df['features'].tolist())


def BR():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = BayesianRidge(n_iter=2000, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, copy_X=True, verbose=False)
    R2 = []
    MSE = []
    MAE = []
    RMSE = []
    print("BayesianRidge")
    for train_index, test_index in kf.split(data):

        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = df["LogS"].iloc[train_index], df["LogS"].iloc[test_index]
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("R2_score:"+str(r2_score(y_test,y_pred)))
        print("MSE:"+str(mean_squared_error(y_test,y_pred)))
        print("MAE:"+str(mean_absolute_error(y_test,y_pred)))
        print("RMSE:"+str(np.sqrt(mean_squared_error(y_test,y_pred))))

        R2.append(r2_score(y_test,y_pred))
        MSE.append(mean_squared_error(y_test,y_pred))
        MAE.append(mean_absolute_error(y_test,y_pred))
        RMSE.append(np.sqrt(mean_squared_error(y_test,y_pred)))

    print(f"Average R2: {np.mean(R2)}")
    print(f"Average MSE: {np.mean(MSE)}")
    print(f"Average MAE: {np.mean(MAE)}")
    print(f"Average RMSE: {np.mean(RMSE)}")

def L():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    R2 = []
    MSE = []
    MAE = []
    RMSE = []
    print("Linear model")
    for train_index, test_index in kf.split(data):

        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = df["LogS"].iloc[train_index], df["LogS"].iloc[test_index]
        

        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        print("R2_score:"+str(r2_score(y_test,y_pred)))
        print("MSE:"+str(mean_squared_error(y_test,y_pred)))
        print("MAE:"+str(mean_absolute_error(y_test,y_pred)))
        print("RMSE:"+str(np.sqrt(mean_squared_error(y_test,y_pred))))

        R2.append(r2_score(y_test,y_pred))
        MSE.append(mean_squared_error(y_test,y_pred))
        MAE.append(mean_absolute_error(y_test,y_pred))
        RMSE.append(np.sqrt(mean_squared_error(y_test,y_pred)))


    print(f"Average R2: {np.mean(R2)}")
    print(f"Average MSE: {np.mean(MSE)}")
    print(f"Average MAE: {np.mean(MAE)}")
    print(f"Average RMSE: {np.mean(RMSE)}")

def R():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Ridge(alpha=0.1)
    R2 = []
    MSE = []
    MAE = []
    RMSE = []
    print("Ridge model")
    for train_index, test_index in kf.split(data):

        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = df["LogS"].iloc[train_index], df["LogS"].iloc[test_index]
        

        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        print("R2_score:"+str(r2_score(y_test,y_pred)))
        print("MSE:"+str(mean_squared_error(y_test,y_pred)))
        print("MAE:"+str(mean_absolute_error(y_test,y_pred)))
        print("RMSE:"+str(np.sqrt(mean_squared_error(y_test,y_pred))))
        

        R2.append(r2_score(y_test,y_pred))
        MSE.append(mean_squared_error(y_test,y_pred))
        MAE.append(mean_absolute_error(y_test,y_pred))
        RMSE.append(np.sqrt(mean_squared_error(y_test,y_pred)))


    print(f"Average R2: {np.mean(R2)}")
    print(f"Average MSE: {np.mean(MSE)}")
    print(f"Average MAE: {np.mean(MAE)}")
    print(f"Average RMSE: {np.mean(RMSE)}")


BR()
L()
R()