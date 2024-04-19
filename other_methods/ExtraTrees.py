import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import warnings
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
warnings.filterwarnings("ignore")

df = pd.read_json('/dataset/benchmark.json')
data = pd.DataFrame(df['features'].tolist())


def ET():
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = ExtraTreesRegressor(n_estimators=200, random_state=0)
    R2 = []
    MSE = []
    MAE = []
    RMSE = []
    print("ExtraTreesRegressor")
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
        np.save(f'y_true_fold_{len(MSE)}.npy', y_test)
        np.save(f'y_pred_fold_{len(MSE)}.npy', y_pred)


    print(f"Average R2: {np.mean(R2)}")
    print(f"Average MSE: {np.mean(MSE)}")
    print(f"Average MAE: {np.mean(MAE)}")
    print(f"Average RMSE: {np.mean(RMSE)}")
    
ET()