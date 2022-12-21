from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

def hyperparameter_tuning(params):
    reg = (XGBRegressor(**params))
    cv = RepeatedKFold(n_splits=10, n_repeats=3,random_state=101)
#     acc = cross_validate(reg, Xtrain_norm, Y,scoring='r2', cv=cv)
    acc = cross_val_score(reg, Xtrain, Ytrain,scoring="neg_mean_absolute_percentage_error",cv=10).mean()
    error_std = cross_val_score(reg, Xtrain, Ytrain,scoring="neg_mean_absolute_percentage_error",cv=10).std()
    
    reg.fit(Xtrain,Ytrain)
    Y_pred=reg.predict(Xtest)
    val_acc= mean_absolute_percentage_error(Ytest,Y_pred)
#     mlflow.log_metric('val_error', val_acc)
#     metrics = mlflow.sklearn.eval_and_log_metrics(MR, Xtest_norm, Ytest, prefix="val_")
    temp={'val_acc':val_acc, 'acc_mean':-acc,'error_std':error_std,'param':params}
    res.append(temp)
    pd.DataFrame(res).to_csv('./HP_tuning_sep_2n.csv', index=False)
#     print(acc)
    return {"loss": -acc, "status": STATUS_OK}