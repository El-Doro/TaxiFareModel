from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from TaxiFareModel.trainer import Trainer
from TaxiFareModel.data import get_data, clean_data, get_Xy, hold_out

models = [LinearRegression(),SGDRegressor(),SVR(epsilon=0.1, C=1, kernel='linear'), RandomForestRegressor(n_estimators=100)]

for model in models :
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X,y = get_Xy(df)
    for i in range(5):
        # hold out
        X_train, X_val, y_train, y_val = hold_out(X,y)
        # train
        trainer = Trainer(X_train,y_train)
        trainer.run(model)
        # evaluate
        trainer.evaluate(X_val, y_val)
        # save model
        trainer.save_model()


    