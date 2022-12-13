import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from prepare_data import create_dataframe


def find_optimal_k(X_train, y_train, X_test, y_test, w='distance'):
    '''
    Optmize k.

    The function compute the model for different values of k (from 1 to 50) and plot the error on the test set at the variation of k.
    The k minimizing the test error should be chosen.
    '''
    MAE = []
    for k in range(1, 50):
        knn = KNeighborsRegressor(n_neighbors=k, weights=w).fit(X_train, y_train)
        predictions_k = knn.predict(X_test)
        MAE.append(mean_absolute_error(y_test, predictions_k))

    # plt.style.use('fivethirtyeight')
    plt.figure(figsize=(20,10))
    plt.plot(range(1, 50), MAE, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.xticks(range(0, 50, 1))
    plt.show()

    optimal_k = np.argmin(MAE)+1 # np.argmin(MAE)+1 -> index of the minimum error aka number of k minimizing the MAE
    print("Minimum error:", min(MAE), "at K =", optimal_k) 
    return optimal_k


def train_knn(aggr_freq, precision):
    '''
    Train knn model with the optimal k.
    '''
    df = create_dataframe(aggr_freq, precision)
    if aggr_freq.lower() == 'd':
        df = df.drop(['hour'], axis=1)
    # shuffle the data 
    df = shuffle(df, random_state=42)
    # one-hot encoding
    df = pd.get_dummies(df)
    # split test and training set
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns.drop('trips_count')], df[['trips_count']], test_size=0.25, random_state=42)
    # normalize features based on TRAIN mean and std of the respective column
    for col in X_train.columns.drop(list(df.filter(regex='cell_code'))):
        avg = X_train[col].mean()
        std = X_train[col].std()
        X_train[col] = (X_train[col] - avg) / std
        X_test[col] = (X_test[col] - avg) / std
    # normalize target based on TRAIN mean and std
    trips_cnt_avg = y_train.mean()
    trips_cnt_std = y_train.std()
    y_train = (y_train - trips_cnt_avg) / trips_cnt_std
    y_test = (y_test - trips_cnt_avg) / trips_cnt_std
    # find optimal value of k
    optimal_k = find_optimal_k(X_train, y_train, X_test, y_test)
    # knn model
    model = KNeighborsRegressor(n_neighbors=optimal_k, weights='distance').fit(X_train, y_train) 
    # make predictions 
    predictions = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, trips_cnt_avg, trips_cnt_std, predictions



def optmize_multiple_param(X_train, y_train,
                        k = list(range(1, 51)), w = ['uniform', 'distance'], fold_cv=10):
    '''
    Optimize multiple parameters (k and weights) via Grid Search Cross Validation.
    Weights can assume as value either 'uniform' (all points in the neighborhood are weighted equally)
    or 'distance' (weights closer neighbors more - closer neighbors, greater influence)

    Takes training data as input (X_train, y_train) and returns best score and best parameters for knn.

    Default for search:
    Search space for optimal k: da 1 a 50
    Possible weights: 'uniform', 'distance'

    By default, it performs a 10 fold cross validation.
    '''
    # grid for parameters
    grid = dict(n_neighbors=k, weights=w)
    # grid search 
    # with 10 fold-cross-validation (default cv=10)
    knn = KNeighborsRegressor()
    gridCV = GridSearchCV(knn, grid, cv=fold_cv)
    gridCV.fit(X_train, y_train)
    # best model
    print(gridCV.best_score_, '\n', gridCV.best_params_)
    return gridCV.best_score_, gridCV.best_params_



def evaluation_metrics(y_test, predictions):
    print('MAE:',mean_absolute_error(y_test, predictions), '\nMSE:', mean_squared_error(y_test, predictions), '\nMAPE:', mean_absolute_percentage_error(y_test, predictions), '\nR-Squared:', r2_score(y_test, predictions))
    return {'MAE':mean_absolute_error(y_test, predictions), 
            'MSE':mean_squared_error(y_test, predictions),
            'MAPE':mean_absolute_percentage_error(y_test, predictions), 
            'R-Squared':r2_score(y_test, predictions)}



def pred_vs_true(y_test, y_pred):
    plt.figure(figsize=(15,7))
    # plt.style.use('fivethirtyeight')

    plt.plot(y_test.trips_count.to_list()[:100], color='#379BDB', label='True Value', alpha=0.5)
    plt.plot(y_pred.flatten()[:100], color='#D22A0D', label='Predicted Value')

    plt.title("k-NN: True vs. Predicted Number of Trips", color='black')
    plt.xlabel('')
    plt.ylabel('Trips')
    plt.legend(loc='best', labelcolor='linecolor')
    plt.show()