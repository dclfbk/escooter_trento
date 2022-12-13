import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# import function to aggregate data at different spatio-temporal levels
from prepare_data import create_dataframe



# EVALUATE MODEL PERFORMANCE

def evaluate(model, test_features, test_labels, trips_cnt_avg, trips_cnt_std):
    '''
    Evaluate RANDOM FOREST model performance.
    '''
    predictions = model.predict(test_features)

    # de-norm
    predictions = (predictions*trips_cnt_std)+trips_cnt_avg
    test_labels = (test_labels*trips_cnt_std)+trips_cnt_avg
    
    # avg_error = np.mean(abs(predictions - test_labels)) -> mae
    mse = mean_squared_error(test_labels, predictions)
    rmse = mean_squared_error(test_labels, predictions, squared=False)
    mae = mean_absolute_error(test_labels, predictions)
    mape = mean_absolute_percentage_error(test_labels, predictions)
    
    return { 'mae': mae, 'mse': mse, 
            'rmse': rmse, 'mape': mape}


# TRAIN MODEL FOR DIFFERENT SPATIO-TEMPORAL AGGREGATION

def train_rf_diff_aggr(aggr_freq, precision, grid = None, aggr_function = 'median', method = 'random_search'):
    '''
    Train a Random Forest model with data having the selected spatio-temporal aggregation.
    Parameters are tuned using the chosen method.

    Takes:
     - aggr_freq: time interval to aggreagte timestamps
     - precision: spatial precision to divide the territory in blocks, defines the dimensions of the blocks
     - grid: (Optional) grid to be used in random search or grid search; if not passed, a default grid is used
     - aggr_function: (Optional) function to be used to aggregate point of interest per region; default "median"
     - method: (Optional) method to use to tune parameters, either "random_search" or "grid_search"; default "random_search"
     
    Returns:
     - random forest model with tuned parameters
     - evaluation metrics of the model
     - test and train data 
     - mean and standard deviation to recompute original values of trips_counts
    '''
    if grid == None:
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)],
                    'max_features': ['log2', 'sqrt', 'auto', 2, 6, 10, 15, None],
                    'max_depth': max_depth,
                    'min_samples_split': [30, 50, 100, 200, 500],
                    'min_samples_leaf': [20, 30, 50, 100, 200, 500],
                    'bootstrap': [True, False]}
    # create dataframe
    df = create_dataframe(aggr_freq, precision, aggr_function)
    # hour not needed if frequency is daily, plus it generates NaN when normalized bc it assumes always same value
    if aggr_freq.lower() == 'd':
        df = df.drop(['hour'], axis=1)
    # get dummies (one-hot encoding for categorical feature - cell_code)
    df = pd.get_dummies(df)
    # shuffle the data
    df = shuffle(df, random_state=42)
    # split train and test, with 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(df.drop('trips_count', axis=1), df.trips_count, test_size=0.25, random_state=42)
    # normalize data (except cell_code) - using each column mean and sd in the TRAIN set
    # compute mean and std on training data
    # normalize features based on TRAIN mean and std of the respective column
    for col in X_train.columns.drop(list(df.filter(regex='cell_code'))):
        avg = X_train[col].mean()
        std = X_train[col].std()
        X_train[col] = (X_train[col] - avg) / std
        X_test[col] = (X_test[col] - avg) / std
    # normalize target based on TRAIN mean and std (return them to denormalize results)
    trips_cnt_avg = y_train.mean()
    trips_cnt_std = y_train.std()
    y_train = (y_train - trips_cnt_avg) / trips_cnt_std
    y_test = (y_test - trips_cnt_avg) / trips_cnt_std
    # initialize random forest
    rf = RandomForestRegressor()
    # PARAMETERS TUNING USING THE SELECTED METHOD
    if method == 'random_search':
        # RANDOM SEARCH 
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter = 100, cv = 5, verbose = 2, random_state = 42, n_jobs = -1)
        rf_random.fit(X_train, y_train)
        rs_model = rf_random.best_estimator_
        # evaluation metrics
        metrics = evaluate(rs_model, X_test, y_test, trips_cnt_avg, trips_cnt_std)
        final_model = rf_random.best_estimator_
    elif method == 'grid_search':
        grid_search = GridSearchCV(estimator = rf, param_grid = grid, cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_
        # evaluate metrics
        metrics = evaluate(best_grid, X_test, y_test, trips_cnt_avg, trips_cnt_std)
        final_model = grid_search.best_estimator_
    else:
        raise ValueError('Invalid method for paramter tuning.')
    return final_model, metrics, X_train, X_test, y_train, y_test, trips_cnt_avg, trips_cnt_std


# PLOTS

def RF_pred_vs_true(y_test, y_test_pred, trips_cnt_std, trips_cnt_avg): 
    '''
    Plot prediction vs ground truth
    '''
    plt.style.use('default')
    # denormalize
    y_test_orig = y_test * trips_cnt_std + trips_cnt_avg
    y_test_pred_orig = y_test_pred * trips_cnt_std + trips_cnt_avg
    # plot predicted vs true values
    plt.figure(figsize=(20,8))
    plt.plot(y_test_orig[0:100].values, label="actual test value", color='royalblue')
    plt.plot(y_test_pred_orig[0:100], label="prediction", color='indianred')
    plt.legend()
    plt.title("Prediction vs. Real Value", color='black')
    plt.xlabel("")
    plt.ylabel("trips_count")
    plt.show()


def RF_features_importance(rf_model, cols): 
    '''
    Plot features by importance and cumulative importance.
    Takes as input a random forest model and the list of all the features (X_train.columns).
    '''
    plt.style.use('fivethirtyeight')

    # check which features are the most important in the random forest model
    feats = {}
    for feature, importance in zip(cols, rf_model.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})

    # Plot most important features
    plt.figure(figsize=(20,8))
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'][:20], data=importances, color='skyblue')
    plt.xlabel('Gini-Importance', fontsize=20)
    plt.ylabel('Features', fontsize=20)
    plt.title('Top-20 Most Important Features', fontsize=25, weight = 'bold')
    plt.show()

    # Plot Cumulative Importance
    cumulative_importances = np.cumsum(importances['Gini-Importance'])
    plt.figure(figsize=(20,8))
    # line graph
    plt.plot(cols, cumulative_importances, 'g-')
    # draw line at 95% of importance:
    plt.hlines(y = 0.95, xmin=0, xmax=len(importances['Features']), color = 'r', linestyles = 'dashed')
    plt.xticks(cols, importances['Features'], rotation = 'vertical')
    plt.xlabel('Variable')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Importances')
    plt.show()