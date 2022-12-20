import pandas as pd
import numpy as np
from utils import create_sequences, training_curve
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import holidays
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams

rcParams['figure.figsize'] = 22, 10



'''
LSTM model is particularly accurate for hourly prediction, while more data are needed to test effectively other temporal aggregations.
For larger intervals relying on ARIMA would lead to good results, when only the current training data are available.
'''

# desired temporal interval for prediction
freq_to_predict = 'H'

def temporal_aggregation(df, aggr_freq):
  # resample data according to temporal freq chosen
  df = df.resample(aggr_freq, on='date_hour').agg({'trips_count':'sum',
              'totalSnow_cm':'mean', 'windspeedKmph':'mean', 'tempC':'mean', 
              'visibility':'median', 'precipMM':'mean'})
  # add holiday (boolean)
  df['holiday'] = [df.index[i] in holidays.Italy() for i in range(len(df))] 
  df['holiday'] = df.holiday.astype('int').to_numpy() 
  # add weekday as number (0-6)
  df['day'] = df.index.weekday 
  return df




# BUILD DATAFRAME 

df = pd.read_parquet('C:/Users/luisa/Desktop/thesis_project/data/tripsv3_augmented.parquet') 
df['date_hour'] = [pd.Timestamp(df.at[i, 'trip_start'].year, df.at[i, 'trip_start'].month, df.at[i, 'trip_start'].day, df.at[i, 'trip_start'].hour) for i in range(len(df))]

# counts
df = df.groupby(['date_hour']).aggregate({'unique_id':['count']}).reset_index()
df.columns = df.columns.droplevel(1)
df.rename(columns = {'unique_id':'trips_count'}, inplace = True)

# IMPORTANT STEP: 
# ADD MISSING TIMESTAMPS

# since freq='H' is the min freq used in this project, use that, then can aggregate times if needed
freq = 'H'
# time interavals
all_times = pd.DataFrame({'date_hour': pd.date_range(df.date_hour.min(), df.date_hour.max(), freq=freq)})
# build dataframe with rows for all times and all cells, even when trips_count = 0 
df_all_times = pd.merge(all_times, df, left_on='date_hour', right_on='date_hour', how='left')

# fill trips count with 0 when there were no trips
df_all_times.trips_count = df_all_times.trips_count.fillna(0)

# add weather info
weather = pd.read_csv('C:/Users/luisa/Desktop/thesis_project/data/weather_info.csv')
weather = weather[['date_time', 'totalSnow_cm', 'windspeedKmph', 'tempC', 'visibility', 'precipMM']]

weather['date_time'] = pd.to_datetime(weather.date_time, format="%Y-%m-%d %H:%M:%S")

df = df_all_times.merge(weather, left_on=['date_hour'], right_on=['date_time'])
df = df.drop(['date_time'], axis=1)

# sort by timestamp
df = df.sort_values(by=['date_hour'])


# TRAINING DATASET AGGREGATED AT THE DESIRED TIME-INTERVAL
d = temporal_aggregation(df = df, aggr_freq = freq_to_predict)


# statistics 
print('Trips Statistics:\nmedian  ', d.trips_count.median())
print(d.trips_count.describe())




## SPLIT TRAINING AND TEST SET
# 80% train - 20% test
# validation -> split from the 80% data used for training (during model fit)
train_size = int(len(d) * 0.8)
test_size = len(d) - train_size
train, test = d.iloc[0:train_size], d.iloc[train_size:len(d)]



## SCALE FEATURES
# take all columns, remove target variable
feat_columns = list(d.columns)
feat_columns.remove('trips_count')

# fit robust scaler on features --> scales features using statistics that are robust to outliers: removes the median and scales the data according to the
# Interquartile Range (IQR). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
feat_transformer = RobustScaler().fit(train[feat_columns].to_numpy()) # scale numeric features on feat_columns
# transform training features
train.loc[:, feat_columns] = feat_transformer.transform(train[feat_columns].to_numpy())
# transform test features
test.loc[:, feat_columns] = feat_transformer.transform(test[feat_columns].to_numpy())


## SCALE TARGET VARIABLE
# fit robust scaler on y-data
trips_transformer = RobustScaler().fit(train[['trips_count']]) # scale the target (trips_count)
# transform training target
train['trips_count'] = trips_transformer.transform(train[['trips_count']])
# transform test target
test['trips_count'] = trips_transformer.transform(test[['trips_count']])



## CREATE SEQUENCES FOR LSTM
# features in train and test test will be shaped as (train_samples, time_steps, number_of_features)
# and the target as (train_samples, ).

# reshape the data using the above function

X_train, y_train = create_sequences(X=train, y=train.trips_count, time_steps=10)
X_test, y_test = create_sequences(X=test, y=test.trips_count, time_steps=10)


## CREATE THE MODEL
# Bi-Directional LSTM + DropOut layer of 20% of neurons, randombly selected, to avoid overfitting
# LSTM layer docs: https://keras.io/api/layers/recurrent_layers/lstm/
model = keras.Sequential()

# model.add(keras.layers.Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2]))) # input_shape = (timesteps, input_dimensions)

model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))

model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))

# model.add(keras.layers.Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# Use EarlyStopping to stop training when the monitored metric has stopped improving
# and ReduceLROnPlateau reduce learning rate when the monitored metric has stopped improving (potentially not needed with ADAM optimizer)

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True) 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 3, min_lr = 1e-6, cooldown = 5, verbose = 0) 


## FIT THE MODEL
fitted_model = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=256, # 32 
    validation_split=0.2,
    shuffle=False, # do NOT shuffle while training
    callbacks=[early_stopping] # reduce_lr
)



## PLOT TRAINING CURVE VS VALIDATION
training_curve(fitted_model)


## PREDICTION ON TEST SET
y_pred = model.predict(X_test)

# inverse of RobustScaler to get the original values of the target variable
y_train_original = trips_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_original = trips_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_original = trips_transformer.inverse_transform(y_pred)


## PLOT HISTORICAL VALUE AND FORECASTING FOR TEST VALUES
plt.plot(np.arange(0, len(y_train)), y_train_original.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_original.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_original.flatten(), 'r', label="prediction")
plt.ylabel('Trip Count')
plt.xlabel('Time Step')
plt.legend()
plt.show()

## PLOT PREDICTED VALUES VS THE TRUE ONES
plt.plot(y_test_original.flatten()[:150], marker='*', label="true")
plt.plot(y_pred_original.flatten()[:150], 'r', label="prediction")
plt.ylabel('Trip Count')
plt.xlabel('Time Step')
plt.legend()
plt.show()

## EVALUATION ON TEST SET
eval = model.evaluate(X_test, y_test)
MAE = eval[0]

mape = tf.keras.metrics.MeanAbsolutePercentageError()
mae = tf.keras.metrics.MeanAbsoluteError()
mse = tf.keras.metrics.MeanSquaredError()

# evaluate on normalized data
# mape.update_state(y_test, y_pred)
# mae.update_state(y_test, y_pred)
# mse.update_state(y_test, y_pred)

# evaluate on original data
mape.update_state(y_test_original.flatten(), y_pred_original.flatten())
mae.update_state(y_test_original.flatten(), y_pred_original.flatten())
mse.update_state(y_test_original.flatten(), y_pred_original.flatten())

print('MAE:', mae.result().numpy(), '\nMAPE:', mape.result().numpy(), '\nMSE:', mse.result().numpy(), 
'\nStandard Deviation:', np.std(abs(( y_test.flatten() - y_pred.flatten() ))))


## SAVE RESULTS
results = pd.DataFrame({'pred':y_pred_original.flatten(), 'rounded_pred': [round(p, 0) for p in y_pred_original.flatten()], 'true':y_test_original.flatten(), 
'abs_diff': [abs(y_test_original.flatten()[i]-y_pred_original.flatten()[i]) for i in range(len(y_pred_original.flatten()))]})#.sort_values(by='true')
# save
# results.to_csv('C:/Users/luisa/Desktop/thesis_project/imgs/LSTM/LSTM_results.csv')