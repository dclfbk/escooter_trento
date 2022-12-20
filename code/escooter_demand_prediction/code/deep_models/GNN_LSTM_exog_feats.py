import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from GNN_LSTM_data_prep import prepare_data, pois_matrix, preprocess, create_tf_dataset
from SpatioTemporalModel import GraphInfo, SpatioTemporalModel # GNN + LSTM model
import warnings
warnings.filterwarnings("ignore")

'''
Training of the Spatio-Temporal Model (Graph NN + LSTM), 
using exogenous temporal features (e.g., temperature, rain, holiday, etc.) 
and spatial features concerning POIs similiarities among cells (functional adjacency matrix).
'''


def save_results(y, y_pred, std, mean, threshold = 1.5): # std from correct prediction ca 1.5
    # convert to original data
    pred_original = y_pred[:, 0, :]*std+mean
    y_original = y[:, 0, :]*std+mean 
    # creata dataframe with the results
    results = pd.DataFrame({'pred':[pred_original[i][0] for i in range(len(pred_original))], 
                            'rounded_pred': [round(pred_original[i][0], 0) for i in range(len(pred_original))], 
                            'true':[y_original[i][0] for i in range(len(y_original))]})
    results['abs_diff'] = abs(results.true-results.pred)
    # acceptable difference (boolean)
    results['accept_diff'] = results.abs_diff <= threshold
    results.to_csv('C:/Users/luisa/Desktop/thesis_project/images/GNN-LSTM/gnn_lstm_results_1h.csv')


## MAIN

def main():
    # get trips count data and temporal features
    df, neighbours_matrix, gh_to_int, int_to_gh, temporal_feats = prepare_data(precision=precision, freq=freq)
    # process trips count data and create train, validation and test set
    train_array, val_array, test_array, std, avg = preprocess(df=df, train_size=0.6, validation_size=0.2, test_size=0.2)
    # split temporal features for train, validation and test (already normalized)
    temporal_feats_train = temporal_feats[:len(train_array)]
    temporal_feats_val = temporal_feats[len(train_array):len(train_array)+len(val_array)].reset_index(drop=True)
    temporal_feats_test = temporal_feats[len(train_array)+len(val_array):].reset_index(drop=True)

    # produce tensorflow datasets
    train_dataset = create_tf_dataset(train_array, temporal_feats_train, input_sequence_length, forecast_horizon, batch_size, shuffle=True)
    val_dataset = create_tf_dataset(val_array, temporal_feats_val, input_sequence_length, forecast_horizon, batch_size, shuffle=True)
    test_dataset = create_tf_dataset(test_array, temporal_feats_test, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False, multi_horizon=multi_horizon)


    # select adjacency matrix
    if chosen_matrix == 'neighbour':
        # neighbouring cells matrix
        adjacency_matrix = neighbours_matrix.to_numpy() 

        node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
        graph = GraphInfo(
            edges=(node_indices.tolist(), neighbor_indices.tolist()),
            num_nodes=adjacency_matrix.shape[0])

    elif chosen_matrix == 'pois':
        # POIs cosine similary matrix
        matrix_pois = pois_matrix(gh_to_int)
        adjacency_matrix = matrix_pois.to_numpy()

        node_indices, neighbor_indices = np.where(adjacency_matrix != 0)
        graph = GraphInfo(
            edges=(node_indices.tolist(), neighbor_indices.tolist()),
            num_nodes=adjacency_matrix.shape[0])


    # run the model
    STM = SpatioTemporalModel(
        in_feat, out_feat, lstm_units,
        input_sequence_length,
        forecast_horizon,
        graph, # graph from adjacency matrix
        graph_conv_params)
        
    inputs = layers.Input(shape=(input_sequence_length, graph.num_nodes, in_feat))
    outputs = STM(inputs)

    model = keras.models.Model(inputs, outputs)
    model.compile( optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate), loss=keras.losses.MeanSquaredError(), metrics=['mse'] )
            
    fitted_model = model.fit(
        train_dataset, # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets will be obtained from x) -> just put sample and label in train df
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit --> first sample then target
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    # plot training curve
    plt.plot(fitted_model.history['loss'], label='train')
    plt.plot(fitted_model.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # forecasts
    x_test, y = next(test_dataset.as_numpy_iterator()) # y has data shifted of one timestep ahead -> x_test to predict the next timestep, on y
    y_pred = model.predict(x_test)


    # PLOT PREDICTIONS:

    # y[time, forecast_horizon, cell] --> y[:, 0, 3] --> select all times, for grid cell num. 3 / node 3
    # here plot predictions for cell/node 0

    # plot true vs predicted values
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(18, 6))
    plt.title("Forecasts on Test Set", color='black')
    plt.plot(y[:, 0, 0], marker='.', label="true", alpha=0.7)
    plt.plot(y_pred[:, 0, 0], 'r', label="prediction")
    plt.legend()
    plt.show()

    # zoom-in first 200 timestemps
    test_timestamps = df[len(train_array)+len(val_array):].index[:-14] # get timestamps for x-axis

    plt.figure(figsize=(18, 6))
    plt.title("Forecasts on Test Set", color='black')
    plt.plot(test_timestamps[:200], y[:, 0, 0][:200], marker='*', alpha=0.7, label="true")
    plt.plot(test_timestamps[:200], y_pred[:, 0, 0][:200], 'r', linestyle="-", marker=".", label="prediction")
    plt.legend()
    plt.show()

    # plot forecasts for all cells in the grid
    # for i in range(len(y[0,0,:])):
    #     plt.figure(figsize=(18, 6))
    #     plt.title("Forecasts on Test Set", color='black')
    #     plt.plot(test_timestamps[:200], y[:, 0, i][:200], marker='*', alpha=0.7, label="true")
    #     plt.plot(test_timestamps[:200], y_pred[:, 0, i][:200], 'r', linestyle="-", marker=".", label="prediction")
    #     plt.legend()
    #     plt.show()


    # metrics to evaluate performances on test set
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    mae = tf.keras.metrics.MeanAbsoluteError()
    mse = tf.keras.metrics.MeanSquaredError()

    mape.update_state(y, y_pred)
    mae.update_state(y, y_pred)
    mse.update_state(y, y_pred)

    print('MAE:', mae.result().numpy(), 
        '\nMAPE:', mape.result().numpy(), 
        '\nMSE:', mse.result().numpy(), 
        '\nStandard Deviation:', np.std(abs(( y - y_pred ))) )
    
    # save results
    # save_results(y, y_pred, std, avg)



if __name__ == "__main__":

    # spatial dimension
    precision = 5
    # temporal dimension
    freq = 'H'

    # model parameters
    chosen_matrix = 'pois'
    in_feat = 12 # num trips + 11 exogenous variables
    batch_size = 64
    epochs = 500
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = True
    out_feat = 10
    lstm_units = 64
    learning_rate = 0.0001 # optimizer: RMSprop
    graph_conv_params = { "aggregation_type": "max", # mean 
                        "combination_type": "concat",
                        "activation": "relu" } # None 
    
    # run model
    main() 