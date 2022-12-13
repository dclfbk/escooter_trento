import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import geohash as pgh
import networkx as nx
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import warnings
warnings.filterwarnings("ignore")
from utils import build_df_sparse, training_curve

'''
Implementation of a spatio-temporal model, without exogenous factors, using the neighbour adjacency matrix, 
inspired by keras examples on timeseries forecasting.
'''


def prepare_data(precision=6, freq='8H'):
    df = build_df_sparse(precision=precision, freq=freq)
    df = df.reset_index()

    # BUILD ADJACENCY MATRIX: 
    # PUT CONNECTION IF THE CELLS ARE NEIGHBOURS
    # encode again to get all geohashes
    geoh = [pgh.encode(gh[0], gh[1], precision) for gh in df.gh.unique()]
    # get all neighbours cells, so that we can put a connection among them in the graph/adjacency matrix
    edges = []
    for current_gh in geoh:
        for neighbor in pgh.neighbors(current_gh):
            if neighbor in geoh: # check that the neighbor is not outside the study area of Trento
                edges.append((current_gh, neighbor))
    # build adjacency matrix based on cells neighbouring
    d = {'nodes':geoh,
        'edges':edges}
    # graph
    g = nx.DiGraph()
    g.add_nodes_from(d['nodes'])
    g.add_edges_from(d['edges'])
    # to adj. matrix
    adjacency_matrix = nx.to_pandas_adjacency(g)
    # add self-loops to adjacency matrix (to consider also the node 
    # itself in the update, as its previous state also influences the following)
    for i in range(len(adjacency_matrix)):
        adjacency_matrix.iloc[i,i] = 1.0

    # BUILD MAPPINGS NUM <-> GEOHASH: 
    # mapping centroid of cell to number and viceversa
    # useful in case you want to retrieve the cell and its location
    df['gh'] = df.gh.astype('str')
    all_gh = df.gh.unique()
    gh_to_int = {gh_id:i for i, gh_id in enumerate(all_gh)}
    int_to_gh = {i:gh_id for i, gh_id in enumerate(all_gh)}

    # replace centroids with the above defined numbers 
    df['gh'] = [gh_to_int[df.at[i, 'gh']] for i in range(len(df))]

    # df with only timeseries with trips count and cells geohash
    df = df.set_index('trip_start')[['trips_count', 'gh']]
    df = df[['gh', 'trips_count']]
    # each geohash as column, timestamps as rows, and as values trips count per time per cell
    df = pd.crosstab(df.index, df.gh, values=df.trips_count, aggfunc='sum')
    
    return df, adjacency_matrix, gh_to_int, int_to_gh


def preprocess(df, train_size=0.65, validation_size=0.15, test_size=0.2):
    '''
    Splits data into trainining, validation and test sets
    and normalize the data based on training mean and standard deviation.
    Takes as input the dataframe with the data, having as rows the moment in time in which the data were recorded 
    and as columns the locations (geohash obtained cells) in which the recording was made.
    '''
    np_data = df.to_numpy()
    if train_size + validation_size + test_size != 1.0:
        raise ValueError('Percentages do not sum up to 100%')
        
    num_timestamps = np_data.shape[0] # number of rows in the dataframe (timestamps recorded)

    # get length of training and validation based on the passed percentages
    num_train = int(num_timestamps * train_size) # default: 65% of the data
    num_val = int(num_timestamps * validation_size) # default: 15% of the data
    
    # get training data as np array
    train_array = np_data[:num_train]
    # compute mean and std on training data
    avg = train_array.mean(axis=0)
    std = train_array.std(axis=0)

    # normalize based on train mean and std
    train = (train_array - avg) / std
    validation = (np_data[num_train : (num_train + num_val)] - avg) / std
    test = (np_data[(num_train + num_val) :] - avg) / std

    return train, validation, test



def create_tf_dataset(data_array: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int = 128, shuffle=False, multi_horizon=True):
    '''
    Creates tensorflow dataset from numpy array (np_data -> in the shape (recorded_times, num_gh_cells)).
    Each element of the dataset is a tuple (input, output), where the input is a tensor of shape
    (batch_size, input_sequence_length, num_gh_cells, 1) containing "input_sequence_length" past values of the timeseries for each node/cell.
    The output is a tensor of shape (batch_size, forecast_horizon, num_gh_cells) containing the "forecast_horizon"-number of future values in time
    for each cell.

    NOTE: If multi_horizon = True, the output will be the valueS (plural) of the timeseries for the steps ahead from 1 to forecast_horizon. 
          If multi_horizon = False, the output will be the value (only one) of the timeseries ahead of forecast_horizon steps.
    '''

    # keras timeseries_dataset_from_array:
    # create batches of size (None, None, 1) -> basically a 3D tensor containing,
    # in the non-null dimension, the value of the target at a given time
    # returns: BatchDatasets of dimensions (None, None, 1)
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1), # data_array -> list with all the values -> expand so that it becomes a list of lists, each list inside containing a single value
        targets=None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()

## GRAPH
class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


## CONVOLUTIONAL GRAPH
class ConvGNN(layers.Layer):
    '''
    Simple implementation of Graph Neaural Network.

    Previous step:
        1. add self loops to adjacency matrix 

    Steps:
        2. Compute nodes representation by multiplying the features with the matrix of weights
        3. Aggregate messages coming from neighbouring nodes
        4. update nodes representation accordingly
    '''
    def __init__(self, in_feat, out_feat, graph_info: GraphInfo, aggregation_type="mean", 
                combination_type="concat", activation: typing.Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        # number of input features
        self.in_feat = in_feat
        # number of output features
        self.out_feat = out_feat
        # graph object
        self.graph_info = graph_info
        # aggregation function
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        # weights --> shape = (in_feat, out_feat)
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(shape=(in_feat, out_feat), dtype="float32"), trainable=True)
        # activation function, passed with input parameters
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor): # features -> shape = (num_nodes, batch_size, input_seq_len, in_feat)
        '''
        Compute each node's representation by multiplying the features with their weights.
        '''
        return tf.matmul(features, self.weight) # shape = (num_nodes, batch_size, input_seq_len, out_feat)

    def compute_aggregated_messages(self, features: tf.Tensor):
        '''
        Aggregate message coming from neighbouring nodes nased on the aggregation function defined above.
        '''
        neighbour_representations = tf.gather(features, self.graph_info.edges[1]) # collect
        aggregated_messages = self.aggregate(neighbour_representations) # aggregate
        return tf.matmul(aggregated_messages, self.weight) # mult. by weights

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        '''
        Update nodes representation using the propagated messages.
        '''
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h) # (num_nodes, batch_size, input_seq_len, out_feat)

    def call(self, features: tf.Tensor):
        # compute nodes representation
        nodes_representation = self.compute_nodes_representation(features)
        # aggregate messages
        aggregated_messages = self.compute_aggregated_messages(features)
        # update nodes representation
        return self.update(nodes_representation, aggregated_messages) 

## LSTM
class SpatioTemporalModel(layers.Layer):
    '''
    Processes a sequence coming from the convolutional graph 
    to produce the spatial embedding to pass to the LSTM layer, 
    which instead aims to capture the temporal dimension.

    Steps:
        1. Convolutional graph layer
        2. LSTM layer
        3. Fully connected layer to decode output and compute loss
    '''

    def __init__(self, in_feat, out_feat, lstm_units: int, input_seq_len: int, output_seq_len: int, graph_info: GraphInfo, graph_conv_params: typing.Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)

        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }

        self.graph_conv = ConvGNN(in_feat, out_feat, graph_info, **graph_conv_params)

        # stack two bidirectional lstm layers
        self.lstm = [ layers.Bidirectional(layers.LSTM(lstm_units, activation="relu", return_sequences=True)),
                    layers.Bidirectional(layers.LSTM(lstm_units//2, activation="relu", return_sequences=False)) ]

        self.drp = layers.Dropout(rate=0.15)

        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        '''
        Final Spatio-Temporal Model.
        '''
        # inputs shape = (batch_size, input_seq_len, num_nodes, in_feat)
        # convert shape to (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        graph_output = self.graph_conv(inputs)  # graph_output -> shape: (num_nodes, batch_size, input_seq_len, out_feat) 

        # returns shape in 4D -> must convert to 3D for LSTM
        shape = graph_output.get_shape() # shape = (40, None, 12, 11) => num_nodes=40, input_seq_len=12
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0], # 40
            -1, # shape[1] = None --> None is described by -1, else TypeError
            shape[2], # 12
            shape[3] # 11
        )

        # 4D to 3D -> LSTM takes only 3D tensors as input
        lstm_input = tf.reshape(graph_output, (batch_size, input_seq_len, out_feat)) 
        # shape = (None aka -1, input_seq_len, out_feat) = (None, 12, 11)
        # print(graph_output)
        
        # must loop over lstm layers
        for layer in self.lstm:
            lstm_input = layer(lstm_input) # shape: (None, lstm_units*2 bc bidirectional)

        # dropout
        drp_out = self.drp(lstm_input) # does not modify shape

        # decode with dense layer
        dense_output = self.dense(drp_out) # shape: (None, output_seq_len)

        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len)) # shape=(num_nodes, None, out)

        return tf.transpose(output, [1, 2, 0]) # shape: (batch_size, output_seq_len, num_nodes) = (None, 10, 40)







## MAIN

def main():
    # define dataset and adjacency matrix
    df, adjacency_matrix, gh_to_int, int_to_gh = prepare_data()

    # from the adj. matrix of the city network, create a simple graph
    # using the above defined class
    adjacency_matrix = adjacency_matrix.to_numpy()
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    graph = GraphInfo(edges=(node_indices.tolist(), 
                        neighbor_indices.tolist()),
                        num_nodes=adjacency_matrix.shape[0])

    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")


    # split train, test and validation, normalize and convert data to np array
    train_array, val_array, test_array = preprocess(df)


    # parameters:
    batch_size = 64
    input_sequence_length = 12
    forecast_horizon = 3
    in_feat = 1
    out_feat = 10
    lstm_units = 64
    graph_conv_params = { "aggregation_type": "mean",
                        "combination_type": "concat",
                        "activation": None }

    # create tensorflow datasets:
    train_dataset, val_dataset = ( create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size, shuffle=False) 
                                    for data_array in [train_array, val_array] )

    test_dataset = create_tf_dataset( test_array, input_sequence_length, forecast_horizon, batch_size=test_array.shape[0], shuffle=False, multi_horizon=False )


    ## SPATIOTEMPORAL MODEL (Graph+LSTM)
    STM = SpatioTemporalModel(in_feat, out_feat, lstm_units, input_sequence_length, forecast_horizon, graph, graph_conv_params)

    # feed input to the model and get output
    inputs = layers.Input(shape=(input_sequence_length, graph.num_nodes, in_feat))
    outputs = STM(inputs)

    # training
    model = keras.models.Model(inputs, outputs)
    model.compile( optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), loss=keras.losses.MeanAbsoluteError() )
        
    fitted_model = model.fit( train_dataset, validation_data=val_dataset, 
            epochs=500, 
            callbacks=[keras.callbacks.EarlyStopping(patience=10)])

    # print(model.summary())

    # plot training curve vs validation
    training_curve(fitted_model)

    # predict on test set
    x_test, y = next(test_dataset.as_numpy_iterator())
    y_pred = model.predict(x_test)

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


    # plot forecasts
    # plt.style.use('fivethirtyeight')
    plt.figure(figsize=(18, 6))
    plt.title("Forecasts on Test Set", color='black')
    plt.plot(y[:, 0, 0])
    plt.plot(y_pred[:, 0, 0])
    plt.legend(["actual", "forecast"])
    plt.show()




if __name__ == "__main__":
    main()

