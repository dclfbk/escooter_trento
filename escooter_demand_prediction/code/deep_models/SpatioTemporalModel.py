import pandas as pd
import numpy as np
import typing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
Spatio-Temporal Model: 
    1. Graph Convolution
    2. LSTM
'''

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
        1. get adjacency matrix 

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
        2. LSTM layers
        3. Dropout
        4. Fully connected layer to decode output and compute loss
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
        
        # add first graph convolutions
        inputs = self.graph_conv(inputs) 
        inputs = tf.transpose(inputs, [0, 1, 3, 2]) # convert to (num_nodes, batch_size, input_seq_len, in_feat)

        # second graph convolution
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

