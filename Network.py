import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.optimizers import Adam
from Node import ENCODE_BOARD, ENCODE_LEGAL, ENCODE_BOTH
def create_snort_model(encode_type):
   
    # Define input layer
    input_size = 202
    if encode_type == ENCODE_BOARD:
        input_size = 302
    elif encode_type == ENCODE_BOTH:
        input_size = 502
        
    input_layer = Input(shape=(input_size,))
    
    # Hidden layers
    hidden_layer1 = Dense(128, activation='relu')(input_layer)
    hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
    
    # Output layer for the first 200 outputs
    output_layer_mse = Dense(200)(hidden_layer2)
    
    # Output layer for the last output with softmax activation for cross-entropy
    output_layer_ce = Dense(1, activation='softmax')(hidden_layer2)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=[output_layer_mse, output_layer_ce])
    
    # Compile the model
    model.compile(optimizer=Adam(), 
                  loss={'dense_3': MeanSquaredError(), 'dense_4': CategoricalCrossentropy()},
                  loss_weights={'dense_3': 1.0, 'dense_4': 1.0})
    
    return model


class Network:
    def __init__(self, encode_type) -> None:
        self.network = create_snort_model(encode_type)


    def predict(self, state):
        return self.network.predict(state)
    
    
    def save_weight(self, filename):
        self.network.save_weight(filename)
    
    
    def load_weight(self, filename):
        self.network.load_weight(filename)