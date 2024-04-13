import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.optimizers import Adam
from Node import ENCODE_BOARD, ENCODE_LEGAL, ENCODE_BOTH


class Network:
    def __init__(self, encode_type, board_size) -> None:
        self.board_size = board_size
        self.encode_type = encode_type
        self.network = self.create_snort_model(self.encode_type)
        self.input_size = (self.board_size * self.board_size * 2) + 2
        if encode_type == ENCODE_BOARD:
            self.input_size = (self.board_size * self.board_size * 3) + 2
        elif encode_type == ENCODE_BOTH:
            self.input_size = (self.board_size * self.board_size * 5) + 2

    def predict(self, state):
        return self.network.predict(state, verbose=0)

    def save_model(self, filename):
        self.network.save(filename)

    def load_model(self, filename):
        self.network = load_model(filename)

    def compile_model(self):
        self.network.compile(
            optimizer=Adam(), loss=[categorical_crossentropy, mean_squared_error]
        )

    def train(self, x, y, epochs):
        self.network.fit(x, y, epochs=epochs, verbose=0, use_multiprocessing=True)

    def create_snort_model(self, encode_type):

        # Define input layer
        input_size = (self.board_size * self.board_size * 2) + 2
        if encode_type == ENCODE_BOARD:
            input_size = (self.board_size * self.board_size * 3) + 2
        elif encode_type == ENCODE_BOTH:
            input_size = (self.board_size * self.board_size * 5) + 2

        input_layer = Input(shape=(input_size,))

        # Define shared hidden layers
        shared_layer_1 = Dense(20, activation="relu")(input_layer)
        shared_layer_2 = Dense(20, activation="relu")(shared_layer_1)

        # First head
        head1 = Dense(
            self.board_size * self.board_size,
            activation="softmax",
        )(shared_layer_2)

        # Second head
        head2 = Dense(1)(shared_layer_2)

        # Define the model
        model = Model(inputs=input_layer, outputs=[head1, head2])

        # Compile the model with different loss functions for each head
        model.compile(
            optimizer=Adam(), loss=[categorical_crossentropy, mean_squared_error]
        )

        model.summary()

        return model
