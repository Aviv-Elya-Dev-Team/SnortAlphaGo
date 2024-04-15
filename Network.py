import torch
import torch.nn as nn
import torch.optim as optim
import random


# Define the neural network class
class Network(nn.Module):
    ENCODE_LEGAL, ENCODE_BOARD, ENCODE_BOTH = 0, 1, 2

    def __init__(self, encode_type, board_size):
        super(Network, self).__init__()
        self.hidden_layer_size = board_size**2
        self.board_size = board_size
        self.encode_type = encode_type
        self.input_size = self.calculate_input_size()

        self.shared_layer_1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.shared_layer_2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.head1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.head2 = nn.Linear(self.hidden_layer_size, 1)
        
        self.optimizer = optim.Adam(self.parameters())
        # Move the network to GPU if available
        if torch.cuda.is_available():
            self = self.cuda()

    def calculate_input_size(self):
        if self.encode_type == self.ENCODE_BOARD:
            return (self.board_size * self.board_size * 3) + 2
        elif self.encode_type == self.ENCODE_BOTH:
            return (self.board_size * self.board_size * 5) + 2
        else:
            return (self.board_size * self.board_size * 2) + 2

    def forward(self, x):
        x = torch.relu(self.shared_layer_1(x))
        x = torch.relu(self.shared_layer_2(x))
        out1 = torch.softmax(self.head1(x), dim=-1)
        out2 = self.head2(x)
        return out1, out2

    def train(self, games_history, epochs=1, batch_size=-1):
        if batch_size == -1:
            batch_size = len(games_history)

        for epoch in epochs:
            random.shuffle(games_history)
            for batchIdx in range(0, len(games_history), batch_size):
                # get batch sample
                sample = games_history[
                    batchIdx : min(len(games_history) - 1, batchIdx + batch_size)
                ]

                encoded_states, probabilities, winning_players = zip(*sample)

                encoded_states, probabilities, winning_players = (
                    torch.tensor(encoded_states),
                    torch.tensor(probabilities),
                    torch.tensor(winning_players),  # .reshape(-1, 1),
                )

                # train the model
                out_policy, out_value = self.predict(encoded_states)

                policy_loss = nn.CrossEntropyLoss(out_policy, probabilities)
                value_loss = nn.MSELoss(out_value, winning_players)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, x_encoded):
        if torch.cuda.is_available():
            x_encoded = x_encoded.cuda()
        return self(x_encoded)

    def save_model(self):
        torch.save(
            self.state_dict(), f"model/model_{self.encode_type}_{self.board_size}.pth"
        )

    @staticmethod
    def load_model(encode_type, board_size):
        model: Network = torch.load(f"model/model_{encode_type}_{board_size}.pth")
        model = model.load_state_dict(model)
        model.eval()
        return model
