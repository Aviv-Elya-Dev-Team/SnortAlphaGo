import numpy as np
class Model:
    def predict(self, state):
        red, blue = np.random.rand(10, 10), np.random.rand(10, 10)
        red_legal, blue_legal = state[:100].reshape((10, 10)), state[100:].reshape((10,10))
        red[red_legal==False], blue[blue==False] = 0, 0
        return red, blue
        