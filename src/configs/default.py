import ml_collections
from ml_collections import ConfigDict

def get__config():
    config = ConfigDict()

    config.model = ConfigDict()
    config.model.hidden_sizes = [256, 256, 256, 256]
    config.model.output_size = 1
    config.model.activation = 'relu'  # Options could be 'relu', 'tanh', etc.
    config.model.weight_fact = None  # Example: {'mean': 0.0, 'stddev': 0.1} or None

    config.training = ConfigDict()
    config.training.learning_rate = 1e-3
    config.training.batch_size = 64
    config.training.num_epochs = 1000
    config.training.momentum = 0.9
    config.training.loss_weights = {}  # Initial loss weights can be set here

    config.weighing.scheme = 'ntk'  # Options could be 'ntk', 'grad_norm', etc.
    config.weighing.momentum = 0.9  # Momentum for updating loss weights
    config.weighing.update_freq = 1000  # Update loss weights every n steps

    return config