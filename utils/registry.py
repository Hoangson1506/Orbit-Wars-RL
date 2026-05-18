MODELS = {}
LOSSES = {}
ALGORITHMS = {}

def register_model(name):
    def wrapper(cls):
        MODELS[name] = cls
        return cls
    return wrapper

def register_loss(name):
    def wrapper(cls):
        LOSSES[name] = cls
        return cls
    return wrapper

def register_algorithm(name):
    def wrapper(cls):
        ALGORITHMS[name] = cls
        return cls
    return wrapper

def build_model(config):
    return MODELS[config.model.model_name](**config.model.model_args)

def build_loss(config):
    return LOSSES[config.loss.loss_name](**config.loss.loss_args)

def build_algorithm(config):
    return ALGORITHMS[config.algorithm.name]