MODELS = {}
LOSSES = {}

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

def build_model(config):
    return MODELS[config.model.model_name](**config.model.model_args)

def build_loss(config):
    return LOSSES[config.loss.loss_name](**config.loss.loss_args)