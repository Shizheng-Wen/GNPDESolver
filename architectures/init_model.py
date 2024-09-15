from .gino import GINO

def init_model(num_features, num_classes, kwargs):
    """
    Initialize the model with the given number of features and classes.
    """

    if kwargs.get("model_type") == "gino":
        return GINO(in_channels=num_features, 
                    out_channels=num_classes, 
                    **kwargs)
    else:
        raise ValueError(f"Model type {kwargs.get('model_type')} not supported.")

    