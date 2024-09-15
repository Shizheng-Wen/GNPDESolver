
class DynamicTrainer(TrainerBase):
    """
    Trainer for dynamic problems, i.e. problems that depend on time.
    """

    def __init__(self, args):
        super().__init__(args)
    
    def init_dataset(self, dataset_config):
        pass

    def init_model(self, model_config):
        pass

    def init_optimizer(self, optimizer_config):
        pass

    def train(self):
        pass    
        