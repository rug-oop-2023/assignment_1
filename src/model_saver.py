import pandas as pd
import os

class ModelSaver:
    def __init__(self, model):
        self.model = model
    
    def to_file(self, mode, directory=None):
        if not directory:
            directory = os.getcwd()

        weights = self.model.get_weights()

        weights_df = pd.DataFrame(
            data=weights,
            index=[i for i in len(weights)]
        )
        
        match mode:
            case 'csv':
                weights_df.to_csv(os.path.join(directory, 'model_parameters.csv'))
            case 'pickle':
                weights_df.to_pickle(os.path.join(directory, 'model_parameters.csv'))
        
        
            