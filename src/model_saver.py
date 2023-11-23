import pandas as pd
from multiple_linear_regression import MultipleLinearRegression
import os

class ModelSaver:
    def __init__(self, model:MultipleLinearRegression) -> None:
        self.model = model
    
    def to_file(self, mode:str, directory:str=None) -> None:
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
        
        
            