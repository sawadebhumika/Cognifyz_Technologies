import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Average_Cost_for_two:float,
                 Votes:float,
                 Has_Table_booking:str,
                 Has_Online_delivery:str,
                 Is_delivering_now:str,
                 Price_range:str,
                 Rating_text:str,
                ):
        
        self.Average_Cost_for_two = Average_Cost_for_two
        self.Votes=Votes
        self.Has_Table_booking=Has_Table_booking
        self.Has_Online_delivery=Has_Online_delivery
        self.Is_delivering_now=Is_delivering_now
        self.Price_range=Price_range
        self.Rating_text = Rating_text
      
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Average Cost for two':[self.Average_Cost_for_two],
                'Votes':[self.Votes],
                'Has Table booking':[self.Has_Table_booking],
                'Has Online delivery':[self.Has_Online_delivery],
                'Is delivering now':[self.Is_delivering_now],
                'Price range':[self.Price_range],
                'Rating text':[self.Rating_text],
              
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)