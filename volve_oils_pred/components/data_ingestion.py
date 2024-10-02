import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from volve_oils_pred.exception import Volve_Exception
from volve_oils_pred.logger import logging

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts","train.xlsx")
    test_data_path:str=os.path.join("artifacts","test.xlsx")
    raw_data_path:str=os.path.join("artifacts","data.xlsx")

class DataIngestion:
    def __init__(self):
        self.data_ingestion=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or components")

        try:
            df=pd.read_excel("notebook\Volve_dataframe.xlsx")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)
            df.to_excel(self.data_ingestion.raw_data_path,index=False,header=True)

            logging.info("Train and Test split intiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_excel(self.data_ingestion.train_data_path,index=False,header=True)
            test_set.to_excel(self.data_ingestion.test_data_path,index=False,header=True)

            logging.info("Ingestion data is completed")

            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )

        except Exception as e:
            raise Volve_Exception(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()