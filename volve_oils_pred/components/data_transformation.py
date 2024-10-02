import sys
import os
from dataclasses import dataclass

from volve_oils_pred.exception import Volve_Exception
from volve_oils_pred.logger import logging

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import MinMaxScaler

from volve_oils_pred.utils.main_utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transfromer_obj(self):
        ''''
        This function is responsible for data transformer
        '''
        try:
            columns_scailing=['ON_STREAM_HRS',
                              'AVG_DOWNHOLE_TEMPERATURE',
                              'AVG_ANNULUS_PRESS',
                              'AVG_CHOKE_SIZE_P',
                              'AVG_WHP_P','AVG_WHT_P']
            scale_pipeline=Pipeline(
                steps=[
                    ("scaler",MinMaxScaler())
                ]
            )

            logging.info(f"Columns used for minmaxscaler:{columns_scailing}")


            preprocessor=ColumnTransformer(
                [
                    ("scale_pipeline",scale_pipeline,columns_scailing)
                ]
            )

            return preprocessor
        except Exception as e:
            raise Volve_Exception(e,sys)
        
    def intiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_excel(train_path)
            test_df=pd.read_excel(test_path)

            logging.info("Reading test and train data completed")
            logging.info("Obtainig preprocessor object")

            preprocessing_obj=self.get_data_transformer_object()


            target_columns='BORE_OIL_VOL'
            
            drop_columns=['BORE_OIL_VOL','ON_STREAM_HRS','AVG_DOWNHOLE_TEMPERATURE',
                          'AVG_ANNULUS_PRESS','AVG_CHOKE_SIZE_P','AVG_WHP_P','AVG_WHT_P']
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_columns]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_columns]

            logging.info("Applying preprocessor on trainig and test data ")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessor object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        

        except Exception as e:
            raise Volve_Exception(e,sys)


