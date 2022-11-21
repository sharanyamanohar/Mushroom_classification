from sklearn.preprocessing import LabelEncoder
import pandas as pd
from logger import logging as lg
import os
import csv

class encoder:

    def __init__(self):

        self.logger = lg


    def DataFrame(self):
       try:
           le = LabelEncoder()
           df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\mushroom_classify\mushrooms.csv")
           for column in df.columns:
               df[column] = le.fit_transform(df[column])

               df1=df.to_csv(r"C:\Users\Admin\PycharmProjects\mushroom_classify\encoder_mushroom_file.csv")
               print(df1)


       except Exception as e:
          lg.exception(e)
          return e
#calling the encoder class
obj=encoder()
obj.DataFrame()


