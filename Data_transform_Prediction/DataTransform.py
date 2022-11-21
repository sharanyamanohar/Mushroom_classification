from datetime import datetime
from os import listdir
import pandas as pd
from logger import logging as lg
import numpy as np


class dataTransform:

    def __init__(self):

        self.logger = lg


    def read_dataset(self):
        """Here we are reading the dataset"""
        try:
            lg.info("we are inside dataframe")
            df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\mushroom_classify\mushrooms.csv")
            print(df)
        except Exception as e:
            print("you can check your log for more info if your code will fail")
            lg.error("error has occured")
            lg.exception(str(e))


#Calling the class
# obj=dataTransform()
# obj.read_dataset()

    def max_column(self):
        """With the help of the class we can display max columns"""
        try:
            lg.info("we are expanding display to see more columns")
            df=pd.set_option('display.max_columns', None)
            print(df)
        except Exception as e:
            print("you can check your log for more info if your code will fail")
            lg.error("error has occured")
            lg.exception(str(e))

#Calling the class
#obj=dataTransform()
#obj.max_column()

    def shape_dataset(self):
        """To know the shape of the dataset"""
        try:
            lg.info("we are inside dataframe")
            df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\mushroom_classify\mushrooms.csv")
            print("Number of rows", df.shape[0])
            print("Number of columns", df.shape[1])
        except Exception as e:
            print("you can check your log for more info if your code will fail")
            lg.error("error has occured")
            lg.exception(str(e))

#calling the class
#obj=dataTransform()
#obj.shape_dataset()

    def null_values_check(self):
        """To check whether it contains null values or not"""
        df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\mushroom_classify\mushrooms.csv")
        try:

            lg.info("We are checking the info of complete DataFrame")
            for i in df:
                if i=='?':
                    print("It contains null values",i)
                else:
                    print("No null values on dataset",i)
        except Exception as e:
            print("you can check your log for more info if your code will fail")
            lg.error("error has occured")
            lg.exception(str(e))

#Calling the dataset
#obj=dataTransform()
#obj.null_values_check()
