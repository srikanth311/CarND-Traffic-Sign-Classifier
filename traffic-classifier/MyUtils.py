import tensorflow as tf
import pickle

class MyUtils(object):
    @staticmethod
    def load_a_pickle_file(file):
        with open(file, mode='rb') as tra_file:
            pickle_file = pickle.load(tra_file)
        return pickle_file