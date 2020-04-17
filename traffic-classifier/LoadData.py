from MyUtils import MyUtils
from MyVariables import MyVariables

import pickle

class LoadData(object):

    def __init__(self, training_file, testing_file, validation_file):
        self.training_file = training_file
        self.testing_file = testing_file
        self.validation_file = validation_file

    # noinspection PyMethodMayBeStatic
    def get_data(self):
        train_data = MyUtils.load_a_pickle_file(self.training_file)
        test_data = MyUtils.load_a_pickle_file(self.testing_file)
        validation_data = MyUtils.load_a_pickle_file(self.validation_file)

        x_train, y_train = train_data['features'], train_data['labels']
        x_test, y_test = test_data['features'], test_data['labels']
        x_valid, y_valid = validation_data['features'], validation_data['labels']

        print(y_train)
        train_test_valid_data = (x_train, y_train,
                                 x_test, y_test,
                                 x_valid, y_valid
                                )
        return train_test_valid_data





