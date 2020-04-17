import numpy as np

class BasicSummary(object):
    # noinspection PyMethodMayBeStatic
    def summary_report(self, x_train, y_train, x_test, y_test, x_valid, y_valid):

        # The size of training set
        print("The size of training set is : {}".format(x_train.shape[0]))

        # The size of validation set
        print("The size of validation set is : {}".format(x_valid.shape[0]))

        # The size of test set is ?
        print("The size of test set is : {}".format(x_test.shape[0]))

        # The shape of a traffic sign image is
        print("The shape of a traffic sign image is : {}".format(x_train.shape[1:]))

        # The number of unique classes/labels in the data set is
        print("The number of unique classes/labels in the data set is : {}".format(len(np.unique(y_train))))

