import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from LoadData import LoadData
from MyVariables import MyVariables
from BasicSummary import BasicSummary
from Visualization import Visualization

from MyImageProcessor import MyImageProcessor

class MyPreprocessingWrapper(object):

    check_preprocessing_completed = None

    def __init__(self):
        ### Initializing all the required variables.
        self.x_train = []
        self.y_train = []

        self.x_test  = []
        self.y_test  = []

        self.x_valid = []
        self.y_valid = []

        self.x_train_aug = []
        self.y_train_aug = []

        self.x_train_gray = []
        self.x_valid_gray = []
        self.x_test_gray = []

        self.x_train_shuffle = []
        self.x_valid_shuffle = []
        self.x_test_shuffle = []

        self.y_train_shuffle = []
        self.y_valid_shuffle = []
        self.y_test_shuffle = []

        self.y_train_hot = []
        self.y_valid_hot = []
        self.y_test_hot = []

        self.x_train_reshape = []
        self.x_valid_reshape = []
        self.x_test_reshape = []

        self.x_train_final = []
        self.x_valid_final = []
        self.x_test_final = []

        self.y_train_final = []
        self.y_valid_final = []
        self.y_test_final = []



        #self._x_train_gray = []
        #self._x_test_gray = []

        ### Creating class instances of supporting classes.
        self.my_variables = MyVariables()
        self.bs = BasicSummary()
        self.vz = Visualization()
        self.my_image_processor = MyImageProcessor()

        ### Invoking the current class methods.
        self.invoke_data_summary()
        self.preprocessing()
        self.image_augmentation()
        self.split_and_shuffle_data()
        self.apply_one_hot_encoding_to_labels()
        self.reshape_features_data()
        self.get_final_data_values()


    ### If the preprocessing is already completed, then it just loads the processed data from the pickle file.
    ### Else, it preprocess the data and saves the training data into a pickle file.
    ### This will be invoked from MyTrainingModelWrapper.py file.
    @classmethod
    def invoke_pre_processing(cls):
        if cls.check_preprocessing_completed is not None:
            print("loading exsting pickle file")
            return cls.check_preprocessing_completed
        elif os.path.isfile("training_output/training_data.p"):
            print("loading exsting pickle file which has processed data")
            with open('training_output/training_data.p', 'rb') as input_training_file:
               cls.check_preprocessing_completed = pickle.load(input_training_file)
        else:
            print("Writing pickle file")
            cls.check_preprocessing_completed = MyPreprocessingWrapper()
            with open('training_output/training_data.p', 'wb') as training_file:
                pickle.dump(cls.check_preprocessing_completed, training_file)

        return cls.check_preprocessing_completed

    # noinspection PyMethodMayBeStatic
    def invoke_data_summary(self):
        # Instantiate classes

        # Load Data from pickle files
        ld = LoadData(self.my_variables.training_file, self.my_variables.testing_file, self.my_variables.validation_file)
        train_test_valid_data = ld.get_data()

        #########################################################################################################
        self.x_train, self.y_train = train_test_valid_data[0], train_test_valid_data[1]
        self.x_test, self.y_test = train_test_valid_data[2], train_test_valid_data[3]
        self.x_valid, self.y_valid = train_test_valid_data[4], train_test_valid_data[5]

        #########################################################################################################
        # Basic Summary of dataset
        self.bs.summary_report(self.x_train, self.y_train, self.x_test, self.y_test, self.x_valid, self.y_valid)

        #########################################################################################################
        # Exploratory visualization for train data
        self.vz.bar_chart(self.y_train, "train_data")
        # Exploratory visualization for train data
        self.vz.bar_chart(self.y_test, "test_data")
        # Exploratory visualization for train data
        self.vz.bar_chart(self.y_valid, "validation_data")

        #########################################################################################################
        self.vz.read_sign_names_from_csv(self.my_variables)
        self.vz.display_random_images(self.x_train, self.y_train, self.my_variables, "train")
        #self.vz.display_random_images(self.x_test, self.y_test, self.my_variables, "test")
        #self.vz.display_random_images(self.x_valid, self.y_valid, self.my_variables, "valid")

    # noinspection PyMethodMayBeStatic
    def preprocessing(self):
        x_train_gray = []
        x_test_gray = []
        x_valid_gray = []

        for image in self.x_train:
            x_train_gray.append(self.my_image_processor.apply_grayscale_and_normalize(image))

        for image in self.x_test:
            x_test_gray.append(self.my_image_processor.apply_grayscale_and_normalize(image))

        for image in self.x_valid:
            x_valid_gray.append(self.my_image_processor.apply_grayscale_and_normalize(image))

        self.x_train_gray = np.array(x_train_gray)
        self.x_test_gray = np.array(x_test_gray)
        self.x_valid_gray = np.array(x_valid_gray)

        self.vz.display_random_images(self.x_train_gray, self.y_train, self.my_variables, "train", True)
        self.vz.display_random_images(self.x_test_gray, self.y_test, self.my_variables, "test", True)

    # noinspection PyMethodMayBeStatic
    '''
    def image_augmentation_common(self, images, labels):
        images = self.x_train  # images will be gray images.
        labels = self.y_train  # labels will be the actual lables.

        image_data = {}
        num_unique_labels = len(np.unique(labels))
        for label_category in np.unique(labels):
            image_data[label_category] = []

        for image, label_category in zip(images, labels):
            image_data[label_category].append(image)

        _X = []
        _Y = []

        disp_images = []
        
        for lbl_category, imgs in image_data.items():
            num_of_images = len(imgs)
            _X.extend(imgs)
            _Y.extend([lbl_category] * num_of_images)
    '''

    # noinspection PyMethodMayBeStatic
    def image_augmentation(self):

        images = self.x_train_gray # images will be gray images.
        labels = self.y_train # labels will be the actual lables.

        image_data = {}
        num_unique_labels = len(np.unique(labels))
        for label_category in np.unique(labels):
            image_data[label_category] = []

        for image, label_category in zip(images, labels):
            image_data[label_category].append(image)

        _X_train = []
        _Y_train = []

        disp_images = []

        for lbl_category, imgs in image_data.items():
            num_of_images = len(imgs)
            _X_train.extend(imgs)
            _Y_train.extend([lbl_category] * num_of_images)

            ind = 0
            cnt = num_of_images
            target = 1200
            counter = 1
            disp_label_category_for_test_output = 6
            print("label cat is :::: {}".format(lbl_category))

            while cnt <= target:
                trans_img = self.my_image_processor.apply_warp_augmentation_on_an_image(imgs[ind])
                _X_train.append(trans_img)
                _Y_train.append(lbl_category)
                cnt += 1
                ind += 1 if (ind < num_of_images - 1) else 0
                #print("label cat is : {}".format(lbl_category))

                if (lbl_category == disp_label_category_for_test_output) and (len(disp_images) < 5):
                    #print("DISP Image for label cat 12 : {}".format(counter))
                    counter = counter + 1
                    disp_images.append(trans_img)


        self.x_train_aug = np.array(_X_train)
        self.y_train_aug = np.array(_Y_train)

        self.vz.bar_chart(self.y_train_aug, "augmented_train_data")
        #print("Disp images are ::: {}".format(len(disp_images)))
        self.vz.show_augmented_images(disp_images, disp_label_category_for_test_output, self.my_variables)


    def split_and_shuffle_data(self):
        self.x_train_shuffle, self.y_train_shuffle = shuffle(self.x_train_aug, self.y_train_aug)
        self.x_valid_shuffle, self.y_valid_shuffle = shuffle(self.x_valid_gray, self.y_valid)
        self.x_test_shuffle, self.y_test_shuffle = shuffle(self.x_test_gray, self.y_test)

    # noinspection PyMethodMayBeStatic
    def apply_one_hot_encoding_to_labels(self):
        num_train_label_classes = len(np.unique(self.y_train_shuffle))
        self.y_train_hot = np.eye(num_train_label_classes)[self.y_train_shuffle]

        num_test_label_classes = len(np.unique(self.y_test_shuffle))
        self.y_test_hot = np.eye(num_test_label_classes)[self.y_test_shuffle]

        num_valid_label_classes = len(np.unique(self.y_valid_shuffle))
        #print(num_valid_label_classes)
        self.y_valid_hot = np.eye(num_valid_label_classes)[self.y_valid_shuffle]


    # noinspection PyMethodMayBeStatic
    def reshape_features_data(self):
        # Reshaping input dataset to match the network requirements
        self.x_train_reshape = np.reshape(self.x_train_shuffle, (-1, 32, 32, 1))
        self.x_test_reshape = np.reshape(self.x_test_shuffle, (-1, 32, 32, 1))
        self.x_valid_reshape = np.reshape(self.x_valid_shuffle, (-1, 32, 32, 1))


    def get_final_data_values(self):
        self.x_train_final = self.x_train_reshape
        self.y_train_final = self.y_train_hot

        self.x_valid_final = self.x_valid_reshape
        self.y_valid_final = self.y_valid_hot

        self.x_test_final = self.x_test_reshape
        self.y_test_final = self.y_test_hot

        ## Empty all other variables as we have the final data in the above variables.
        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        self.x_valid = []
        self.y_valid = []

        self.x_train_aug = []
        self.y_train_aug = []

        self.x_train_gray = []
        self.x_valid_gray = []
        self.x_test_gray = []

        self.x_train_shuffle = []
        self.x_valid_shuffle = []
        self.x_test_shuffle = []

        self.y_train_shuffle = []
        self.y_valid_shuffle = []
        self.y_test_shuffle = []

        self.y_train_hot = []
        self.y_valid_hot = []
        self.y_test_hot = []

        self.x_train_reshape = []
        self.x_valid_reshape = []
        self.x_test_reshape = []


        # The size of final data sets
        print("The size of the final training set is : {}".format(self.x_train_final.shape[0]))
        print("The size of the final training set is : {}".format(self.x_valid_final.shape[0]))
        print("The size of the final training set is : {}".format(self.x_test_final.shape[0]))

        print("The size of the initial training set is : {}".format(len(self.x_train)))
        print("The size of the initial valid set is : {}".format(len(self.x_valid)))
        print("The size of the initial test set is : {}".format(len(self.x_test)))


#if __name__ == "__main__":
    # main_wrapper = MyPreprocessingWrapper().invoke_pre_processing()
    #main_wrapper.invoke_data_summary()
    #main_wrapper.preprocessing()
    ##main_wrapper.image_augmentation()
    #main_wrapper.split_and_shuffle_data()
    #main_wrapper.apply_one_hot_encoding_to_labels()
    ##main_wrapper.reshape_features_data()
    #main_wrapper.get_final_data_values()

