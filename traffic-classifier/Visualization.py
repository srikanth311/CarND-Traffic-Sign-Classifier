import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
import cv2
import csv

from MyVariables import MyVariables

class Visualization(object):

    # noinspection PyMethodMayBeStatic
    def bar_chart(self, y_labels, file_name):
        classes, counts = np.unique(y_labels, return_counts=True)
        saved_filename = "output_images/label_" + file_name + ".png"

        fig, ax = plt.subplots()

        ax.plot(classes, counts, color='green')
        ax.grid()

        plt.bar(classes, counts)
        plt.xlabel("Classes")
        plt.xticks(rotation=0)
        plt.ylabel("Count")
        plt.title(file_name + " --- Number of items per each class")

        fig.savefig(saved_filename)
        #plt.savefig("t1.png", dpi=300, format="png", bbox_inches='tight')
        plt.show()

    @classmethod
    def read_sign_names_from_csv_return(cls):
        l_sign_names = {}
        with open('../signnames.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # This skips the 1st row of the file as it has a header.
            for line in reader:
                l_sign_names[int(line[0])] = line[1]

        return l_sign_names

    # noinspection PyMethodMayBeStatic
    def read_sign_names_from_csv(self, my_varibales):
        l_sign_names = {}
        with open('../signnames.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader) # This skips the 1st row of the file as it has a header.
            for line in reader:
                l_sign_names[int(line[0])] = line[1]

        my_varibales.sign_names = l_sign_names



    # noinspection PyMethodMayBeStatic
    def display_random_images(self, images, labels, my_varibales, file_name, image_is_gray=False):

        images, labels = shuffle(images, labels)

        image_data = {}

        for label_category in np.unique(labels):
            image_data[label_category] = []

        for image, label_category in zip(images, labels):
            if len(image_data[label_category]) < 5: # We just need few (5) images to show.
                image_data[label_category].append(image)

        #print("Image Data length is :: {} ".format(len(image_data)))

        idx = 0
        fig, ax = plt.subplots(43, 1, figsize=(200, 60))
        fig.subplots_adjust(hspace = .5)
        ax = ax.ravel()

        for display_image_labels, display_images  in image_data.items():
            if image_is_gray:
                bin_image = np.zeros((32, 32*5)) # To display 5 images.
            else:
                bin_image = np.zeros((32, 32*5, 3)) # Last parameter is # of channels (R, G, B)

            ind = 0
            for image in display_images:
                if image_is_gray:
                    bin_image[:, ind:ind+32] = image
                else:
                    bin_image[:, ind:ind+32, :] = image
                ind = ind + 32

            ax[idx].axis('off')
            if image_is_gray:
                ax[idx].imshow(bin_image, cmap='gray') # For gray image
            else:
                ax[idx].imshow(np.uint8(bin_image))

            ax[idx].set_title(str(display_image_labels) + ":" + my_varibales.sign_names[display_image_labels])
            idx = idx+1

        if image_is_gray:
            saved_filename = "output_images/sample_images_gray_{}.png".format(file_name)
        else:
            saved_filename = "output_images/sample_images_{}.png".format(file_name)
        fig.savefig(saved_filename)

    # noinspection PyMethodMayBeStatic

    def show_augmented_images(self, augmented_images, disp_label_category_for_test_output, my_varibales):
        fig, axs = plt.subplots(1, 5, figsize=(6, 6))
        plt.title("Trans-Images for sample label 12 - " + my_varibales.sign_names[disp_label_category_for_test_output], loc='right')
        axs = axs.ravel()
        for i in range(5):
            image = augmented_images[i]
            axs[i].axis('off')
            axs[i].imshow(image*255, cmap='gray')

        fig.savefig("output_images/augmented_image_for_label_{}.png".format(disp_label_category_for_test_output))


