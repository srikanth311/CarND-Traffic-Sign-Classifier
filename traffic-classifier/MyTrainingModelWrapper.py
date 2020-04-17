import tensorflow as tf
import numpy as np
import os
import sys

from MyPreprocessingWrapper import MyPreprocessingWrapper
from MyImageProcessor import MyImageProcessor
from MyUtils import MyUtils

from Visualization import Visualization

class MyTrainingModelWrapper(object):

    save_my_model_tf_session = None

    def __init__(self):
        ### Invoke preprocessing of the data
        ### If preprocessing is already complete, it just loads the pickle file which has the preprocessed data.
        ### If the preprocessing results file does not exist under "/training_output" folder, it will execute the preprocessing steps.
        self.training_data = MyPreprocessingWrapper.invoke_pre_processing()

        self.X_TRAIN = self.training_data.x_train_final
        self.Y_TRAIN = self.training_data.y_train_final

        self.X_VALID = self.training_data.x_valid_final
        self.Y_VALID = self.training_data.y_valid_final

        self.X_TEST = self.training_data.x_test_final
        self.Y_TEST = self.training_data.y_test_final

        self.learning_rate = None
        self.epochs = None
        self.batch_size = None

        self.features = None
        self.labels = None

        self.neural_network = None
        self.training_operation = None
        self.top_k_operations = None
        self.accuracy_operation = None

        ### Calling local functions
        self.init_hyper_parameters()
        self.init_network_params()
        self.create_a_network()
        self.create_training_pipeline()
        self.create_model_evaluation()

        self.create_top_k_operations()


    def init_hyper_parameters(self):
        self.learning_rate = 0.001
        self.epochs = 20
        self.batch_size = 128

    def init_network_params(self):
        self.features = tf.placeholder(tf.float32, [None, 32,32,1])
        #self.labels = tf.placeholder(tf.float32, [None, 43])
        self.labels = tf.placeholder(tf.int32, None)
        self.labels = tf.one_hot(self.labels, 43)

    # noinspection PyMethodMayBeStatic
    def create_a_network(self):
        mean = 0
        standard_deviation = 0.1
        dropout = 0.5

        ## ============================================================== ##
        # Layer 1 - Input = 32x32x1 - output = 32x32x32
        ## ============================================================== ##
        filter_size, input_channels, output_channels = 5, 1, 32
        conv1_Weights = tf.Variable(tf.truncated_normal((filter_size, filter_size, input_channels, output_channels), mean=mean, stddev=standard_deviation))
        conv1_biases = tf.Variable(tf.zeros(output_channels, 1))
        conv1 = tf.nn.conv2d(self.features, conv1_Weights, strides = [1,1,1,1], padding = "SAME")
        conv1 = tf.nn.bias_add(conv1, conv1_biases)

        # Activation
        conv1 = tf.nn.relu(conv1)

        #Polling. Input = 32x32x1. output = 16x16x32
        conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], 'VALID')

        print("Layer 1 completed")

        ## ============================================================== ##
        # Layer 2 - Input = 16x16x32 - output = 16x16x64
        ## ============================================================== ##
        filter_size, input_channels, output_channels = 5, 32, 64
        conv2_Weights = tf.Variable(tf.truncated_normal((filter_size, filter_size, input_channels, output_channels), mean=mean, stddev=standard_deviation))
        conv2_biases = tf.Variable(tf.zeros(output_channels, 1))
        conv2 = tf.nn.conv2d(conv1, conv2_Weights, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, conv2_biases)

        # Activation
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 16x16x64. Output = 8x8x64.
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        print("Layer 2 completed")


        ## ============================================================== ##
        # Layer 3 - Input = 8x8x64 - output = 8x8x128
        ## ============================================================== ##

        filter_size, input_channels, output_channels = 5, 64, 128
        conv3_Weights = tf.Variable(tf.truncated_normal((filter_size, filter_size, input_channels, output_channels), mean=mean, stddev=standard_deviation))
        conv3_biases = tf.Variable(tf.zeros(output_channels, 1))
        conv3 = tf.nn.conv2d(conv2, conv3_Weights, strides=[1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.bias_add(conv3, conv3_biases)

        # Activation
        conv3 = tf.nn.relu(conv3)

        # Pooling. Input = 8x8x128. Output = 4x4x128.
        conv3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        print("Layer 3 completed")


        ## ============================================================== ##
        # Flatten. Input = 4x4x128. Output = 2048
        ## ============================================================== ##
        tensor_size = 4 * 4 * 128
        fc = tf.contrib.layers.flatten(conv3, [1, tensor_size])
        #fc = tf.contrib.layers.flatten(conv2)

        print("Layer FLATTEN completed")
        ## ============================================================== ##
        # Layer 4 - Fully connected. Input = 2048 - output = 1024
        ## ============================================================== ##
        input_size, output_size = 2048, 1024
        fc1_weights = tf.Variable(tf.truncated_normal((input_size, output_size), mean, standard_deviation))
        fc1_biases = tf.Variable(tf.zeros(output_size))
        fc1 = tf.matmul(fc, fc1_weights)
        fc1 = tf.nn.bias_add(fc1, fc1_biases)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

        print("Layer 4 completed")
        ## ============================================================== ##
        # Layer 5 - Fully connected. Input = 1024 - output = 256
        ## ============================================================== ##
        input_size, output_size = 1024, 256
        fc2_weights = tf.Variable(tf.truncated_normal((input_size, output_size), mean, standard_deviation))
        fc2_biases = tf.Variable(tf.zeros(output_size))
        fc2 = tf.matmul(fc1, fc2_weights)
        fc2 = tf.nn.bias_add(fc2, fc2_biases)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)

        print("Layer 5 completed")
        ## ============================================================== ##
        # Layer 6 - Fully connected. Input = 256 - output = 43
        ## ============================================================== ##
        input_size, output_size = 256, 43
        fc3_weights = tf.Variable(tf.truncated_normal((input_size, output_size), mean, standard_deviation))
        fc3_biases = tf.Variable(tf.zeros(output_size))
        fc3 = tf.matmul(fc2, fc3_weights)
        fc3 = tf.nn.bias_add(fc3, fc3_biases)

        #fc3 = tf.nn.relu(fc3)
        #fc3 = tf.nn.dropout(fc3, dropout)
        print("Layer 6 completed")
        self.neural_network = fc3

    def create_training_pipeline(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.neural_network, labels=self.labels)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

    def create_top_k_operations(self):
        softmax_logits = tf.nn.softmax(logits = self.neural_network)
        self.top_k_operations = tf.nn.top_k(softmax_logits, k = 5)

    def create_model_evaluation(self):
        correct_prediction = tf.equal(tf.argmax(self.neural_network, 1), tf.argmax(self.labels, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def evaluate(self, X_data, y_data, BATCH_SIZE=128):
        NUM_EXAMPLES = X_data.shape[0]
        total_accuracy = 0
        #sess = tf.Session()
        sess = tf.get_default_session()

        for offset in range(0, NUM_EXAMPLES, BATCH_SIZE):
            endindex = offset + BATCH_SIZE
            batch_X, batch_Y = X_data[offset:endindex], y_data[offset:endindex]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.features:batch_X, self.labels:batch_Y })
            total_accuracy += (accuracy * len(batch_X))
            if (offset // BATCH_SIZE) % 10 == 0: print(".", end="", flush=True)
        return total_accuracy / NUM_EXAMPLES

    """
    def train_my_model(self):
        EPOCHS = self.epochs
        BATCH_SIZE = self.batch_size
        NUM_EXAMPLES = self.training_data.x_train.shape[0]
        print("Input number of examples : {}".format(NUM_EXAMPLES))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            print("EPOCH {}".format(epoch))
            for offset in range(0, NUM_EXAMPLES, BATCH_SIZE):
                endindex = offset + BATCH_SIZE
                batch_X, batch_Y = self.training_data.x_train[offset:endindex], self.training_data.y_train[offset:endindex]
                sess.run(self.training_operation, feed_dict={self.features: batch_X, self.labels: batch_Y})

                if (offset // BATCH_SIZE) % 10 == 0: print(".", end="", flush=True)

            train_accuracy = self.evaluate(self.training_data.x_train, self.training_data.y_valid, BATCH_SIZE, sess)
            validation_accuracy = self.evaluate(self.training_data.x_valid, self.training_data.y_valid, BATCH_SIZE, sess)

            print("Train Accuracy = {:.3f}".format(train_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, "saved_models/lenet_model2")

        print("Model Saved")
        
    """

    def train_my_model(self):
        EPOCHS = self.epochs
        BATCH_SIZE = self.batch_size
        NUM_EXAMPLES = self.X_TRAIN.shape[0]
        print("Input number of examples : {}".format(NUM_EXAMPLES))
        print("Input number of test data : {}".format(self.X_TEST.shape[0]))
        print("Input number of valid data : {}".format(self.X_VALID.shape[0]))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            print("EPOCH {}".format(epoch), end=" ")
            for offset in range(0, NUM_EXAMPLES, BATCH_SIZE):
                endindex = offset + BATCH_SIZE
                batch_X, batch_Y = self.X_TRAIN[offset:endindex], self.Y_TRAIN[offset:endindex]
                sess.run(self.training_operation, feed_dict={self.features:batch_X, self.labels:batch_Y })
                if (offset // BATCH_SIZE) % 10 == 0: print(".", end="", flush=True)

            valid_batch_X, valid_batch_Y = self.X_VALID, self.Y_VALID
            validation_accuracy = sess.run(self.accuracy_operation, feed_dict={self.features:valid_batch_X, self.labels:valid_batch_Y})

            #test_batch_X, test_batch_Y = self.X_TEST, self.Y_TEST
            #test_accuracy = sess.run(self.accuracy_operation, feed_dict={self.features: test_batch_X, self.labels: test_batch_Y})

            print(" :: Validation Accuracy - {0:8.3%}".format(validation_accuracy))
            #print(" :: Test Accuracy - {0:8.3%}".format(test_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, "saved_models/lenet_model2")

        print("Model Saved")


    def test(self):
        #sess = MyTrainingModelWrapper.invoke_model()
        #sess = tf.Session()
        #saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))

        saver = tf.train.Saver()
        saved_model_session = tf.Session()
        saved_model_session.run(tf.global_variables_initializer())
        saver.restore(saved_model_session, 'saved_models/lenet_model2')

        test_features, test_labels = self.X_TEST, self.Y_TEST
        test_accuracy = saved_model_session.run(self.accuracy_operation, feed_dict={self.features:test_features, self.labels:test_labels})
        print("Test Accuracy is :" + str(test_accuracy))

        """
        for offset in range(0, NUM_EXAMPLES, BATCH_SIZE):
            endindex = offset + BATCH_SIZE
            test_batch_X, test_batch_Y = self.training_data.x_test[offset:endindex], self.training_data.y_test[offset:endindex]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.features: test_batch_X, self.labels: test_batch_Y})
            total_accuracy += (accuracy * len(test_batch_X))
            print(total_accuracy)
            #if (offset // BATCH_SIZE) % 10 == 0: print(".", end="", flush=True)
        #return total_accuracy / NUM_EXAMPLES
        
        #tacc = self.evaluate(self.training_data.x_test, self.training_data.y_test)

        #test_features, test_labels=self.training_data.x_test, self.training_data.y_test
        #test_accuracy = sess.run(self.accuracy_operation, feed_dict={self.features:test_features, self.labels:test_labels})
        print("Testing Accuracy is :" + str(tacc))
        """

    def predict(self, images, labels):
        session = MyTrainingModelWrapper.invoke_model()
        my_image_processor = MyImageProcessor()
        _imgs=[]

        for img in images:
            _imgs.append( my_image_processor.apply_grayscale_and_normalize(img) )

        imgs = np.array(_imgs)
        imgs = np.reshape(imgs,(-1,32,32,1) )

        values, indices = session.run(self.top_k_operations, feed_dict = {self.features:imgs})
        signnames = Visualization.read_sign_names_from_csv_return()

        for idx, pset in enumerate(indices):
            print("")
            print( '=======================================================')
            print("Correct Sign :", labels[idx],"-",signnames[labels[idx]])
            print( '-------------------------------------------------------')
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][0],pset[0],signnames[pset[0]]))
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][1],pset[1],signnames[pset[1]]))
            print( '{0:7.2%} : {1: <2} - {2: <40}'.format(values[idx][2],pset[2],signnames[pset[2]]))

        print( '-------------------------------------------------------')

    @classmethod
    # noinspection PyMethodMayBeStatic
    def invoke_model(cls):
        #if cls.save_my_model_tf_session is not None:
        #    print("Model already exists - just returning it.")
        #    return cls.save_my_model_tf_session

        if not os.path.isfile("saved_models/lenet_model2.meta"): #Model will create these files.
            my_model = MyTrainingModelWrapper()
            my_model.train_my_model()
        #else:
        #    print("Model already exists - reusing it.")

        model_saver = tf.train.Saver()
        cls.save_my_model_tf_session = tf.Session()
        cls.save_my_model_tf_session.run(tf.global_variables_initializer())
        model_saver.restore(cls.save_my_model_tf_session, 'saved_models/lenet_model2')

        return cls.save_my_model_tf_session



#if __name__ == "__main__":
    #mytraining_wrapper = MyTrainingModelWrapper()
    #MyTrainingWrapper.invoke_model()
    #mytraining_wrapper.invoke_model()
    #mytraining_wrapper.test()
    #sys.exit(0)
