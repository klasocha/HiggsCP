import tensorflow as tf
from math import floor
import pickle
import numpy as np
from .tf_model import calculate_deltas_unsigned, calculate_deltas_signed


class DataGenerator(tf.keras.utils.Sequence):
    """ Generates data for Keras models """
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset
    
    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(self.dataset.n / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        x, weights, _, _, _, _, _  = self.dataset.next_batch(self.batch_size)
        weights = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), (1, weights.shape[-1]))
        return x, weights


class NeuralNetwork(tf.keras.Model):
    def __init__(self, num_features, args):
        super(NeuralNetwork, self).__init__()
        self.n_features = num_features
        self.n_classes = int(args.NUM_CLASSES)
        self.n_epochs = int(args.EPOCHS)
        self.n_layers = int(args.LAYERS)
        self.n_units_per_layer = int(args.SIZE)
        self.dense_layers, self.batch_norm_layers, self.activation_layers = [], [], []
        for i in range(self.n_layers):
            if i == 0:
                self.dense_layers.append(tf.keras.layers.Dense(input_shape=(self.n_features,),
                    units=self.n_units_per_layer, name=f"dense_{i}", use_bias=False))
            else:
                self.dense_layers.append(tf.keras.layers.Dense(
                    units=self.n_units_per_layer, name=f"dense_{i}", use_bias=False))
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization(name=f"batch_norm_{i}"))
            self.activation_layers.append(tf.keras.layers.ReLU(name=f"relu_{i}"))
        self.linear_layer = tf.keras.layers.Dense(units=self.n_classes, use_bias=False, name="linear")
        self.softmax_layer = tf.keras.layers.Softmax()
    
    def call(self, x):
        for i in range(self.n_layers):
            x = self.dense_layers[i](x)
            x = self.batch_norm_layers[i](x, training=True)
            x = self.activation_layers[i](x)
        x = self.linear_layer(x)
        return self.softmax_layer(x)
   
    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), 
                     optimizer=optimizer, metrics=['accuracy'])

    def train(self, data, batch_size):
        train_data_generator = DataGenerator(batch_size=batch_size, dataset=data.train)
        validation_data_generator = DataGenerator(batch_size=batch_size, dataset=data.valid)
        history = self.fit(train_data_generator,
                           validation_data=validation_data_generator,
                           epochs=self.n_epochs)

def run(args):
    points_path = "data/event_datasets.obj"
    with open(points_path, 'rb') as f:
            points = pickle.load(f)

    num_features = points.train.x.shape[1]
    print(f"{num_features} features have been prepared.")
    model = NeuralNetwork(num_features, args)
    model.compile_model()
    model.train(points, batch_size=128)
    model.summary()
    calc_w = points.train.weights
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, args.NUM_CLASSES))
    pred_w = model.predict(points.train.x, batch_size=128)

    # Computing the mean of the difference between the most probable predicted 
    # class and the most probable true class (∆_class)      
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    calc_pred_argmaxs_abs_distances = calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, args.NUM_CLASSES)
    calc_pred_argmaxs_signed_distances = calculate_deltas_signed(pred_argmaxs, calc_argmaxs, args.NUM_CLASSES)
    mean = np.mean(calc_pred_argmaxs_signed_distances)

    # ACC (accuracy): averaging that most probable predicted class match for t
    # the most probable class within the ∆_max tolerance. ∆max specifiec the maximum 
    # allowed difference between the predicted class and the true class for an event 
    # to be considered correctly classified.
    delt_max = args.DELT_CLASSES
    acc = (calc_pred_argmaxs_abs_distances <= delt_max).mean()

    print(mean, acc)