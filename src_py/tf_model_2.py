import tensorflow as tf
import pickle, os, sys
import numpy as np
from .tf_model import calculate_deltas_unsigned, calculate_deltas_signed


class DataGenerator(tf.keras.utils.Sequence):
    """ Generates data for Keras models """
    def __init__(self, batch_size, dataset, configuration):
        self.batch_size = batch_size
        self.dataset = dataset
        self.configuration = configuration

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return self.dataset.n // self.batch_size

    def __getitem__(self, index):
        """ Generate one batch of data """
        x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s, _ = self.dataset.next_batch(self.batch_size)
        if self.configuration == "soft_weights":
            labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), 
                                       (1, weights.shape[-1]))
        if self.configuration == "soft_argmaxs":
            labels = hits_argmaxs / tf.tile(tf.reshape(tf.reduce_sum(hits_argmaxs, axis=1), 
                                                       (-1, 1)), (1, hits_argmaxs.shape[-1]))
        if self.configuration == "soft_c012s":
            labels = hits_c012s / tf.tile(tf.reshape(tf.reduce_sum(hits_c012s, axis=1), 
                                                     (-1, 1)), (1, hits_c012s.shape[-1]))
        if self.configuration == "regr_argmaxs":
            labels = argmaxs
        if self.configuration == "regr_c012s":
            labels = c012s
        if self.configuration == "regr_weights":
            labels = weights

        return x, labels


def compute_accuracy_and_mean(model, dataset, batch_size, args, at_most=None, filtered=False):
    """ Compute accuracy (within the ∆_max tolerance) and the error mean value """
    x = dataset.x
    calc_w = dataset.weights
    filt = dataset.filt
    
    if at_most:
        x = x[:at_most]
        calc_w = calc_w[:at_most]
        filt = filt[:at_most]
    if filtered:
        x = x[filt == 1.0]
        calc_w = calc_w[filt == 1.0]
    
    n_classes = calc_w.shape[-1]
    pred_w = model.predict(x, batch_size=batch_size, verbose=0)
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, n_classes))

    # Computing the mean of the difference between the most probable predicted 
    # class and the most probable true class (∆_class)      
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    mean = np.mean(calculate_deltas_signed(pred_argmaxs, calc_argmaxs, n_classes))

    # ACC (accuracy): averaging that most probable predicted class match for t
    # the most probable class within the ∆_max tolerance. ∆max specifiec the maximum 
    # allowed difference between the predicted class and the true class for an event 
    # to be considered correctly classified.
    delt_max = int(args.DELT_CLASSES)
    acc = (calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, n_classes) <= delt_max).mean()

    return acc, mean


class MonitoringUtils(tf.keras.callbacks.Callback):
    """ Callback for monitoring the model performance """
    def __init__(self, train_data, val_data, batch_size, args):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.n_batches = train_data.n // batch_size
        self.args = args

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        sys.stdout.write(f"Epoch {epoch + 1}/{int(self.args.EPOCHS)}\n")

    def on_batch_begin(self, batch, logs=None):
        if (batch + 1) % 10 == 0:
            sys.stdout.write(f" >>> batch {batch + 1}/{self.n_batches}\r")
    
    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.write("\nTraining:    acc: {:.4f} | mean: {:.4f}\n".format(
              *compute_accuracy_and_mean(self.model, self.train_data, self.batch_size,
                                         self.args, at_most=100_000, filtered=True)))
        sys.stdout.write("Validation:  acc: {:.4f} | mean: {:.4f}\n\n".format(
              *compute_accuracy_and_mean(self.model, self.val_data, self.batch_size, self.args)))


def regr_argmaxs_loss(y_true, y_pred):
    # TODO: not well learning close to angle = 0, 2pi
    # Old implementation:
    # self.loss = loss = tf.losses.mean_squared_error(self.argmaxs, sx)
    # new proposal by J. Kurek, does not work without correcting at analysis step
    # use for plotting script with "_topo" extension.
    # New implementation:
    return tf.reduce_mean(1 - tf.math.cos(y_true - y_pred))


class NeuralNetwork(tf.keras.Model):
    """ Configurable Neural Network class """

    def __init__(self, num_features, batch_size, args, lr=1e-3, input_noise=0.0):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.lr = lr
        self.configuration = args.TRAINING_METHOD
        self.n_features = num_features
        self.batch_size = batch_size
        self.n_classes = int(self.args.NUM_CLASSES)
        self.n_classes = {"soft_weights": self.n_classes, "soft_argmaxs": self.n_classes, 
                          "soft_c012s": self.n_classes, "regr_argmaxs": 1,
                          "regr_c012s": 3, "regr_weights": self.n_classes}
        self.n_classes = self.n_classes[self.configuration]
        self.n_epochs = int(self.args.EPOCHS)
        self.n_layers = int(self.args.LAYERS)
        self.n_units_per_layer = int(self.args.SIZE)
        self.dropout_rate = float(self.args.DROPOUT)
        self.input_layer = tf.keras.Input(shape=(self.n_features))
        self.input_noise = input_noise
        if input_noise:
            self.input_noise_layer = tf.keras.layers.GaussianNoise(self.input_noise)
        self.dense_layers, self.batch_norm_layers = [], []
        self.activation_layers, self.dropout_layers = [], []
        for i in range(self.n_layers):
            self.dense_layers.append(tf.keras.layers.Dense(
                units=self.n_units_per_layer, name=f"dense_{i}", use_bias=False))
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization(name=f"batch_norm_{i}"))
            self.activation_layers.append(tf.keras.layers.ReLU(name=f"relu_{i}"))
            self.dropout_layers.append(tf.keras.layers.Dropout(rate=self.dropout_rate))
        self.linear_layer = tf.keras.layers.Dense(units=self.n_classes, use_bias=False, name="linear")
        if self.configuration in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
            self.softmax_layer = tf.keras.layers.Softmax()
    
    def call(self, x):
        """ Pass tensors forward """
        input = x
        if self.input_noise:
            input = self.input_noise_layer(input)
        for i in range(self.n_layers):
            input = self.dense_layers[i](input)
            input = self.batch_norm_layers[i](input)
            input = self.activation_layers[i](input)
            input = self.dropout_layers[i](input)
        input = self.linear_layer(input)
        if self.configuration in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
            input = self.softmax_layer(input)
        return input
   
    def compile_model(self):
        """ Compile the model by setting an appropriate optimizer, 
        as well as the loss function """
        optimizer = {"GradientDescentOptimizer": tf.keras.optimizers.SGD, 
                     "AdadeltaOptimizer": tf.keras.optimizers.Adadelta, 
                     "AdagradOptimizer": tf.keras.optimizers.Adagrad,
                     "ProximalAdagradOptimizer": tf.compat.v1.train.ProximalAdagradOptimizer, 
                     "AdamOptimizer": tf.keras.optimizers.Adam,
                     "FtrlOptimizer": tf.keras.optimizers.Ftrl,
                     "RMSPropOptimizer": tf.keras.optimizers.RMSprop,
                     "ProximalGradientDescentOptimizer": tf.compat.v1.train.ProximalGradientDescentOptimizer}
        if self.configuration in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        elif self.configuration in ["regr_c012s", "regr_weights"]:
            loss = tf.keras.losses.MSE()
        elif self.configuration == "regr_argmaxs":
            loss = regr_argmaxs_loss
        else:
            raise ValueError(f"Unknown training method has been provided: {self.configuration}")
        self.compile(loss=loss, optimizer=optimizer[self.args.OPT](learning_rate=self.lr))

    def train(self, data):
        """ Train the model """
        train_data_generator = DataGenerator(batch_size=self.batch_size, dataset=data.train,
                                             configuration=self.configuration)
        self.fit(train_data_generator, epochs=self.n_epochs, verbose=0,
                callbacks=[MonitoringUtils(data.train, data.valid, self.batch_size, self.args)])

    def build_graph(self):
        """ Build the computational graph (you can call build_graph.summary() to
        see the architecture of the model: layers, output shapes) """
        x = tf.keras.layers.Input(shape=(self.n_features))
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=f"HiggsCP DNN ({self.configuration})")


def run(args):
    # Loading data
    data_points_path = os.path.join(args.IN, f"event_datasets{args.NUM_CLASSES}.obj")
    with open(data_points_path, 'rb') as f:
            data_points = pickle.load(f)
    num_features = data_points.train.x.shape[1]
    print(f"{num_features} features have been prepared.")
    
    # Building the model
    model = NeuralNetwork(num_features, 128, args)
    model.compile_model()
    
    # Training the model
    model.build_graph().summary()
    history = model.train(data_points)