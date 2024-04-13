""" This module contains the model itself (in its different configurations),
data generators needed for feeding it with the data provided batch-by-batch,
as well as the Keras callback class for utilising all the evaluation methods
available in evaluation_utils.py """

import tensorflow as tf, numpy as np
import pickle, os, sys, json, time, pickle
from .evaluation_utils import compute_accuracy_and_mean, compute_loss


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
    

class MonitoringUtils(tf.keras.callbacks.Callback):
    """ Callback for monitoring the model performance """
    def __init__(self, train_data, val_data, batch_size, history_path, args, previous_results=None):
        # Configuration attributes required by serialization mechanism        
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.history_path = history_path

        self.n_batches = train_data.n // batch_size
        self.args = args
        if previous_results is None:
            self.results = {"training_epoch_relative_loss" : [], 
                            "training_final_loss" : [], "validation_loss" : [],
                            "training_accuracy" : [], "validation_accuracy" : [],
                            "training_mean" : [], "validation_mean" : [],
                            "training_l1_norm" : [], "validation_l1_norm" : [],
                            "training_l2_norm" : [], "validation_l2_norm" : [],}
        else:
            self.results = previous_results

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        sys.stdout.write(f"\nEpoch {epoch + 1}/{int(self.args.EPOCHS)}\n")

    def on_batch_begin(self, batch, logs=None):
        if (batch + 1) % 10 == 0:
            sys.stdout.write(f" >>> batch {batch + 1}/{self.n_batches}\r")
    
    def on_epoch_end(self, epoch, logs=None):
        # Training loss
        epoch_relative_loss = logs.get('loss')
        sys.stdout.write("\nTraining epoch relative loss (convergence): {:.4f}\n".format(epoch_relative_loss))
        self.results["training_epoch_relative_loss"].append(str(epoch_relative_loss))
        
        if self.model.configuration != "soft_weights":
            # We do not need to monitor true training loss during the training as
            # the loss computed as an average over the batches is enough for tracing convergence
            if (epoch + 1) == int(self.args.EPOCHS):
                train_loss = compute_loss(self.model, self.train_data, self.batch_size, self.args)
                self.results["training_final_loss"].append(str(train_loss))
                sys.stdout.write("Training loss: {:.4f}\n".format(train_loss))
            
            # Computing validation loss for all the configurations except soft_weights
            val_loss = compute_loss(self.model, self.val_data, self.batch_size, self.args)
            self.results["validation_loss"].append(str(val_loss))
            sys.stdout.write("Validation loss: {:.4f}\n".format(val_loss))

        if self.model.configuration == "soft_weights":
            # Accuracy, mean, l1, l2 for training data
            acc, mean, l1, l2 = compute_accuracy_and_mean(
                self.model, self.train_data, self.batch_size, self.args, 
                at_most=100_000, filtered=True)
            self.results["training_accuracy"].append(str(acc))
            self.results["training_mean"].append(str(mean))
            self.results["training_l1_norm"].append(str(l1))
            self.results["training_l2_norm"].append(str(l2))
            sys.stdout.write("Training:      accuracy: {:.4f} | mean: {:.4f} | ".format(acc, mean) +
                            "L1 norm: {:.4f} | L2 norm: {:.4f}\n".format(l1, l2))
            
            # Accuracy, mean, l2, l2 for validation data
            acc, mean, l1, l2 = compute_accuracy_and_mean(
                self.model, self.val_data, self.batch_size, self.args, 
                at_most=None, filtered=True)
            self.results["validation_accuracy"].append(str(acc))
            self.results["validation_mean"].append(str(mean))
            self.results["validation_l1_norm"].append(str(l1))
            self.results["validation_l2_norm"].append(str(l2))
            sys.stdout.write("Validation:    accuracy: {:.4f} | mean: {:.4f} | ".format(acc, mean) +
                            "L1 norm: {:.4f} | L2 norm: {:.4f}\n".format(l1, l2))
    
    def on_train_end(self, logs=None):
        with open(os.path.join(self.history_path, "history.json"), "w") as file:
            json.dump(self.results, file, indent=2)
        output = os.path.join(self.history_path, f"configuration.json")
        with open(output, "w") as file:
            json.dump(self.model.args.__dict__, file, indent=2)


@tf.keras.utils.register_keras_serializable(package="ML_Model", name="regr_argmaxs_loss")
def regr_argmaxs_loss(y_true, y_pred):
    """ Loss function for the regr_argmaxs configuration. """
    return tf.reduce_mean(1 - tf.math.cos(y_true - y_pred))


@tf.keras.utils.register_keras_serializable(package="ML_Model", name="NeuralNetwork")
class NeuralNetwork(tf.keras.Model):
    """ Configurable Neural Network class """
    def __init__(self, args, num_features, batch_size, lr=1e-3, input_noise=0.0):
        super(NeuralNetwork, self).__init__()
        
        # Configuration attributes required by serialization mechanism
        self.n_features = num_features
        self.batch_size = batch_size
        self.args = args
        self.lr = lr
        self.input_noise = input_noise

        # Attributes defining the model architecture
        self.n_classes = int(self.args.NUM_CLASSES)
        self.n_classes = {"soft_weights": self.n_classes, "soft_argmaxs": self.n_classes, 
                          "soft_c012s": self.n_classes, "regr_argmaxs": 1,
                          "regr_c012s": 3, "regr_weights": self.n_classes}
        self.n_classes = self.n_classes[args.TRAINING_METHOD]
        self.n_layers = int(self.args.LAYERS)
        self.n_units_per_layer = int(self.args.SIZE)
        self.dropout_rate = float(self.args.DROPOUT)

        # Computational layers
        self.input_layer = tf.keras.Input(shape=(self.n_features))
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
        if self.args.TRAINING_METHOD in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
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
        if self.args.TRAINING_METHOD in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
            input = self.softmax_layer(input)
        return input
   
    def get_config(self):
        """ Return configuration data needed for model saving. """
        base_config = super().get_config()
        config = {
            "args" : pickle.dumps(self.args).decode("latin1"),
            "num_features" : self.n_features,
            "batch_size" : self.batch_size,
            "lr" : self.lr,
            "input_noise" : self.input_noise
        } 
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """ Load configuration data returned by self.get_config(). """
        args = pickle.loads(config.pop("args").encode("latin1"))
        return cls(args, **config)

    def compile_and_build(self):
        """ Compile the model by configuring some of the attributes, 
        setting an appropriate optimizer, as well as the loss function """
        self.configure(self.args)
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
            loss = tf.keras.losses.MeanSquaredError()
        elif self.configuration == "regr_argmaxs":
            loss = regr_argmaxs_loss
        else:
            raise ValueError(f"Unknown training method has been provided: {self.configuration}")
        self.compile(loss=loss, optimizer=optimizer[self.args.OPT](learning_rate=self.lr))
        self.build(input_shape=(None, self.n_features))

    def configure(self, args):
        """ Speicifying attributes (used by compile_and_build() and can also be used to
        continue training with new parameters ) """
        self.args = args
        self.configuration = args.TRAINING_METHOD
        self.history_path = os.path.normpath(f"results/{self.configuration}/")
        self.checkpoint_path = self.history_path
        if args.WEIGHTS_OUTPUT is None:
          timestamp = time.strftime("%Y-%m-%d_on_%H-%M-%S")
          self.checkpoint_path = os.path.join(self.checkpoint_path, timestamp, 
                                              os.path.normpath("checkpoint/cp.ckpt"))
          self.history_path = os.path.join(self.history_path, timestamp)
        else:
          self.checkpoint_path = os.path.join(self.checkpoint_path, args.WEIGHTS_OUTPUT, 
                                              os.path.normpath("checkpoint/cp.ckpt"))
          self.history_path = os.path.join(self.history_path, args.WEIGHTS_OUTPUT)
        
    def train(self, data, n_epochs, use_old_history=False):
        """ Train the model """
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, 
                                                         save_weights_only=True, verbose=1)
        # Training the model
        train_data_generator = DataGenerator(batch_size=self.batch_size, dataset=data.train,
                                        configuration=self.configuration)
        if use_old_history:
            with open(os.path.join("results", self.configuration, self.args.WEIGHTS_INPUT, 
                                   "history.json"), "r") as file:
                previous_results = json.load(file)
        else:
            previous_results = None
        monitoring_callback = MonitoringUtils(
            data.train, data.valid, self.batch_size, self.history_path, self.args, previous_results)
        self.fit(train_data_generator, epochs=n_epochs, verbose=0,
                callbacks=[monitoring_callback, cp_callback])

    def build_graph(self):
        """ Build the computational graph (you can call build_graph.summary() to
        see the architecture of the model: layers, output shapes) """
        x = tf.keras.layers.Input(shape=(self.n_features))
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=f"HiggsCP DNN ({self.configuration})")
    
    def save_model(self):
        """ Save the whole model (weights, variables, optimizer state). """
        if not os.path.exists(self.history_path):
            os.makedirs(self.history_path)
        tf.keras.models.save_model(self, os.path.join(self.history_path, "model.keras"))


def run(args):
    # Loading data
    data_points_path = os.path.join(args.IN, f"event_datasets_{args.NUM_CLASSES}.obj")
    with open(data_points_path, 'rb') as f:
        data_points = pickle.load(f)
    num_features = data_points.train.x.shape[1]
    print(f"Loaded data: {num_features} features have been prepared.")
    
    # Building the model
    model = NeuralNetwork(args, num_features, 128)
    model.compile_and_build()

    if args.ACTION == "train": 
        # Training and saving the whole model
        model.train(data_points, args.EPOCHS)
        model.save_model()

    if args.ACTION == "continue_training":
        if args.USE_CHECKPOINT:
            # This way of loading the model checkpoint makes it possible
            # to continue training not only with the last values of weights
            # but also with the last state of the optimizer.
            checkpoint = tf.train.Checkpoint(root=model, optimizer=model.optimizer)
            checkpoint.restore(os.path.join(
                "results", args.TRAINING_METHOD, args.WEIGHTS_INPUT, 
                os.path.normpath("checkpoint/cp.ckpt")))
        else:
            # This way of loading the whole model can provide us with the last 
            # saved values of the model weights. However, the optimizer will be
            # initialised one more time.
            model = tf.keras.models.load_model(os.path.join(
                "results", args.TRAINING_METHOD, args.WEIGHTS_INPUT, "model.keras"))
        model.configure(args)
        model.train(data_points, args.EPOCHS, use_old_history=True)
        model.save_model()
        
    if args.ACTION == "predict":
        if args.USE_CHECKPOINT:
            # This way of loading allows us to use only weights (which is enough 
            # for inference), so it is similar to model.load_model() taking 
            # the result of model.save() as an argument
            model.load_weights(str(os.path.join(
                "results", args.TRAINING_METHOD, args.WEIGHTS_INPUT, 
                os.path.normpath("checkpoint/cp.ckpt")).replace('\\', '/'))).expect_partial()
        else:
            model = tf.keras.models.load_model(
                os.path.join("results", args.TRAINING_METHOD, 
                             args.WEIGHTS_INPUT, "model.keras"))
        model.configure(args)
        
        print("Making predictions for the training and validation sets...")
        train_preds = model.predict(data_points.train.x)
        valid_preds = model.predict(data_points.valid.x)
        
        pred_path = os.path.join("results", args.TRAINING_METHOD, args.WEIGHTS_INPUT, "predictions")
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        
        train_preds_path = os.path.join(pred_path, "train_preds.npy")
        with open(train_preds_path, 'wb') as f:
            np.save(f, train_preds)
        print(f"Predictions for training data have been saved in {train_preds_path}")
        
        valid_preds_path = os.path.join(pred_path, "valid_preds.npy")
        with open(valid_preds_path, 'wb') as f:
            np.save(f, valid_preds)
        print(f"Predictions for validation data have been saved in {valid_preds_path}")