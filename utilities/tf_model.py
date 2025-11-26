""" This module contains the model itself (in its different configurations),
data generators needed for feeding it with the data provided batch-by-batch,
as well as the Keras callback class for utilising all the evaluation methods
available in evaluation_utils.py """

import tensorflow as tf, numpy as np
from tensorflow import keras as keras
import pickle, os, sys, json, pickle, math, shutil
from .evaluation_utils import compute_accuracy_and_mean, compute_loss


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras models """
    def __init__(self, batch_size, dataset, configuration):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.configuration = configuration

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return math.ceil(self.dataset.n / self.batch_size)

    def __getitem__(self, idx):
        """ Generate one batch of data """
        x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s, _ = self.dataset.next_batch(self.batch_size)
        if self.configuration == "soft_weights":
            labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), 
                                       (1, weights.shape[-1]))
        if self.configuration == "soft_argmaxs":
            hits_argmaxs = hits_argmaxs[:, :-1]
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
    

class MonitoringUtils(keras.callbacks.Callback):
    """ Callback for monitoring the model performance """
    def __init__(self, train_data, val_data, batch_size, n_epochs, delta_max_tolerance, 
                 output_location):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.results_location = os.path.join(output_location, "model_state")
        self.output_location = os.path.join(output_location, "history.json")
        self.delta_max_tolerance = delta_max_tolerance

        self.n_batches = math.ceil(train_data.n / batch_size)
        self.results = {
            "training_epoch_relative_loss" : [], 
            "training_final_loss" : [], "validation_loss" : [],
            "training_accuracy" : [], "validation_accuracy" : [],
            "training_mean" : [], "validation_mean" : [],
            "training_l1_norm" : [], "validation_l1_norm" : [],
            "training_l2_norm" : [], "validation_l2_norm" : [],
            "best_epoch" : [0]}
        if os.path.exists(self.output_location):
            with open(self.output_location, "r") as file:
                self.results = json.load(file)

    def on_epoch_begin(self, epoch, logs=None):
        sys.stdout.write(f"\nEpoch {epoch + 1}/{self.n_epochs}\n")

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
            if (epoch + 1) == self.n_epochs:
                train_loss = compute_loss(self.model, self.train_data, self.batch_size, filtered=True)
                self.results["training_final_loss"].append(str(train_loss))
                sys.stdout.write("Training loss: {:.4f}\n".format(train_loss))
            
            # Computing validation loss for all the configurations except soft_weights
            val_loss = compute_loss(self.model, self.val_data, self.batch_size, filtered=True)
            self.results["validation_loss"].append(str(val_loss))
            sys.stdout.write("Validation loss: {:.4f}\n".format(val_loss))

        if self.model.configuration == "soft_weights":
            # Accuracy, mean, l1, l2 for training data
            acc, mean, l1, l2 = compute_accuracy_and_mean(
                self.model, self.train_data, self.batch_size, self.delta_max_tolerance, 
                at_most=100_000, filtered=True)
            self.results["training_accuracy"].append(str(acc))
            self.results["training_mean"].append(str(mean))
            self.results["training_l1_norm"].append(str(l1))
            self.results["training_l2_norm"].append(str(l2))
            sys.stdout.write("Training:      accuracy: {:.4f} | mean: {:.4f} | ".format(acc, mean) +
                            "L1 norm: {:.4f} | L2 norm: {:.4f}\n".format(l1, l2))
            
            # Accuracy, mean, l2, l2 for validation data
            acc, mean, l1, l2 = compute_accuracy_and_mean(
                self.model, self.val_data, self.batch_size, self.delta_max_tolerance, 
                at_most=None, filtered=True)
            self.results["validation_accuracy"].append(str(acc))
            self.results["validation_mean"].append(str(mean))
            self.results["validation_l1_norm"].append(str(l1))
            self.results["validation_l2_norm"].append(str(l2))
            sys.stdout.write("Validation:    accuracy: {:.4f} | mean: {:.4f} | ".format(acc, mean) +
                            "L1 norm: {:.4f} | L2 norm: {:.4f}\n".format(l1, l2))
        
        # Updating the history file
        with open(self.output_location, "w") as file:
            json.dump(self.results, file, indent=2)

    def on_train_end(self, logs=None):
        sys.stdout.write(
            f"\nTraining has finished. Training history is available in {str(self.output_location)}\n")
        
        # Saving the best model as "model.weights" based on either the maximum value
        # of validation accuracy or the minimum value of validation loss and then
        # writing the best epoch index in history.json
        if self.model.configuration == "soft_weights":
            best_epoch = np.argmax(self.results["validation_accuracy"]) + 1
            print("The best results (validation accuracy) have been achieved",
                  f"at epoch #{best_epoch}.")
        else:
            best_epoch = np.argmin(self.results["validation_loss"]) + 1
            print("The best results (validation loss) have been achieved",
                  f"at epoch #{best_epoch}.")  
            
        self.results["best_epoch"][0] = str(best_epoch)
        with open(self.output_location, "w") as file:
            json.dump(self.results, file, indent=2)
        
        best_weights_path = os.path.join(self.results_location, 
                                            f"model_epoch_{best_epoch:02d}.weights.h5")
        new_best_weights_path = os.path.join(self.results_location, "model.weights.h5")
        shutil.copy2(best_weights_path, new_best_weights_path)
        print(f"The best weights have been saved in {new_best_weights_path}.")


def regr_argmaxs_loss(y_true, y_pred):
    """ Loss function for the regr_argmaxs configuration. """
    return tf.reduce_mean(1 - tf.math.cos(y_true - y_pred))


class NeuralNetwork(keras.Model):
    """ Configurable Neural Network class """
    def __init__(self, configuration, n_features, n_classes, n_layers, n_units_per_layer, input_noise_rate, 
                 dropout_rate, opt, **kwargs):
        super().__init__(**kwargs)

        # Architecture parameters
        self.n_features = n_features
        self.configuration = configuration
        self.n_classes = {"soft_weights": n_classes, "soft_argmaxs": n_classes - 1, 
                          "soft_c012s": n_classes, "regr_argmaxs": 1,
                          "regr_c012s": 3, "regr_weights": n_classes}[self.configuration]
        self.n_layers = n_layers
        self.n_units_per_layer = n_units_per_layer
        self.input_noise_rate = input_noise_rate
        self.dropout_rate = dropout_rate
        self.opt = opt

        # Computational layers
        if self.input_noise_rate > 0:
            self.input_noise_layer = keras.layers.GaussianNoise(self.input_noise_rate)
        self.dense_layers, self.batch_norm_layers = [], []
        self.activation_layers, self.dropout_layers = [], []
        for i in range(self.n_layers):
            self.dense_layers.append(keras.layers.Dense(
                units=self.n_units_per_layer, name=f"dense_{i}", use_bias=False))
            self.batch_norm_layers.append(keras.layers.BatchNormalization(name=f"batch_norm_{i}"))
            self.activation_layers.append(keras.layers.ReLU(name=f"relu_{i}"))
            self.dropout_layers.append(keras.layers.Dropout(rate=self.dropout_rate))
        self.linear_layer = keras.layers.Dense(units=self.n_classes, use_bias=False, name="linear")
        if self.configuration in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
            self.softmax_layer = keras.layers.Softmax()

    def call(self, x):
        """ Pass tensors forward """
        input = x
        if self.input_noise_rate > 0:
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

    def compile(self, optimizer, loss_fn, metrics=None):
        super().compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model_optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metrics = metrics

    def build(self, input_shapes=None):
        """ Build the model """
        self.call(keras.layers.Input(shape=(self.n_features,)))
        
    def train(self, data, n_epochs, batch_size, delta_max_tolerance, output_location,
              continue_training=False):
        """ Train the model by calling keras.Model.fit() """

        # Preparing the data generator (training and validation)
        train_data_generator = DataGenerator(
            batch_size=batch_size, dataset=data.train, configuration=self.configuration)
        
        # Preparing the callback reponsible for monitoring the model performance
        output_location = os.path.join("results", self.configuration, output_location)
        last_n_epochs = 0
        if (continue_training):
            with open(os.path.join(output_location, "history.json"), "r") as file:
                last_n_epochs = len(json.load(file)["training_epoch_relative_loss"])
        n_epochs = n_epochs + last_n_epochs
        monitoring_callback = MonitoringUtils(data.train, data.valid, batch_size, 
                                              n_epochs, delta_max_tolerance, output_location)
        
        # Preparing the callback for saving checkpoints (weights)
        output_location = os.path.join(
            output_location, os.path.normpath("model_state/model_epoch_{epoch:02d}.weights.h5"))
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=output_location,
            save_weights_only=True)

        # Training the model
        self.fit(train_data_generator, epochs=n_epochs, verbose=0,
                 callbacks=[monitoring_callback, cp_callback], initial_epoch=last_n_epochs)


def save_configuration(location, args):
    """ Save the full set of command line arguments """
    if not os.path.exists(location):
            os.makedirs(location)
    with open(os.path.join(location, "configuration.json"), "w") as file:
        json.dump(args.__dict__, file, indent=2) 


def get_predictions_and_labels(model, dataset, training_method, filtered=False):
    if filtered:
        dataset.x = dataset.x[dataset.filt == 1]

    preds = model.predict(dataset.x)

    if training_method == "soft_weights":
        if filtered:
            dataset.weights = dataset.weights[dataset.filt == 1]
        calc = dataset.weights / tf.tile(
            tf.reshape(tf.reduce_sum(dataset.weights, axis=1), (-1, 1)), 
            (1, dataset.weights.shape[-1]))
        
    if training_method == "soft_c012s":
        if filtered:
            dataset.hits_c012s = dataset.hits_c012s[dataset.filt == 1]
        calc = dataset.hits_c012s / tf.tile(
            tf.reshape(tf.reduce_sum(dataset.hits_c012s, axis=1), 
                        (-1, 1)), (1, dataset.hits_c012s.shape[-1]))
    
    if training_method == "soft_argmaxs":
        if filtered:
            dataset.hits_argmaxs = dataset.hits_argmaxs[dataset.filt == 1]
        calc = dataset.hits_argmaxs / tf.tile(
            tf.reshape(tf.reduce_sum(dataset.hits_argmaxs, axis=1), 
                       (-1, 1)), (1, dataset.hits_argmaxs.shape[-1]))
        calc = calc[:, :-1]    
    
    if training_method == "regr_argmaxs":
        if filtered:
            dataset.argmaxs = dataset.argmaxs[dataset.filt == 1]
        calc = dataset.argmaxs
    
    if training_method == "regr_c012s":
        if filtered:
            dataset.c012s = dataset.c012s[dataset.filt == 1]
        calc = dataset.c012s
    
    if training_method == "regr_weights":
        if filtered:
            dataset.weights = dataset.weights[dataset.filt == 1]
        calc = dataset.weights

    return preds, calc


def save_file(filepath, data, message):
    with open(filepath, 'wb') as f:
        np.save(f, data)
    print(message)
        

def run(args):
    # Loading data
    data_points_path = os.path.join(
        args.IN, 
        f"event_datasets_{args.NUM_CLASSES}_{args.HITS_C012s}_{args.FEAT}.obj")
    with open(data_points_path, 'rb') as f:
        data_points = pickle.load(f)
    n_features = data_points.train.x.shape[1]
    print(f"Loaded data: {n_features} features have been prepared.")
    
    # Creating a new model instance
    model = NeuralNetwork(
        configuration=args.TRAINING_METHOD, 
        n_features=n_features, 
        n_classes=int(args.NUM_CLASSES),
        n_layers=int(args.LAYERS), 
        n_units_per_layer=int(args.SIZE),
        input_noise_rate=0.0,
        dropout_rate=float(args.DROPOUT),
        opt=args.OPT)
    model.build()

    # Configuring the optimizer and loss function
    opt = {
        "GradientDescentOptimizer": keras.optimizers.SGD, 
        "AdadeltaOptimizer": keras.optimizers.Adadelta, 
        "AdagradOptimizer": keras.optimizers.Adagrad,
        "ProximalAdagradOptimizer": tf.compat.v1.train.ProximalAdagradOptimizer, 
        "AdamOptimizer": keras.optimizers.Adam,
        "FtrlOptimizer": keras.optimizers.Ftrl,
        "RMSPropOptimizer": keras.optimizers.RMSprop,
        "ProximalGradientDescentOptimizer": tf.compat.v1.train.ProximalGradientDescentOptimizer
    }[args.OPT](learning_rate=float(args.LEARNING_RATE))

    if args.TRAINING_METHOD in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
        loss = keras.losses.CategoricalCrossentropy(from_logits=False)
    elif args.TRAINING_METHOD  in ["regr_c012s", "regr_weights"]:
        loss = keras.losses.MeanSquaredError()
    elif args.TRAINING_METHOD  == "regr_argmaxs":
        loss = regr_argmaxs_loss
    else:
        raise ValueError(f"Unknown training method has been provided: {args.TRAINING_METHOD}")

    # Compiling the model (loss, optimizer)
    model.compile(optimizer=opt, loss_fn=loss)

    # Running the action (training, training continuation, predicting)
    action = args.ACTION
    model_location = os.path.join("results", args.TRAINING_METHOD, args.MODEL_LOCATION)
    
    if action in ["train", "continue_training"]:
        # Saving command line arguments
        save_configuration(model_location, args)

        continue_training = False
        if action == "continue_training":
            # Loading the model weights
            model.load_weights(os.path.join(model_location, os.path.normpath("model_state/model.weights.h5")))
            continue_training = True

        # Training the model (checkpoints with weights are saved at the end of each epoch)
        model.train(
            data=data_points, 
            n_epochs=int(args.EPOCHS),
            batch_size=128,     
            delta_max_tolerance=int(args.DELT_CLASSES),
            output_location=args.MODEL_LOCATION,
            continue_training=continue_training
        )

    if action in ["predict_train_and_valid", "predict_test"]:
        # Loading the model weights
        model.load_weights(os.path.join(model_location, os.path.normpath("model_state/model.weights.h5")))
        pred_path = os.path.join(model_location, "predictions")
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

    if action == "predict_train_and_valid":
        print("Making predictions for the training and validation sets...")
        train_preds, train_calc = get_predictions_and_labels(
            model, data_points.train, args.TRAINING_METHOD, args.USE_FILTERED_DATA)
        valid_preds, valid_calc = get_predictions_and_labels(
            model, data_points.valid, args.TRAINING_METHOD, args.USE_FILTERED_DATA)
        filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
        train_preds_path = os.path.join(pred_path, f"{filtered}_train_preds.npy")
        save_file(train_preds_path, train_preds,
                  f"Predictions for training data have been saved in {train_preds_path}")
        train_calc_path = os.path.join(pred_path, f"{filtered}_train_calc.npy")
        save_file(train_calc_path, train_calc,
                  f"True values for training data have been saved in {train_calc_path}")
        valid_preds_path = os.path.join(pred_path, f"{filtered}_valid_preds.npy")
        save_file(valid_preds_path, valid_preds,
                  f"Predictions for validation data have been saved in {valid_preds_path}")
        valid_calc_path = os.path.join(pred_path, f"{filtered}_valid_calc.npy")
        save_file(valid_calc_path, valid_calc,
                  f"True values for validation data have been saved in {valid_calc_path}")

    if action == "predict_test":
        print("Making final predictions for the testing data set...")
        test_preds, test_calc = get_predictions_and_labels(
            model, data_points.test, args.TRAINING_METHOD, args.USE_FILTERED_DATA)
        filtered = "unfiltered" if not args.USE_FILTERED_DATA else "filtered"
        test_preds_path = os.path.join(pred_path, f"{filtered}_test_preds.npy")
        save_file(test_preds_path, test_preds,
                  f"Predictions for testing data have been saved in {test_preds_path}")
        test_calc_path = os.path.join(pred_path, f"{filtered}_test_calc.npy")
        save_file(test_calc_path, test_calc,
                  f"True values for testing data have been saved in {test_calc_path}")