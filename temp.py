from utilities.data_utils import read_np
import os, pickle, numpy as np
from utilities.tf_model import NeuralNetwork, regr_argmaxs_loss
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

args_IN = "data"
args_NUM_CLASSES = 21
args_HYPOTHESIS = 19
args_HITS_C012s = "hits_c0s"
args_FEAT = "Variant-All"
args_TRAINING_METHOD = "soft_weights"
n_features = 24
args_LAYERS = 6
args_SIZE = 100
args_DROPOUT = 0.0
args_OPT = "AdamOptimizer"

# Loading data points
data_points_path = os.path.join(
        args_IN, 
        f"event_datasets_{args_NUM_CLASSES}_{args_HITS_C012s}_{args_FEAT}.obj")
with open(data_points_path, 'rb') as f:
    data_points = pickle.load(f)
W = data_points.train.weights
weights_normalised = W / 2
data_len = len(weights_normalised)
unweighted_weights = []
monte_carlo = lambda x : 0.0 if x < np.random.random() else 1.0
unweighted_weights = np.vectorize(monte_carlo)(weights_normalised)
X = data_points.train.x
print("Mean X received from event dataset: ", np.mean(X))
print(X[0:5])
print(W[0:5])
print(unweighted_weights[0:5])

# Loading unweighted weights
# unweighted_weights_path = os.path.join(
#     args_IN, 
#     f"unwt_multiclass_{args_NUM_CLASSES}.npy") 
# unweighted_weights_path = os.path.join(
#     args_IN, 
#     f"unwt_multiclass_{args_NUM_CLASSES}.npy") 
# unweighted_weights = read_np(unweighted_weights_path)
hyp = args_HYPOTHESIS
unweighted_weights = unweighted_weights[:, hyp]
print("UNWEIGHTED WEIGHTS:")
print(unweighted_weights.shape)
print(unweighted_weights[0:5])

# Loading weights
# W = read_np(os.path.join(args_IN, f"weights_multiclass_{args_NUM_CLASSES}.npy"))
W = W[unweighted_weights == 1.0, :]
W = W / tf.tile(tf.reshape(tf.reduce_sum(W, axis=1), (-1, 1)), (1, W.shape[-1]))
print("WEIGHTS")
print(W.shape)
print(W[0:5])
print("W argmax argmax examples:", np.argmax(W, axis=1)[0:20])
print("W argmax sum:", np.argmax(np.sum(W, axis=0)))

# Loading features
X_path = os.path.join(args_IN, f"rhorho_event_{args_FEAT}.obj")
with open(X_path, 'rb') as f:
    X = pickle.load(f)
X = X.cols[:, :-1]
X = X[:-200000]
X = X[unweighted_weights == 1.0]
print("FEATURES")

print("Mean X received from rhorho_event: ", np.mean(X))
print(type(X))
print(len(X))
print(X.shape)
print(X[0:5])

A = np.copy(W)
plt.plot(np.linspace(0, 20, num=21), np.sum(A, axis=0))
plt.show()
plt.clf()
hyp_w = W[:, hyp]
A = A * hyp_w[:, np.newaxis]
plt.plot(np.linspace(0, 20, num=21), np.sum(A, axis=0))
plt.show()
plt.clf()

# Making predictions on the features
model = NeuralNetwork(
    configuration=args_TRAINING_METHOD, 
    n_features=n_features, 
    n_classes=int(args_NUM_CLASSES),
    n_layers=int(args_LAYERS), 
    n_units_per_layer=int(args_SIZE),
    input_noise_rate=0.0,
    dropout_rate=float(args_DROPOUT),
    opt=args_OPT)
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
}[args_OPT](learning_rate=0.001)

if args_TRAINING_METHOD in ["soft_weights", "soft_argmaxs", "soft_c012s"]:
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)
elif args_TRAINING_METHOD  in ["regr_c012s", "regr_weights"]:
    loss = keras.losses.MeanSquaredError()
elif args_TRAINING_METHOD  == "regr_argmaxs":
    loss = regr_argmaxs_loss
else:
    raise ValueError(f"Unknown training method has been provided: {args_TRAINING_METHOD}")

# Compiling the model (loss, optimizer)
model.compile(optimizer=opt, loss_fn=loss)
model.load_weights(
    "results/soft_weights/21_classes_variant_all/model_state/model.weights.h5")

preds = model.predict(X)
argmax = np.argmax(np.sum(preds, axis=0))
print("Preds argmax sum:", argmax)

A = np.copy(preds)
plt.plot(np.linspace(0, 20, num=21), np.sum(A, axis=0))
plt.show()
plt.clf()
hyp_w = preds[:, argmax]
A = A * hyp_w[:, np.newaxis]
plt.plot(np.linspace(0, 20, num=21), np.sum(A, axis=0))
plt.show()
plt.clf()

plt.plot(np.linspace(0, 20, num=21), A[0], label="A[0]")
plt.plot(np.linspace(0, 20, num=21), A[1], label="A[1]")
plt.plot(np.linspace(0, 20, num=21), A[2], label="A[2]")
plt.plot(np.linspace(0, 20, num=21), A[3], label="A[3]")
plt.plot(np.linspace(0, 20, num=21), A[4], label="A[4]")
plt.plot(np.linspace(0, 20, num=21), A[5], label="A[5]")
plt.show()
plt.clf()

