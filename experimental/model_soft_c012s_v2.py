""" Target: soft_c012s, Variant-All """

import tensorflow as tf
import os, pickle, math
import matplotlib.pyplot as plt
import numpy as np
from utilities.cpmix_utils import weight_fun
from scipy import stats
from utilities.metrics_utils import calculate_deltas_signed
from utilities.tf_model import DataGenerator

# Configuration settings
datapath = "data"
feature_config = "Variant-All"
n_classes = 21
filtered = True
miniset = False

# Hyperparameters
batch_size = 128
learning_rate = 0.001
n_epochs = 15

# Loading data
dataset_c0_path = os.path.join(
    datapath, f"event_datasets_{n_classes}_hits_c0s_{feature_config}.obj")
with open(dataset_c0_path, 'rb') as f:
    dataset_c0 = pickle.load(f)
features = dataset_c0.train.x
print(f"Loaded data: {features.shape[1]} features have been prepared.")

dataset_c1_path = os.path.join(
    datapath, f"event_datasets_{n_classes}_hits_c1s_{feature_config}.obj")
with open(dataset_c1_path, 'rb') as f:
    dataset_c1 = pickle.load(f)

dataset_c2_path = os.path.join(
    datapath, f"event_datasets_{n_classes}_hits_c2s_{feature_config}.obj")
with open(dataset_c2_path, 'rb') as f:
    dataset_c2 = pickle.load(f)

# Model definition
input = tf.keras.Input(shape=(features.shape[-1],), name="input")
x = input

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=100, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

c0_output = tf.keras.layers.Dense(units=n_classes, use_bias=False, name="linear_c0")(x)
c0_output = tf.keras.layers.Softmax(name="output_c0")(c0_output)

c1_output = tf.keras.layers.Dense(units=n_classes, use_bias=False, name="linear_c1")(x)
c1_output = tf.keras.layers.Softmax(name="output_c1")(c1_output)

c2_output = tf.keras.layers.Dense(units=n_classes, use_bias=False, name="linear_c2")(x)
c2_output = tf.keras.layers.Softmax(name="output_c2")(c2_output)

model = tf.keras.Model(inputs=[input], outputs=[c0_output, c1_output, c2_output])
# tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
# model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss={
        "output_c0": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "output_c1": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        "output_c2": tf.keras.losses.CategoricalCrossentropy(from_logits=False)    
    },
    loss_weights={"output_c0": 1/3, "output_c1": 1/3, "output_c2": 1/3}
)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset_c0, dataset_c1, dataset_c2):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_c0, self.dataset_c1, self.dataset_c2 = \
            dataset_c0, dataset_c1, dataset_c2

    def __len__(self):
        return math.ceil(self.dataset_c0.n / self.batch_size)

    def __getitem__(self, idx):
        x, _, _, _, _, hits_c0s, _ = self.dataset_c0.next_batch(self.batch_size)
        _, _, _, _, _, hits_c1s, _ = self.dataset_c1.next_batch(self.batch_size)
        _, _, _, _, _, hits_c2s, _ = self.dataset_c2.next_batch(self.batch_size)
        hits_c0s = hits_c0s / tf.tile(
            tf.reshape(tf.reduce_sum(
                hits_c0s, axis=1), (-1, 1)), (1, hits_c0s.shape[-1]))
        hits_c1s = hits_c1s / tf.tile(
            tf.reshape(tf.reduce_sum(
                hits_c1s, axis=1), (-1, 1)), (1, hits_c1s.shape[-1]))
        hits_c2s = hits_c2s / tf.tile(
            tf.reshape(tf.reduce_sum(
                hits_c2s, axis=1), (-1, 1)), (1, hits_c2s.shape[-1]))
        return (x), (hits_c0s, hits_c1s, hits_c2s)

# Training
if miniset:
    dataset_c0.train.x = dataset_c0.train.x[0:100000]
    dataset_c0.train.hits_c012s = dataset_c0.train.hits_c012s[0:100000]
    dataset_c0.train.n = 100000
    dataset_c1.train.x = dataset_c1.train.x[0:100000]
    dataset_c1.train.hits_c012s = dataset_c1.train.hits_c012s[0:100000]
    dataset_c1.train.n = 100000
    dataset_c2.train.x = dataset_c2.train.x[0:100000]
    dataset_c2.train.hits_c012s = dataset_c2.train.hits_c012s[0:100000]
    dataset_c2.train.n = 100000
    
training_generator = DataGenerator(
    batch_size, dataset_c0.train, dataset_c1.train, dataset_c2.train)

validation_generator = DataGenerator(
    batch_size, dataset_c0.valid, dataset_c1.valid, dataset_c2.valid)

# Training
model.fit(
    training_generator,
    epochs=n_epochs,
    validation_data=validation_generator
)

# Evaluation on validation data
if filtered:
    preds = model.predict(dataset_c0.valid.x[dataset_c0.valid.filt == 1])
else:
    preds = model.predict(dataset_c0.valid.x)

calc_hits_c0s = dataset_c0.valid.hits_c012s
if filtered:
    calc_hits_c0s = calc_hits_c0s[dataset_c0.valid.filt == 1]
calc_hits_c0s = calc_hits_c0s / tf.tile(
    tf.reshape(tf.reduce_sum(calc_hits_c0s, axis=1), (-1, 1)), (1, calc_hits_c0s.shape[-1]))

calc_hits_c1s = dataset_c1.valid.hits_c012s
if filtered:
    calc_hits_c1s = calc_hits_c1s[dataset_c1.valid.filt == 1]
calc_hits_c1s = calc_hits_c1s / tf.tile(
    tf.reshape(tf.reduce_sum(calc_hits_c1s, axis=1), (-1, 1)), (1, calc_hits_c1s.shape[-1]))

calc_hits_c2s = dataset_c2.valid.hits_c012s
if filtered:
    calc_hits_c2s = calc_hits_c2s[dataset_c2.valid.filt == 1]
calc_hits_c2s = calc_hits_c2s / tf.tile(
    tf.reshape(tf.reduce_sum(calc_hits_c2s, axis=1), (-1, 1)), (1, calc_hits_c2s.shape[-1]))

preds_hits_c0s = preds[0]
preds_hits_c1s = preds[1]
preds_hits_c2s = preds[2]

# Preparing the plot
def calc_weights(num_classes, coeffs):
    k2PI = 2 * np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights

data_len = calc_hits_c0s.shape[0]
preds_c0s = np.argmax(preds_hits_c0s, axis=1)
calc_c0s = np.argmax(calc_hits_c0s, axis=1)
preds_c1s = np.argmax(preds_hits_c1s, axis=1)
calc_c1s = np.argmax(calc_hits_c1s, axis=1)
preds_c2s = np.argmax(preds_hits_c2s, axis=1)
calc_c2s = np.argmax(calc_hits_c2s, axis=1)    

calc_c012s = np.zeros((data_len, 3))
preds_c012s = np.zeros((data_len, 3))

calc_c012s[:, 0] = calc_c0s * (2./(n_classes - 1))
calc_c012s[:, 1] = calc_c1s * (2./(n_classes - 1)) - 1.0
calc_c012s[:, 2] = calc_c2s * (2./(n_classes - 1)) - 1.0
preds_c012s[:, 0] = preds_c0s * (2./(n_classes - 1))
preds_c012s[:, 1] = preds_c1s * (2./(n_classes - 1)) - 1.0
preds_c012s[:, 2] = preds_c2s * (2./(n_classes - 1)) - 1.0

k2PI = 2 * np.pi
calc_w  =  calc_weights(n_classes, calc_c012s)
preds_w =  calc_weights(n_classes, preds_c012s)
delt_argmax = calculate_deltas_signed(
    np.argmax(preds_w[:], axis=1), np.argmax(calc_w[:], axis=1), n_classes)      

mean = np.mean(delt_argmax) 
std  = np.std(delt_argmax) 
meanerr = stats.sem(delt_argmax)
meanrad = np.mean(delt_argmax) * k2PI/n_classes
stdrad  = np.std(delt_argmax) * k2PI/n_classes
meanraderr = stats.sem(delt_argmax) * k2PI/n_classes

plt.hist(delt_argmax, histtype='step', bins=n_classes, color='black')
plt.xlabel(r'$\Delta_{class} [idx]$')
plt.gca()

table_vals=[
    [r'Classification: $C_0, C_1, C_2$'],
    [" "],
    [r"mean = {:0.3f}$\pm$ {:1.3f} [idx]".format(mean, meanerr)],
    ["std = {:1.3f} [idx]".format(std)],
    [" "],
    [r"mean = {:0.3f}$\pm$ {:1.3f} [rad]".format(meanrad, meanraderr)],
    ["std = {:1.3f} [rad]".format(stdrad)]]

table = plt.table(
    cellText=table_vals,
    colWidths = [0.40],
    cellLoc="left",
    loc='upper right')
table.set_fontsize(14)

for _, cell in table.get_celld().items():
    cell.set_linewidth(0)

plt.tight_layout()

output_path = "experimental/figures"
if not os.path.exists(output_path):
        os.makedirs(output_path)
plt.savefig(os.path.join(output_path, "model_c012s_v2.pdf"))

plt.clf()