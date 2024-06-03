""" Target: regr_c012s, Variant-4.1 and Variant-1.0 """

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
feature_config = "Variant-1.0"
n_classes = 51
filtered = True
miniset = False

# Hyperparameters
batch_size = 128
learning_rate = 0.001
n_epochs = 25

# Loading data
dataset = os.path.join(
    datapath, f"event_datasets_{n_classes}_hits_c0s_{feature_config}.obj")
with open(dataset, 'rb') as f:
    dataset = pickle.load(f)
features = dataset.train.x
print(f"Loaded data: {features.shape[1]} features have been prepared.")

# Model definition
input = tf.keras.Input(shape=(features.shape[-1],), name="input")
x = input

x = tf.keras.layers.Dense(units=1024)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=512)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=256)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.ReLU()(x)

x = tf.keras.layers.Dense(units=128)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.ReLU()(x)

output = tf.keras.layers.Dense(units=3, activation="sigmoid")(x)
model = tf.keras.Model(inputs=input, outputs=output)
# tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
model.summary()

def loss(y_true, y_pred):
    c0s, c1s, c2s = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    # Sigmoid -> C0/C1/C2
    c0s, c1s, c2s = c0s * 2, c1s * 2 - 1, c2s * 2 - 1
    loss = tf.reduce_mean(
        tf.square(y_true[:, 0] - c0s) + \
        tf.square(y_true[:, 1] - c1s) + \
        tf.square(y_true[:, 2] - c2s)) / 3
    return loss

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=loss)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset

    def __len__(self):
        return math.ceil(self.dataset.n / self.batch_size)

    def __getitem__(self, idx):
        x, _, _, c012s, _, _, _ = self.dataset.next_batch(self.batch_size)
        x, _, _, c012s, _, _, _ = self.dataset.next_batch(self.batch_size)
        x, _, _, c012s, _, _, _ = self.dataset.next_batch(self.batch_size)
        return x, c012s

# Training
if miniset:
    dataset.train.x = dataset.train.x[0:100000]
    dataset.train.c012s = dataset.train.c012s[0:100000]
    dataset.train.n = 100000
    
training_generator = DataGenerator(batch_size, dataset.train)
validation_generator = DataGenerator(batch_size, dataset.valid)

# Training and saving weights
model_weights_outpath = "history/model_regr_c012s"
if not os.path.exists(model_weights_outpath):
    os.makedirs(model_weights_outpath)
model_weights_outpath = os.path.join(
    model_weights_outpath, os.path.normpath("model_epoch_{epoch:02d}.weights.h5"))
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_weights_outpath,
    save_weights_only=True)

model.fit(training_generator, epochs=n_epochs, validation_data=validation_generator)

# Evaluation on validation data
if filtered:
    preds = model.predict(dataset.valid.x[dataset.valid.filt == 1])
else:
    preds = model.predict(dataset.valid.x)

# Sigmoid -> C0/C1/C2
preds[:, 0] = preds[:, 0] * 2
preds[:, 1] = preds[:, 1] * 2 - 1
preds[:, 2] = preds[:, 2] * 2 - 1

calc = dataset.valid.c012s
if filtered:
    calc = calc[dataset.valid.filt == 1]

# Preparing the plot
def calc_weights(num_classes, coeffs):
    k2PI = 2 * np.pi
    x = np.linspace(0, k2PI, num_classes)
    data_len = coeffs.shape[0]
    weights = np.zeros((data_len, num_classes))
    for i in range(data_len):
        weights[i] = weight_fun(x, *coeffs[i])
    return weights

k2PI = 2 * np.pi
calc_w  =  calc_weights(n_classes, calc)
preds_w =  calc_weights(n_classes, preds)

delt_argmax = calculate_deltas_signed(
    np.argmax(preds_w[:], axis=1), np.argmax(calc_w[:], axis=1), n_classes)      
delt_rad = delt_argmax * k2PI / (n_classes - 1)
meanrad = np.mean(delt_rad)
stdrad  = np.std(delt_rad)
meanraderr = stats.sem(delt_rad)

plt.hist(delt_rad, histtype='step', bins=n_classes, color='black')
plt.xlabel(r'$\Delta\alpha^{CP}_{max}$ [rad]')
plt.gca()

table_vals=[
    [r'Classification: $C_0, C_1, C_2$'],
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
plt.savefig(os.path.join(output_path, "model_regr_c012s.pdf"))

plt.clf()