import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import sys

def train(model, dataset, batch_size=128):
    sess = tf.get_default_session()
    epoch_size = dataset.n / batch_size
    losses = []

    for i in range(epoch_size):
        x, wa, wb, filt = dataset.next_batch(batch_size)
        loss, _ = sess.run([model.loss, model.train_op],
                           {model.x: x, model.wa: wa, model.wb: wb})
        losses.append(loss)
        if i % (epoch_size / 10) == 5:
          sys.stdout.write(". %.3f " % np.mean(losses))
          sys.stdout.flush()
    return np.mean(losses)

def get_loss(model, dataset, batch_size=128):
    epoch_size = dataset.n / batch_size
    losses = []
    sess = tf.get_default_session()
    for i in range(epoch_size):
        x, wa, wb, filt = dataset.next_batch(batch_size)
        loss = sess.run([model.loss],
                           {model.x: x, model.wa: wa, model.wb: wb})
        losses.append(loss)
    return np.mean(losses)


def total_train(pathOUT, model, data, emodel=None, batch_size=128, epochs=25, metric = "roc_auc"):
    if emodel is None:
        emodel = model
        
    train_aucs = []
    valid_aucs = []
    test_aucs  = []

    train_losses = []
    valid_losses = []
    test_losses  = []

    max_perf = evaluate2(emodel, data.valid, filtered=True, metric = metric)
    print  "max perf: ", max_perf
    
    test_auc = 0
    for i in range(epochs):
        sys.stdout.write("EPOCH: %d " % (i + 1))
        
        _ = train(model, data.train, batch_size)

        train_loss = get_loss(model, data.train, batch_size=128)
        valid_loss = get_loss(model, data.valid, batch_size=128)
        test_loss  = get_loss(model, data.test, batch_size=128)

        train_losses += [train_loss]
        valid_losses += [valid_loss]
        test_losses  += [test_loss]
        
        train_auc = evaluate(emodel, data.train, 100000, filtered=True, metric = metric)
        valid_auc = evaluate(emodel, data.valid, filtered=True, metric = metric)
        msg_str = "TRAIN LOSS: %.3f TRAIN AUC: %.3f VALID AUC: %.3f" % (train_loss, train_auc, valid_auc)
        print msg_str
        tf.logging.info(msg_str)
        
        train_aucs += [train_auc]
        valid_aucs += [valid_auc]
        if valid_auc == np.max(valid_aucs):
            test_auc = evaluate(emodel, data.test, filtered=True, metric = metric)
            test_aucs += [test_auc]

            x, pred_wa, true_wa, true_wb  = predictions(emodel, data.test, filtered=True)

            pred_wb = 1 -pred_wa
            np.save(pathOUT+'event_colls.npy', x)
            np.save(pathOUT+'event_pred_wa.npy', pred_wa)
            np.save(pathOUT+'event_pred_wb.npy', pred_wb)
            np.save(pathOUT+'event_true_wa.npy', true_wa)
            np.save(pathOUT+'event_true_wb.npy', true_wb)

            roc_curve_data = get_roc(emodel, data.test, filtered=True)

            np.save(pathOUT+'roc_curve_data.npy', roc_curve_data)

            
    print "TEST AUC: ", test_auc
    
    # storing history of training

    np.save(pathOUT+'train_losses.npy', train_losses)
    np.save(pathOUT+'valid_losses.npy', valid_losses)
    np.save(pathOUT+'test_losses.npy',  test_losses)
    
    np.save(pathOUT+'train_aucs.npy', train_aucs)
    np.save(pathOUT+'valid_aucs.npy', valid_aucs)
    np.save(pathOUT+'test_aucs.npy', test_aucs)

    return train_aucs, valid_aucs, test_aucs


def get_roc(model, dataset, at_most=None, filtered=False):
    _, ps, was, wbs = predictions(model, dataset, at_most, filtered)

    n = len(ps)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([ps, ps])
    weights = np.concatenate([was, wbs])

    return roc_curve(true_labels, preds, sample_weight=weights)


def predictions(model, dataset, at_most=None, filtered=False):
    sess = tf.get_default_session()
    x = dataset.x
    wa = dataset.wa
    wb = dataset.wb
    filt = dataset.filt

    if at_most is not None:
      filt = filt[:at_most]
      x = x[:at_most]
      wa = wa[:at_most]
      wb = wb[:at_most]

    p = sess.run(model.p, {model.x: x})


    if filtered:
      p = p[filt == 1]
      x = x[filt == 1]
      wa = wa[filt == 1]
      wb = wb[filt == 1]

    return x, p, wa, wb


def evaluate(model, dataset, at_most=None, filtered=False, metric = "roc_auc"):
    _, ps, was, wbs = predictions(model, dataset, at_most, filtered)

    return evaluate_preds(ps, was, wbs, metric = metric)    
    # replace by line below to get max sensitivity
    #return evaluate_preds(was/(was+wbs), was, wbs)

def evaluate2(model, dataset, at_most=None, filtered=False, metric = "roc_auc"):
    _, ps, was, wbs = predictions(model, dataset, at_most, filtered)
    return evaluate_preds(was/(was+wbs), was, wbs, metric = metric)


def evaluate_preds(preds, wa, wb, metric = "roc_auc"):
    n = len(preds)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds, preds])
    weights = np.concatenate([wa, wb])
    
    if metric == "roc_auc":
    	return roc_auc_score(true_labels, preds, sample_weight=weights)
    elif metric == "prec_score":
    	return average_precision_score(true_labels, preds, sample_weight=weights)

def linear(x, name, size, bias=True):
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [1, size],
                        initializer=tf.zeros_initializer())
    return tf.matmul(x, w)  + b #vanishes in batch normalization?


def batch_norm(x, name):
    mean, var = tf.nn.moments(x, [0])
    normalized_x = (x - mean) * tf.rsqrt(var + 1e-8)
    gamma = tf.get_variable(name + "/gamma", [x.get_shape()[-1]], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable(name + "/beta", [x.get_shape()[-1]])
    return gamma * normalized_x + beta


class NeuralNetwork(object):

    def __init__(self, num_features, num_layers=1, size=100, lr=1e-3, keep_prob=1.0,
                 tloss="soft", input_noise=0.0, optimizer="AdamOptimizer"):
        batch_size = None
        self.x = x = tf.placeholder(tf.float32, [batch_size, num_features])
        self.wa = wa = tf.placeholder(tf.float32, [batch_size])
        self.wb = wb = tf.placeholder(tf.float32, [batch_size])

        if input_noise > 0.0:
          x = x * tf.random_normal(tf.shape(x), 1.0, input_noise)

        for i in range(num_layers):
            x = tf.nn.relu(linear(batch_norm(x, "bn_%d" % i), "linear_%d" % i, size)) 
            if keep_prob < 1.0:
              x = tf.nn.dropout(x, keep_prob)

        if tloss == "log":
            x = linear(x, "regression", 1)
            self.p = tf.nn.sigmoid(x)
            y = wa / (wa + wb)
            self.loss = loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=x, labels=tf.reshape(y, [-1, 1])))
        elif tloss == "soft":
            sx = linear(x, "regression", 3)
            self.preds = preds = tf.nn.softmax(sx)
            self.p = preds[:, 0] / (preds[:, 0] + preds[:, 1])

            wa = tf.reshape(wa, [-1, 1])
            wb = tf.reshape(wb, [-1, 1])
            # wa = p_a / p_c
            # wb = p_b / p_c
            # wa + wb + 1 = (p_a + p_b + p_c) / p_c
            # wa / (wa + wb + 1) = p_a / (p_a + p_b + p_c)
            labels = tf.concat([wa, wb, tf.ones_like(wa)], axis=1) / (wa + wb + 1) # + 1 should be here

            self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
        else:
            raise ValueError("tloss unrecognized: %s" % tloss)

        optimizer = {"GradientDescentOptimizer": tf.train.GradientDescentOptimizer, 
        "AdadeltaOptimizer": tf.train.AdadeltaOptimizer, "AdagradOptimizer": tf.train.AdagradOptimizer,
        "ProximalAdagradOptimizer": tf.train.ProximalAdagradOptimizer, "AdamOptimizer": tf.train.AdamOptimizer,
        "FtrlOptimizer": tf.train.FtrlOptimizer, "RMSPropOptimizer": tf.train.RMSPropOptimizer,
        "ProximalGradientDescentOptimizer": tf.train.ProximalGradientDescentOptimizer}[optimizer]
        self.train_op = optimizer(lr).minimize(loss)


