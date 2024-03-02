""" This program initialises and trains a neural network model using TensorFlow 
for solving the problems related to the rhorho events analysis """
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_auc_score
import sys
from math import floor
from .metrics_utils import calculate_deltas_unsigned, calculate_deltas_signed

# TODO (comment, probably from ERW): 
# Issue with regr_c_coefficients, not learning well after the modifications 
# of the last weeks; the last good production was on June 20th
def train(model, dataset, batch_size=128):
    """ Implement a training step by invoking the sess.run() on a model.train_operation.
    NOTICE: "epoch_size" is not the same as "epochs". It shows us the number of steps required 
    to process the whole data volume (and thus, not the number of epochs) taken into account 
    the "batch_size" defining the way the data itself is supplied in little portions (mini-batches). """
    sess = tf.get_default_session()
    epoch_size = floor(dataset.n / batch_size)
    losses = []

    sys.stdout.write("<losses>):")
    for i in range(epoch_size):
        x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s, filt,  = dataset.next_batch(batch_size)
        loss, _ = sess.run([model.loss, model.train_op],
                           {model.x: x, model.weights: weights, model.argmaxs: argmaxs, model.c012s: c012s,
                            model.hits_argmaxs: hits_argmaxs, model.hits_c012s: hits_c012s})
        losses.append(loss)
        if i % (epoch_size / 10) == 5:
          sys.stdout.write(". %.3f " % np.mean(losses))
          losses =[]
          sys.stdout.flush()
    sys.stdout.write("\n")
    return np.mean(losses)

def get_loss(model, dataset, batch_size=128):
    batches = floor(dataset.n / batch_size)
    losses = []
    sess = tf.get_default_session()
    for i in range(batches):
        x, weights, argmaxs, c012s, hits_argmaxs, hits_c012s, filt, = dataset.next_batch(batch_size)

        # Getting the loss value from the computational graph and providing all the values needed to compute it 
        # (they are all inside the dictionary passed as the second parameter to the sess.run())
        loss = sess.run([model.loss],
                           {model.x: x, model.weights: weights, model.argmaxs: argmaxs, model.c012s: c012s,
                            model.hits_argmaxs: hits_argmaxs, model.hits_c012s: hits_c012s})
        losses.append(loss)
    return np.mean(losses)


# TODO (Find out whether it was a comment or proposition):
# Comment from ERW: 
# tf_model knows nothing about classes <--> angle relations
# operates on arrays which has dimention of num_classes
def total_train(pathOUT, model, data, args, emodel=None, batch_size=128, epochs=25):
    tf.get_default_session()

    if emodel is None:
        emodel = model
    
    train_accs   = []
    valid_accs   = []
    test_accs    = []
    train_losses = []
    valid_losses = []
    test_losses  = []
    train_L1_deltas = []
    train_L2_deltas = []
    valid_L1_deltas = []
    valid_L2_deltas = []
    test_L1_deltas  = []
    test_L2_deltas  = []


    # ERW
    # maximal sensitivity not defined in  multi-class case?
    # please reintroduce this option
    # max_perf = evaluate2(emodel, data.valid, filtered=True)
    # print max_perf

    # Comment from ERW:
    # TODO: isn't the maximal sensitivity defined in multi-class case?
    # Please, reintroduce this option:
    # max_perf = evaluate2(emodel, data.valid, filtered=True)
    # print max_perf
    print ("model = ", model.tloss)

    # Iterating over the epochs
    for i in range(epochs):
        sys.stdout.write("\nEPOCH: %d \n" % (i + 1))

        # The training step itself (train() invokes the loss function minimalisation) 
        _ = train(model, data.train, batch_size)

        train_loss = get_loss(model, data.train, batch_size=128)
        valid_loss = get_loss(model, data.valid, batch_size=128)
        test_loss  = get_loss(model, data.test, batch_size=128)

        train_losses += [train_loss]
        valid_losses += [valid_loss]
        test_losses  += [test_loss]


        print("TRAINING:    LOSS: %.3f \n" % (train_loss))
        print("VALIDATION:  LOSS: %.3f \n" % (valid_loss))
        print("TEST:        LOSS: %.3f \n" % (test_loss))

        
        if model.tloss == 'soft_weights':
            train_acc, train_mean, train_l1_delta_w, train_l2_delta_w = evaluate(emodel, data.train, args, 100000, filtered=True)
            valid_acc, valid_mean, valid_l1_delta_w, valid_l2_delta_w = evaluate(emodel, data.valid, args, filtered=True)
            msg_str_0 = "TRAINING:     loss: %.3f \n" % (train_loss)
            msg_str_1 = "TRAINING:     acc: %.3f mean: %.3f L1_delta_w: %.3f  L2_delta_w: %.3f \n" % (train_acc, train_mean, train_l1_delta_w, train_l2_delta_w)
            msg_str_2 = "VALIDATION:   acc: %.3f mean  %.3f L1_delta_w: %.3f, L2_delta_w: %.3f \n" % (valid_acc, valid_mean, valid_l1_delta_w, valid_l2_delta_w)
            print(msg_str_0)
            print(msg_str_1)
            print(msg_str_2)
            tf.logging.info(msg_str_0)
            tf.logging.info(msg_str_1)
            tf.logging.info(msg_str_2)
            
            train_accs   += [train_acc]
            valid_accs   += [valid_acc]
            
            train_L1_deltas += [train_l1_delta_w]
            train_L2_deltas += [train_l2_delta_w]
            valid_L1_deltas += [valid_l1_delta_w]
            valid_L2_deltas += [valid_l2_delta_w]

            if valid_acc == np.max(valid_accs):
                test_acc, test_mean, test_l1_delta_w, test_l2_delta_w = evaluate(emodel, data.test, args, filtered=True)
                msg_str_3 = "TESTING:      acc: %.3f mean  %.3f L1_delta_w: %.3f, L2_delta_w: %.3f \n" % (test_acc, test_mean, test_l1_delta_w, test_l2_delta_w)
                print(msg_str_3)
                tf.logging.info(msg_str_3)
                
                test_accs += [test_acc]
                test_L1_deltas += [test_l1_delta_w]
                test_L2_deltas += [test_l2_delta_w]

                calc_w, preds_w = softmax_predictions(emodel, data.test, filtered=True)

                # calc_w, preds_w normalisation to probability
                calc_w = calc_w / np.sum(calc_w, axis=1)[:, np.newaxis]
                preds_w = preds_w / np.sum(preds_w, axis=1)[:, np.newaxis]

                # TODO: figure out what was the meaning of the below comment
                # Comment from ERW:
                # control print
                # print("ERW test on softmax: calc_w \n")
                # print(calc_w)
                # print("ERW test on softmax: preds_w \n")
                # print(preds_w)
                np.save(pathOUT+'softmax_calc_w.npy', calc_w)
                np.save(pathOUT+'softmax_preds_w.npy', preds_w)


        if model.tloss == 'soft_c012s':

            train_calc_c012s, train_pred_c012s = soft_c012s_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'train_soft_calc_hits_c012s.npy', train_calc_c012s)
            np.save(pathOUT + 'train_soft_preds_hits_c012s.npy', train_pred_c012s)

            valid_calc_c012s, valid_pred_c012s = soft_c012s_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'valid_soft_calc_hits_c012s.npy', valid_calc_c012s)
            np.save(pathOUT + 'valid_soft_preds_hits_c012s.npy', valid_pred_c012s)

            test_calc_c012s, test_pred_c012s = soft_c012s_predictions(emodel, data.test, filtered=True)
            np.save(pathOUT + 'test_soft_calc_hits_c012s.npy', test_calc_c012s)
            np.save(pathOUT + 'test_soft_preds_hits_c012s.npy', test_pred_c012s)

        if model.tloss == 'regr_c012s':

            train_calc_c012s, train_pred_c012s = regr_c012s_predictions(emodel, data.train, filtered=True)
            np.save(pathOUT + 'train_regr_calc_c012s.npy', train_calc_c012s)
            np.save(pathOUT + 'train_regr_preds_c012s.npy', train_pred_c012s)

            valid_calc_c012s, valid_pred_c012s = regr_c012s_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'valid_regr_calc_c012s.npy', valid_calc_c012s)
            np.save(pathOUT + 'valid_regr_preds_c012s.npy', valid_pred_c012s)

            test_calc_c012s, test_pred_c012s = regr_c012s_predictions(emodel, data.test, filtered=True)
            np.save(pathOUT + 'test_regr_calc_c012s.npy', test_calc_c012s)
            np.save(pathOUT + 'test_regr_preds_c012s.npy', test_pred_c012s)

        if model.tloss == 'regr_argmaxs':

            train_calc_argmaxs, train_pred_argmaxs = regr_argmaxs_predictions(emodel, data.train, filtered=True)
            np.save(pathOUT + 'train_regr_calc_argmaxs.npy', train_calc_argmaxs)
            np.save(pathOUT + 'train_regr_preds_argmaxs.npy', train_pred_argmaxs)

            valid_calc_argmaxs, valid_pred_argmaxs = regr_argmaxs_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'valid_regr_calc_argmaxs.npy', valid_calc_argmaxs)
            np.save(pathOUT + 'valid_regr_preds_argmaxs.npy', valid_pred_argmaxs)

            test_calc_argmaxs, test_pred_argmaxs = regr_argmaxs_predictions(emodel, data.test, filtered=True)
            np.save(pathOUT + 'test_regr_calc_argmaxs.npy', test_calc_argmaxs)
            np.save(pathOUT + 'test_regr_preds_argmaxs.npy', test_pred_argmaxs)

        if model.tloss == 'soft_argmaxs':

            train_calc_argmaxs, train_pred_argmaxs = soft_argmaxs_predictions(emodel, data.train, filtered=True)
            np.save(pathOUT + 'train_soft_calc_hits_argmaxs.npy', train_calc_argmaxs)
            np.save(pathOUT + 'train_soft_preds_hits_argmaxs.npy', train_pred_argmaxs)

            valid_calc_argmaxs, valid_pred_argmaxs = soft_argmaxs_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'valid_soft_calc_hits_argmaxs.npy', valid_calc_argmaxs)
            np.save(pathOUT + 'valid_soft_preds_hits_argmaxs.npy', valid_pred_argmaxs)

            test_calc_argmaxs, test_pred_argmaxs = soft_argmaxs_predictions(emodel, data.test, filtered=True)
            np.save(pathOUT + 'test_soft_calc_hits_argmaxs.npy', test_calc_argmaxs)
            np.save(pathOUT + 'test_soft_preds_hits_argmaxs.npy', test_pred_argmaxs)

        if model.tloss == 'regr_weights':

            train_calc_weights, train_pred_weights = regr_weights_predictions(emodel, data.train, filtered=True)
            np.save(pathOUT + 'train_regr_calc_weights.npy', train_calc_weights)
            np.save(pathOUT + 'train_regr_preds_weights.npy', train_pred_weights)

            valid_calc_weights, valid_pred_weights = regr_weights_predictions(emodel, data.valid, filtered=True)
            np.save(pathOUT + 'valid_regr_calc_weights.npy', valid_calc_weights)
            np.save(pathOUT + 'valid_regr_preds_weights.npy', valid_pred_weights)

            test_calc_weights, test_pred_weights = regr_weights_predictions(emodel, data.test, filtered=True)
            np.save(pathOUT + 'test_regr_calc_weights.npy', test_calc_weights)
            np.save(pathOUT + 'test_regr_preds_weights.npy', test_pred_weights)

    # Saving the results of the training      
    if model.tloss == 'soft_weights':
        """ TODO (Proposition): we do not save the value returned by the function below,
        so I suppose we can safely delete it: """
        test_roc_auc(preds_w, calc_w)             

        # Storing the training history            
        np.save(pathOUT+'train_losses.npy', train_losses)
        print("train_losses", train_losses)
        np.save(pathOUT+'valid_losses.npy', valid_losses)
        print("valid_losses", valid_losses)
        np.save(pathOUT+'test_losses.npy',  test_losses)
        print("tets_losses", test_losses)

        np.save(pathOUT+'train_accs.npy', train_accs)
        print("train_accs", train_accs)
        np.save(pathOUT+'valid_accs.npy', valid_accs)
        print("valid_accs", valid_accs)
        np.save(pathOUT+'test_accs.npy', test_accs)
        print("test_accs", test_accs)

        np.save(pathOUT+'train_L1_deltas.npy', train_L1_deltas)
        print("train_L1_deltas", train_L1_deltas)
        np.save(pathOUT+'valid_L1_deltas.npy', valid_L1_deltas)
        print("valid_L1_deltas", valid_L1_deltas)
        np.save(pathOUT+'test_L1_deltas.npy', test_L1_deltas)
        print("test_L1_deltas", test_L1_deltas)

        np.save(pathOUT+'train_L2_deltas.npy', train_L2_deltas)
        print("train_L2_deltas", train_L2_deltas)
        np.save(pathOUT+'valid_L2_deltas.npy', valid_L2_deltas)
        print("valid_L2_deltas", valid_L2_deltas)
        np.save(pathOUT+'test_L2_deltas.npy', test_L2_deltas)
        print("test_L2_deltas", test_L2_deltas)

                  
    if model.tloss == 'soft_c012s':

        # storing history of training            
        np.save(pathOUT+'train_losses_soft_c012s.npy', train_losses)
        print("train_losses_soft_c012s", train_losses)
        np.save(pathOUT+'valid_losses_soft_c012s.npy', valid_losses)
        print("valid_losses_soft_c012s", valid_losses)
        np.save(pathOUT+'test_losses_soft_c012s.npy',  test_losses)
        print("tets_losses_soft_c012s", test_losses)
                
    if model.tloss == 'regr_argmaxs':

        # storing history of training            
        np.save(pathOUT+'train_losses_regr_argmaxs.npy', train_losses)
        print("train_losses_regr_argmaxs", train_losses)
        np.save(pathOUT+'valid_losses_regr_argmax.npy', valid_losses)
        print("valid_losses_regr_argmax", valid_losses)
        np.save(pathOUT+'test_losses_regr_argmax.npy',  test_losses)
        print("test_losses_regr_argmax", test_losses)
                
    if model.tloss == 'soft_argmaxs':

        # storing history of training            
        np.save(pathOUT+'train_losses_soft_argmaxs.npy', train_losses)
        print("train_losses_soft_argmaxs", train_losses)
        np.save(pathOUT+'valid_losses_soft_argmaxs.npy', valid_losses)
        print("valid_losses_soft_argmaxs", valid_losses)
        np.save(pathOUT+'test_losses_soft_argmaxs.npy',  test_losses)
        print("test_losses_soft_argmaxs", test_losses)

                 
    if model.tloss == 'regr_c012s':

        # storing history of training            
        np.save(pathOUT+'train_losses_regr_c012s.npy', train_losses)
        print("train_losses_regr_c012s", train_losses)
        np.save(pathOUT+'valid_losses_regr_c012ss.npy', valid_losses)
        print("valid_losses_regr_c012s", valid_losses)
        np.save(pathOUT+'test_losses_regr_c012s.npy',  test_losses)
        print("test_losses_regr_c012s", test_losses)
                 
    if model.tloss == 'regr_weights':

        # storing history of training            
        np.save(pathOUT+'train_losses_weights.npy', train_losses)
        print("train_losses_weights", train_losses)
        np.save(pathOUT+'valid_losses_weights.npy', valid_losses)
        print("valid_losses_weights", valid_losses)
        np.save(pathOUT+'test_losses_weights.npy',  test_losses)
        print("test_losses_weights", test_losses)
   
    return train_accs, valid_accs, test_accs


def predictions(model, dataset, at_most=None, filtered=True):
    """ Given a built model instance and a dataset return the predictions as well
    as a list of dataset arguments (the number of data points is specified by the
    at_most argument limiting the range of the records received from the dataset) """
    sess = tf.get_default_session()

    x = dataset.x[dataset.mask]
    weights = dataset.weights[dataset.mask]
    filt = dataset.filt[dataset.mask]
    argmaxs = dataset.argmaxs[dataset.mask]
    c012s = dataset.c012s[dataset.mask]
    hits_argmaxs = dataset.hits_argmaxs[dataset.mask]
    hits_c012s = dataset.hits_c012s[dataset.mask]

    if at_most is not None:
      filt = filt[:at_most]
      x = x[:at_most]
      weights = weights[:at_most]
      argmaxs = argmaxs[:at_most]
      c012s = c012s[:at_most]
      hits_argmaxs = hits_argmaxs[:at_most]
      hits_c012s = hits_c012s[:at_most]
    
    # Fetching  preds (p), passing the features (x) - in that way we run the model
    # on the data from the provided dataset in TensorFlow 1 session: 
    p = sess.run(model.p, {model.x: x})

    if filtered:
      p = p[filt == 1]
      x = x[filt == 1]
      weights = weights[filt == 1]
      argmaxs = argmaxs[filt == 1]
      c012s = c012s[filt == 1]
      hits_argmaxs = hits_argmaxs[filt == 1]
      hits_c012s = hits_c012s[filt == 1]

    # Comment from ERW:
    # TODO: solve the problem with consistency - p is normalised to unity, 
    # but weights are not, which leads to the wrong estimate of the L1 and L2 metrics
    return x, p, weights, argmaxs, c012s, hits_argmaxs, hits_c012s



def softmax_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the softmax (i.e. in the form of probability values)
    predictions of the weights. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the weights values. 
    
    Moreover, it is literaly the EXACT COPY of regr_weights_predictions().
    The main difference in terms of the output of the model is visible not inside
    any of these two functions but in the match-case block of the neural network 
    class which describes the way of computing the loss function, as well as the
    choice of the appropriate labels. """
    sess = tf.get_default_session()

    x = dataset.x[dataset.mask]
    weights = dataset.weights[dataset.mask]
    filt = dataset.filt[dataset.mask]
    
    if at_most is not None:
        filt = filt[:at_most]
        weights = weights[:at_most]
        x = x[:at_most]

    if filtered:
        weights = weights[filt == 1]
        x = x[filt == 1]

    preds = sess.run(model.preds, {model.x: x})
    return weights, preds


""" TODO (Proposition): it seems that the first function from the two below 
is redundant as we have the same functionality provided by evaluate() function.
As for the second one, maybe we just need to modify it a little to make sure it is
compatible with evaluate() and the results then are going to be exactly the same.

Additionaly, there is probably an error in calculate_classification_metrics() as 
the calc_pred_argmaxs_signed_distances variable is utilising the same function stored
in the calc_pred_argmaxs_abs_distances variable. """
""" ============================================================================================ """
#prepared by Michal
def calculate_classification_metrics(pred_w, calc_w, args):

    num_classes = calc_w.shape[1]
    # normalising calc_w to probabilities
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, num_classes))
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    calc_pred_argmaxs_abs_distances = calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, num_classes)
    calc_pred_argmaxs_signed_distances = calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, num_classes)
    # Accuracy: average that most probable predicted class match most probable class
    # delta_class for matching  should be a variable in args
    delt_max = args.DELT_CLASSES
    acc = (calc_pred_argmaxs_abs_distances <= delt_max).mean()

    mean = np.mean(calc_pred_argmaxs_signed_distances)
    l1_delta_w = np.mean(np.abs(calc_w - pred_w)) / num_classes
    l2_delta_w = np.sqrt(np.mean((calc_w - pred_w) ** 2)) / num_classes

    return np.array([acc, mean, l1_delta_w, l2_delta_w])

#prepared by Michal
def evaluate_test(model, dataset, args, at_most=None, filtered=True):
    _, pred_w, calc_w, argmaxs, c012s, hits_argmaxs, hits_c012s = predictions(model, dataset, at_most, filtered)

    pred_w = calc_w  # Assume for tests that calc_w equals calc_w
    return calculate_classification_metrics(pred_w, calc_w, args)
""" ============================================================================================ """


def regr_weights_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the regression version of the weights. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the weights values. """
    sess = tf.get_default_session()

    x = dataset.x[dataset.mask]
    calc_weights = dataset.weights[dataset.mask]
    filt = dataset.filt[dataset.mask]

    if at_most is not None:
        filt = filt[:at_most]
        calc_weights = calc_weights[:at_most]
        x = x[:at_most]

    if filtered:
        calc_weights = calc_weights[filt == 1]
        x = x[filt == 1]

    pred_weights = sess.run(model.p, {model.x: x})
    return calc_weights, pred_weights



#prepared by Michal
def regr_c012s_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the regression version of the C0/C1/C2 values. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the c_coefficients values. """
    sess = tf.get_default_session()
    x = dataset.x[dataset.mask]
    calc_c012s = dataset.c012s[dataset.mask]
    filt = dataset.filt[dataset.mask]

    if at_most is not None:
        filt = filt[:at_most]
        calc_c012s = calc_c012s[:at_most]
        x = x[:at_most]

    if filtered:
        calc_c012s = calc_c012s[filt == 1]
        x = x[filt == 1]

    pred_c012s = sess.run(model.p, {model.x: x})
    return calc_c012s, pred_c012s



def soft_c012s_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the softmax (i.e. in the form of probability values)
    predictions of the C0/C1/C2 values. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the one-hot 
    encoded c_coefficients. """
    sess = tf.get_default_session()

    x = dataset.x[dataset.mask]
    calc_hits_c012s = dataset.hits_c012s[dataset.mask]
    filt = dataset.filt[dataset.mask]

    if at_most is not None:
        filt = filt[:at_most]
        calc_hits_c012s = calc_hits_c012s[:at_most]
        x = x[:at_most]

    if filtered:
        calc_hits_c012s = calc_hits_c012s[filt == 1]
        x = x[filt == 1]

    pred_hits_c012s = sess.run(model.p, {model.x: x})
    return calc_hits_c012s, pred_hits_c012s



def regr_argmaxs_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the regression version of the argmax values predictions. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the argmax values. """
    sess = tf.get_default_session()
    x = dataset.x[dataset.mask]
    calc_argmaxs = dataset.argmaxs[dataset.mask]
    filt = dataset.filt[dataset.mask]

    if at_most is not None:
        filt = filt[:at_most]
        calc_argmaxs = calc_argmaxs[:at_most]
        x = x[:at_most]

    if filtered:
        calc_argmaxs = calc_argmaxs[filt == 1]
        x = x[filt == 1]

    # Fetching  preds (p), passing the features (x) - in that way we run the model
    # on the data from the provided dataset in TensorFlow 1 session
    pred_argmaxs = sess.run(model.p, {model.x: x})
    return calc_argmaxs, pred_argmaxs



def soft_argmaxs_predictions(model, dataset, at_most=None, filtered=True):
    """ Compute and return the softmax (i.e. in the form of probability values)
    predictions of the argmax values. 
    TODO (Proposition): the function seems to implement part of the functionality 
    already available in predictions() function. It relies upon the one-hot 
    encoded argmax values. """
    sess = tf.get_default_session()

    x = dataset.x[dataset.mask]
    calc_hits_argmaxs = dataset.hits_argmaxs[dataset.mask]
    filt = dataset.filt[dataset.mask]

    if at_most is not None:
        filt = filt[:at_most]
        calc_hits_argmaxs = calc_hits_argmaxs[:at_most]
        x = x[:at_most]

    if filtered:
        calc_hits_argmaxs = calc_hits_argmaxs[filt == 1]
        x = x[filt == 1]

    # Fetching  preds (p), passing the features (x) - in that way we run the model
    # on the data from the provided dataset in TensorFlow 1 session:
    pred_hits_argmaxs = sess.run(model.p, {model.x: x})
    return calc_hits_argmaxs, pred_hits_argmaxs


#prepared by Michal
def calculate_roc_auc(pred_w, calc_w, index_a, index_b):
    """ Calculate the ROC for a specific pair of classes (useful for multiclass classification)"""
    n, _ = calc_w.shape
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([pred_w[:, index_a], pred_w[:, index_a]])
    weights = np.concatenate([calc_w[:, index_a], calc_w[:, index_b]])
    return roc_auc_score(true_labels, preds, sample_weight=weights)


def test_roc_auc(preds_w, calc_w):
    """ Test the ROC AUC for each class. This function calculates and prints the ROC AUC 
    for each class based on the predicted weights and the calculated weights. """
    n, num_classes = calc_w.shape
    for i in range(0, num_classes):
         print(i + 1, 'oracle_roc_auc: {}'.format(calculate_roc_auc(calc_w, calc_w, 0, i)),
                  'roc_auc: {}'.format(calculate_roc_auc(preds_w, calc_w, 0, i)))
 


def evaluate(model, dataset, args, at_most=None, filtered=True):
    """ Evaluate the performance of a given model on a dataset using various metrics.
    Return the accuracy (classes), meadn delta (classes), L1 and L2 (weights) """
    _, pred_w, calc_w, argmaxs, c012s, hits_argmaxs, hits_c012s = predictions(model, dataset, at_most, filtered)

    # Normalising calc_w (calculated weights) to probabilities
    num_classes = calc_w.shape[1]
    calc_w = calc_w / np.tile(np.reshape(np.sum(calc_w, axis=1), (-1, 1)), (1, num_classes))

    # Computing the mean of the difference between the most probable predicted 
    # class and the most probable true class (∆_class)      
    pred_argmaxs = np.argmax(pred_w, axis=1)
    calc_argmaxs = np.argmax(calc_w, axis=1)
    calc_pred_argmaxs_abs_distances = calculate_deltas_unsigned(pred_argmaxs, calc_argmaxs, num_classes)
    calc_pred_argmaxs_signed_distances = calculate_deltas_signed(pred_argmaxs, calc_argmaxs, num_classes)
    mean = np.mean(calc_pred_argmaxs_signed_distances)

    # ACC (accuracy): averaging that most probable predicted class match for t
    # the most probable class within the ∆_max tolerance. ∆max specifiec the maximum 
    # allowed difference between the predicted class and the true class for an event 
    # to be considered correctly classified.
    delt_max = args.DELT_CLASSES
    acc = (calc_pred_argmaxs_abs_distances <= delt_max).mean()
      
    # Computing the L1 and L2 norms for the weights  
    l1_delt_w = np.mean(np.abs(calc_w - pred_w))
    l2_delt_w = np.sqrt(np.mean((calc_w - pred_w)**2))
    
    return acc, mean, l1_delt_w, l2_delt_w


# Comment from ERW:
# TODO: evaluate_oracle and evaluate_preds should be
# implemented for multi-class classification.
# VERY IMPORTANT CLOSURE TEST
def evaluate_oracle(model, dataset, at_most=None, filtered=True):
    """ Evaluate the model performance on the optimal ("oracle")
    predictions. Serve as a closure test. """
    _, _, W_a, W_b = predictions(model, dataset, at_most, filtered)
    return evaluate_preds(W_a / (W_a + W_b), W_a, W_b)


def evaluate_preds(preds, wa, wb):
    """ Computhe the weighted Area Under Curve """
    n = len(preds)
    true_labels = np.concatenate([np.ones(n), np.zeros(n)])
    preds = np.concatenate([preds, preds])
    weights = np.concatenate([wa, wb])
    return roc_auc_score(true_labels, preds, sample_weight=weights)


def linear(x, name, size, bias=True):
    """ Linear activation function (used in the NN last layer) """
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [1, size],
                        initializer=tf.zeros_initializer())
     # + b vanishes in batch normalization
    return tf.matmul(x, w) 


def batch_norm(x, name):
    """ Batch normalisation (normalise the given layer input) """
    mean, var = tf.nn.moments(x, [0])
    normalized_x = (x - mean) * tf.rsqrt(var + 1e-8)
    gamma = tf.get_variable(name + "/gamma", [x.get_shape()[-1]], initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable(name + "/beta", [x.get_shape()[-1]])
    return gamma * normalized_x + beta


class NeuralNetwork(object):
    """ A neural network model for classification or regression tasks, allowing customization 
    of various parameters such as network architecture, loss function, and optimizer."""

    def __init__(self, num_features, num_classes, num_layers=1, size=100, lr=1e-3, keep_prob=1.0,
                 tloss="soft", activation='linear', input_noise=0.0, optimizer="AdamOptimizer"):
        
        # Defining TensorFlow placeholders for storing data
        batch_size = None
        self.x = x = tf.placeholder(tf.float32, [batch_size, num_features])
        self.weights = weights = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.argmaxs = argmaxs = tf.placeholder(tf.float32, [batch_size, 1])
        self.c012s = c012s = tf.placeholder(tf.float32, [batch_size, 3])
        self.hits_argmaxs = hits_argmaxs = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.hits_c012s = hits_c012s = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.tloss = tloss

        # Adding Gaussian noise to the input data (to make it more realistic)
        if input_noise > 0.0:
          x = x * tf.random_normal(tf.shape(x), 1.0, input_noise)

        # Passing the input forward through the hidden layers
        for i in range(num_layers):
            x = tf.nn.relu(batch_norm(linear(x, "linear_%d" % i, size), "bn_%d" % i)) 
            if keep_prob < 1.0:
              x = tf.nn.dropout(x, keep_prob)

        # Calculating predictions (the last layer output) and the loss function
        if tloss == "soft_weights":
            # "soft_weights", "soft_argmaxs", "soft_c012s" are a simple extension of 
            # what was implemented earlier as binary classification
            sx = linear(x, "classes", num_classes)
            self.preds = tf.nn.softmax(sx)
            self.p = self.preds            
            # labels: class probabilities, calculated as normalised weighs (probabilities)
            labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), (1,num_classes))
            self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
       
        elif tloss == "soft_argmaxs":
            sx = linear(x, "classes", num_classes)
            self.preds = tf.nn.softmax(sx)
            self.p = self.preds
            # labels: use hits map for argmaxs
            labels = hits_argmaxs / tf.tile(tf.reshape(tf.reduce_sum(hits_argmaxs, axis=1), (-1, 1)), (1,num_classes))
            self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
       
        elif tloss == "soft_c012s":
            sx = linear(x, "classes", num_classes)
            self.preds = tf.nn.softmax(sx)
            self.p = self.preds
            # labels: use hits map for single coefficient c012s
            labels = hits_c012s / tf.tile(tf.reshape(tf.reduce_sum(hits_c012s, axis=1), (-1, 1)), (1,num_classes))
            self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
       
        elif tloss == "regr_argmaxs":
            # TODO: not well learning close to angle = 0, 2pi
            # Old implementation:
            # self.loss = loss = tf.losses.mean_squared_error(self.argmaxs, sx)
            # new proposal by J. Kurek, does not work without correcting at analysis step
            # use for plotting script with "_topo" extension.
            # New implementation:
            sx = linear(x, "regr", 1)
            self.sx = sx
            self.p = sx
            self.loss = loss = tf.reduce_mean(1 - tf.math.cos(self.argmaxs - sx))
       
        elif tloss == "regr_c012s":
            sx = linear(x, "regr", 3)
            self.sx = sx
            self.p = sx
            self.loss = loss = tf.losses.mean_squared_error(self.c012s, sx)
        
        elif tloss == "regr_weights":
            sx = linear(x, "regr", num_classes)
            self.sx = sx
            self.p = sx
            self.loss = loss = tf.losses.mean_squared_error(self.weights, sx)

        else:
            raise ValueError("tloss unrecognized: %s" % tloss)

        # Initialising the optimiser
        optimizer = {"GradientDescentOptimizer": tf.train.GradientDescentOptimizer, 
                     "AdadeltaOptimizer": tf.train.AdadeltaOptimizer, 
                     "AdagradOptimizer": tf.train.AdagradOptimizer,
                     "ProximalAdagradOptimizer": tf.train.ProximalAdagradOptimizer, 
                     "AdamOptimizer": tf.train.AdamOptimizer,
                     "FtrlOptimizer": tf.train.FtrlOptimizer, 
                     "RMSPropOptimizer": tf.train.RMSPropOptimizer,
                     "ProximalGradientDescentOptimizer": tf.train.ProximalGradientDescentOptimizer}[optimizer]
        
        self.train_op = optimizer(lr).minimize(loss)

