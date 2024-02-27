import tensorflow.compat.v1 as tf


def linear(x, name, size, bias=True):
    """ Linear activation function (used in the NN last layer) """
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [1, size], initializer=tf.zeros_initializer())
    # TODO: + b vanishes in batch normalization
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
                 loss_type="soft", activation='linear', input_noise=0.0, optimizer="AdamOptimizer"):
        
        # Defining TensorFlow placeholders for storing data
        batch_size = None
        self.x = x = tf.placeholder(tf.float32, [batch_size, num_features])
        self.weights = weights = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.argmaxes = argmaxes = tf.placeholder(tf.float32, [batch_size, 1])
        self.c_coefficients = c_coefficients = tf.placeholder(tf.float32, [batch_size, 3])
        self.ohe_argmaxes = ohe_argmaxes = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.ohe_coefficients = ohe_coefficients = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.loss_type = loss_type

        # Adding Gaussian noise to the input data (to make it more realistic)
        if input_noise > 0.0:
          x = x * tf.random_normal(tf.shape(x), 1.0, input_noise)

        # Passing the input forward through the hidden layers
        for i in range(num_layers):
            x = tf.nn.relu(batch_norm(linear(x, "linear_%d" % i, size), "bn_%d" % i)) 
            if keep_prob < 1.0:
              x = tf.nn.dropout(x, keep_prob)
              
        # Calculating predictions (the last layer output) and the loss function
        match loss_type:
            # "soft_weights", "soft_argmaxs", "soft_c012s" are a simple extension of 
            # what was implemented earlier as binary classification
            case "soft_weights":
                sx = linear(x, "classes", num_classes)
                self.preds = tf.nn.softmax(sx)
                self.p = self.preds            
                # labels: the class probabilities, calculated as normalised weights (probabilities)
                labels = weights / tf.tile(tf.reshape(tf.reduce_sum(weights, axis=1), (-1, 1)), (1, num_classes))
                self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
           
            case "soft_argmaxes":
                sx = linear(x, "classes", num_classes)
                self.preds = tf.nn.softmax(sx)
                self.p = self.preds
                # labels: normalised one-hot encoded argmaxes
                labels = ohe_argmaxes / tf.tile(tf.reshape(tf.reduce_sum(ohe_argmaxes, axis=1), (-1, 1)), (1,num_classes))
                self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
            
            case "soft_c_coefficients":
                sx = linear(x, "classes", num_classes)
                self.preds = tf.nn.softmax(sx)
                self.p = self.preds
                # labels: normalised one-hot encoded coefficients
                labels = ohe_coefficients / tf.tile(tf.reshape(tf.reduce_sum(ohe_coefficients, axis=1), (-1, 1)), (1,num_classes))
                self.loss = loss = tf.nn.softmax_cross_entropy_with_logits(logits=sx, labels=labels)
            
            case "regr_argmaxes":
                # TODO: not well learning close to angle = 0, 2pi
                # Old implementation:
                # self.loss = loss = tf.losses.mean_squared_error(self.argmaxs, sx)
                # new proposal by J. Kurek, does not work without correcting at analysis step
                # use for plotting script with "_topo" extension.
                # New implementation:
                sx = linear(x, "regr", 1)
                self.sx = sx
                self.p = sx
                self.loss = loss = tf.reduce_mean(1 - tf.math.cos(self.argmaxes - sx))
            
            case "regr_c_coefficients":
                sx = linear(x, "regr", 3)
                self.sx = sx
                self.p = sx
                self.loss = loss = tf.losses.mean_squared_error(self.c_coefficients, sx)
            
            case "regr_weights":
                sx = linear(x, "regr", num_classes)
                self.sx = sx
                self.p = sx
                self.loss = loss = tf.losses.mean_squared_error(self.weights, sx)
           
            case default:
                raise ValueError(f"Unrecognised loss type: {loss_type}")

        # Initialising the optimiser
        optimizer = {"GradientDescentOptimizer": tf.train.GradientDescentOptimizer, 
                     "AdadeltaOptimizer": tf.train.AdadeltaOptimizer, 
                     "AdagradOptimizer": tf.train.AdagradOptimizer,
                     "ProximalAdagradOptimizer": tf.train.ProximalAdagradOptimizer, 
                     "AdamOptimizer": tf.train.AdamOptimizer,
                     "FtrlOptimizer": tf.train.FtrlOptimizer, 
                     "RMSPropOptimizer": tf.train.RMSPropOptimizer,
                     "ProximalGradientDescentOptimizer": tf.train.ProximalGradientDescentOptimizer}[optimizer]
        
        self.train_operation = optimizer(lr).minimize(loss)

