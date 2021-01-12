import tensorflow as tf

def circular_loss(y_true, y_pred):
    '''
    Computes the circular loss of two tensors.
    
    Positional arguments:
    y_true -- a tensor with shape (N, 1), where N is the number of ground-truth angles
    y_pred -- a tensor with shape (N, 1), where N is the number of predicted angles
    '''
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return tf.sqrt(1.001 - tf.cos(y_pred - y_true)) # not using a flat 1 prevents that tf.sqrt will occasionally return nan

def mean_abs_zscore_difference(y_true, y_pred):
    '''
    Performs z-scoring within a batch, and computes loss as the mean absolute difference.
    
    Positional arguments:
    y_true -- a tensor with shape (N, 1)
    y_pred -- a tensor with shape (N, 1)
    '''
    mean = tf.math.reduce_mean(y_true)
    std = tf.math.reduce_std(y_true)
    y_true = y_true - mean
    y_pred = y_pred - mean
    y_true = y_true / std
    y_pred = y_pred / std
    return tf.reduce_mean(tf.math.abs(y_true - y_pred))
