def circular_loss(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    return 1 - tf.cos(angle - y_true)
  
## training_dataset function

### data generators


