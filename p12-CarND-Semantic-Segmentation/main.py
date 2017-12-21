import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    print('Load VGG Called')

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    # https://www.tensorflow.org/api_docs/python/tf/saved_model/loader/load
    # sess -> The TensorFlow session to restore the variables.
    # tags -> Set of string tags to identify the required MetaGraphDef
    # export_dir-> Directory in which the SavedModel protocol buffer and variables to be loaded are located.
    # **saver_kwargs: Optional keyword arguments passed through to Saver.

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Once the model is loaded, Get the tuple of tensors
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    #:return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def conv_1x1(vgg_out, num_classes):
    return tf.layers.conv2d(vgg_out, num_classes, 1, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

def transpose_Convolution(conv_out, num_classes, kernel_size, strides):
    return tf.layers.conv2d_transpose(conv_out, num_classes, kernel_size, strides=(strides,strides), padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    print('Load Layers Called')
    # TODO: Implement function
    # VGG Architecture is based on FCN - 8 Architecture. In this architecture Fully connected layers are replaced by 1-by-1 convolutions
    # and thus preserving Spatial information.
    # The image H is h/4 (3rd), h/8(4th), h/16(7th) and h/32(fully connected Layer)

    # Convertin fully connected layer to 1x1 Convolution
    conv_out_7 = conv_1x1(vgg_layer7_out, num_classes)
    conv_out_4 = conv_1x1(vgg_layer4_out, num_classes)
    conv_out_3 = conv_1x1(vgg_layer3_out, num_classes)


    # With 1-by-1 Convolution complete, upsample the image to the original size. This is converting the Low Res Image to High Res agian.
    # The process is called Upsampling or Transpose Convolution. Upsample the h/4
    trans_Convolution_4 = transpose_Convolution(conv_out_7,num_classes,4,2)
    # Skip Layer - We will connect trans_Convolution_1 layer to the vgg_layer4_out to get additional spatial info
    skip_layer_4 = tf.add(trans_Convolution_4, conv_out_4)

    # Continue the process for Layer 4 and Layer 3
    trans_Convolution_3 = transpose_Convolution(skip_layer_4,num_classes,4,2)
    skip_layer_3 = tf.add(trans_Convolution_3, conv_out_3)


    trans_Convolution_16 = transpose_Convolution(skip_layer_3,num_classes,16,8)

    return trans_Convolution_16
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape predictions and labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Define loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # Define optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

     # Define train_op to minimise loss
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_stat = 0.8
    learning_rate_stat = 1e-4
    print('Training started ....')
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: keep_prob_stat,
                                          learning_rate:learning_rate_stat})
        print("Epoch %d of %d: Training loss: %.4f" %(epoch+1, epochs, loss))
    print('Training completed ....')
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

     # Set parameters
    epochs = 20
    batch_size = 1

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    print('Run started ....')
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # TODO: Train NN using the train_nn function
        final_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, num_classes)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
