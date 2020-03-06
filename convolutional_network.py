# Import MINST data

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




import numpy
import numpy as np; npl = np.linalg
from numpy import *
#from numpy.random import randn
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import math
#import filterpy
import matplotlib.pyplot as plt
from numpy.random import *
from PIL import Image
import os
import io

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
##############################################################################################################
class AddSign(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign"):
        super(AddSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        init_forPrint = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_forPrint)
            lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
            beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
            alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

            eps = 1e-7  # cap for moving average

            m = self.get_slot(var, "m")
            m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

            # my_variable = tf.get_variable("grad", [1])
            #var_update = state_ops.assign_sub(var,self._lr_t*grad) #grandiance descent
            var_update = state_ops.assign_sub(var, lr_t * grad * (1.0 + alpha_t * tf.sign(grad) * tf.sign(m_t)))
            return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

#########################################################################################################
class SGD(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="SGD"):
        super(SGD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):

        var_update = state_ops.assign_sub(var,self._lr_t*grad) #grandiance descent

        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

####################################################################################################
class PowerSign(optimizer.Optimizer):
    """Implementation of PowerSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
        super(PowerSign, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")

        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))


        var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(
            tf.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value


        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
####################################################################################################

class PF(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="PF"):
        super(PF, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr,name="learning_rate")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)


    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        m = self.get_slot(var, "m")
        NumberParticle = 180
        thres = 0
        with tf.Session() as sessL:
            sessL.run(tf.global_variables_initializer())
            #print("grad:....")
            #print(sessL.run(grad))
            #print("var.....")
            #print(sessL.run(var))
            iii = 0
            while thres == 0:
                #------step 1:ini------------------
                #aa = tf.reduce_sum(m,keepdims=True)
               # print("aa")
               # print(sessL.run(aa))

                var_n = tf.reduce_sum(m )+ np.random.normal(0, lr_t.eval(session=sessL), NumberParticle)

                #print(sessL.run(m))
                #print(sessL.run(var_n))

                # ------step 2&3 : predict&update ------------------
                y_hat_n = []
                w = []

                for i in range(NumberParticle):
                    #print("var")
                    #print(sessL.run(var))
                    y_n = var - (lr_t*var_n[i]) #predict
                    #print("lr_t*var_n[i]")
                    #print(sessL.run(lr_t*var_n[i]))
                   # print("y_n")
                   # print(sessL.run(y_n))
                    tep = grad - y_n
                    print("y_n")

                    #*******dumpe y_n******and new file test tf.reduce_mean
                    print(sessL.run(y_n))

                    tep2 = tf.reduce_mean(y_n)

                    y_hat_n.append(tf.reduce_mean(tep)) #update



                    # ------step 4 : multinomial_resample  ------------------

                cumulative_sum = np.cumsum(w)
                cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
                index = np.searchsorted(cumulative_sum, random(len(w)))

                #-------------- replace index-----------------------
                var_n_hat = []
                for i in range(len(w)):
                    var_n_hat.append(var_n[index[i]].eval(session=sessL))


                m_t = mean(var_n_hat) #cal mean of selected particle
                #print("m_t")
                #print(m_t)
                get_grad = var.eval(session=sessL) - m_t    #cal update gradcal mean of selected particle
                print("get_grad: ")
                print(get_grad)
                print("origin_grad: ")
                print(grad.eval(session=sessL))

                #if np.abs(get_grad) > np.abs(grad.eval(session=sessL)):
                #    thres = 0
                # else:
                #     thres = 1

                thres = 1


        coeff = tf.Variable(m_t,dtype=tf.float32)
        var_update = state_ops.assign_sub(var,self._lr_t*grad+coeff)
        print(var_update)


        return control_flow_ops.group(*[var_update,coeff])



    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
####################################################################################################




# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add\
                      (tf.nn.conv2d(img, w,\
                                    strides=[1, 1, 1, 1],\
                                    padding='VALID'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding='VALID')

# Store layers weight & bias

wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32])) # 5x5 conv, 1 input, 32 outputs
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64])) # 5x5 conv, 32 inputs, 64 outputs
wd1 = tf.Variable(tf.random_normal([4*4*64, 1024])) # fully connected, 7*7*64 inputs, 1024 outputs
wout = tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)


bc1 = tf.Variable(tf.random_normal([32]))
bc2 = tf.Variable(tf.random_normal([64]))
bd1 = tf.Variable(tf.random_normal([1024]))
bout = tf.Variable(tf.random_normal([n_classes]))


# Construct model
_X = tf.reshape(x, shape=[-1, 28, 28, 1])


# Convolution Layer
conv1 = conv2d(_X,wc1,bc1)

# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)

# Apply Dropout
conv1 = tf.nn.dropout(conv1,keep_prob)


# Convolution Layer
conv2 = conv2d(conv1,wc2,bc2)

# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)

# Apply Dropout
conv2 = tf.nn.dropout(conv2, keep_prob)


# Fully connected layer
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) # Relu activation
dense1 = tf.nn.dropout(dense1, keep_prob) # Apply Dropout

# Output, class prediction
pred = tf.add(tf.matmul(dense1, wout), bout)

#pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

optimizer = PF(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print( "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
