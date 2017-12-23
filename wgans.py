from __future__ import print_function, division
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def disp_images(images):
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    figure = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return

def preprocess(x):
    return 2*x-1.0

def deprocess(x):
    return (x+1.0) / 2.0

def rel_error(x):
    return np.max(np.abs(x-y)/ (np.maximum(1e-8. np.abs(x) + np.abs(y))))

def count_params():
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count

def session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    return session

answers = np.load('gan-checks-tf.npz')

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

disp_images(mnist.train.next_batch(16)[0])

def activation(x, alpha=0.01):
    a = tf.maximum(x, alpha*x)
    return a

def sample_noise(batch_size, dim):
    random_noise = tf.random_uniform(maxval=1,minval=-1,shape=[batch_size, dim])
    
    return random_noise

def get_solvers(lr= 1e-3, beta1=0.5):
    d_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    g_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    return d_solver,g_solver

def generator(x):
    with tf.variable_scope("generator"):
        fc1 = tf.layers.dense(inputs=z, units=1024,activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
        img = tf.layers.dense(inputs=fc2, units=784, activation=tf.nn.tanh)
        
        return img


def train(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step,
        D_extra_step,show_every=250, print_every=50, batch_size=128, num_epoch=10):

    max_iter = int(mnist.train.num_examples*num_epoch/batch_size)
    for it in range(max_iter):
        
        if(it % show_every == 0):
            samples = sess.run(G_sample)
            fig = disp_images(samples[:16])
            plt.show()
            print()
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        if(it % print_every == 0):
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(it,D_loss_curr,G_loss_curr))
    print('Final images')
    samples = sess.run(G_sample)

    fig = disp_images(samples[:16])
    plt.show()

def discriminator(x):
    with tf.variable_scope('discriminator'):
        init = tf.contrib.layers.xavier_initializer()
        x = tf.reshape(x, [-1,28,28,1])
        x = tf.layers.conv2d(x, 64, 4, activation=activation, strides=2, padding='valid',
                            kernel_initializer=init, name='conv_0')
        x = tf.layers.conv2d(x,128,4, activation=activation, strides=2, padding='valid',
                            kernel_initializer=init, name='conv_1')
        x = tf.layers.batch_normalization(x, name='batchnorm_0')
        x = tf.reshape(x, [-1, 3200])
        x = tf.layers.dense(x, 1024, activation=activation, kernel_initializer=init,
                           name='dense_0')
        logits = tf.layers.dense(x,1,kernel_initializer=init, name='logits')
        return logits
    
def test_discriminator(true_count=267009):
    tf.reset_default_graph()
    with session() as sess:
        y = discriminator(tf.ones((2, 784)))
        cur_count = count_params()
        if cur_count != true_count:
            print('Incorrect number of parameters in discriminator. {0} instead of {1}. Check your achitecture.'.format(cur_count,true_count))
        else:
            print('Correct number of parameters in discriminator.')
        
test_discriminator(3411649)

tf.reset_default_graph()

batch_size = 128
noise_dim = 96

x = tf.placeholder(tf.float32, [None,784])
z = sample_noise(batch_size, noise_dim)

G_sample = generator(z)

with tf.variable_scope("") as scope:
    
    logits_real = discriminator(preprocess(x))
    
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)
    
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

d_solver, g_solver = get_solvers()

def wgan_loss(logits_real, logits_fake, batch_size,x,G_sample):
    d_loss =- tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake)
    g_loss =- tf.reduce_mean(logits_fake)
    
    lam = 10
    
    eps = tf.random_uniform([batch_size,1], minval=0.0,maxval=1.0)
    x_h = eps*x+(1-eps)*G_sample
    
    with tf.variable_scope("", reuse=True) as scope:
        grad_d_x_h = tf.gradients(discriminator(x_h), x_h)
    
    grad_norm = tf.norm(grad_d_x_h[0], axis=1, ord='euclidean')
    grad_pen = tf.reduce_mean(tf.square(grad_norm-1))
    
    d_loss+=lam*grad_pen
    return d_loss, g_loss

d_loss, g_loss = wgan_loss(logits_real, logits_fake,128,x,G_sample)
d_train_step = d_solver.minimize(d_loss, var_list=D_vars)
g_train_step = g_solver.minimize(g_loss, var_list=G_vars)
d_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
g_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

with session() as sess:
    sess.run(tf.global_variables_initializer())
    train(sess,g_train_step,g_loss,d_train_step,d_loss,g_extra_step,d_extra_step,batch_size=128,
          num_epoch=5)
    

