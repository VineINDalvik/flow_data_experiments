import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from scipy import misc,ndimage
import myDataBatch

# mnist = input_data.read_data_sets('./data/fashion')
data_path = ['/home/vine/data_all','/home/vine/yaoye/result']
zone = "WALL-5"
#zone = ""

batch_size = 100
width,height = 128,128
mnist_dim = width*height*5
random_dim = 10
epochs = 1000000

def my_init(size):
    return tf.random_uniform(size, -0.5, 0.5)

D_W1 = tf.Variable(my_init([mnist_dim, 128]))
D_b1 = tf.Variable(tf.zeros([128]))
D_W2 = tf.Variable(my_init([128, 32]))
D_b2 = tf.Variable(tf.zeros([32]))
D_W3 = tf.Variable(my_init([32, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

G_W1 = tf.Variable(my_init([random_dim, 32]))
G_b1 = tf.Variable(tf.zeros([32]))
G_W2 = tf.Variable(my_init([32, 128]))
G_b2 = tf.Variable(tf.zeros([128]))
G_W3 = tf.Variable(my_init([128, mnist_dim]))
G_b3 = tf.Variable(tf.zeros([mnist_dim]))
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

def D(X):
    X = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    X = tf.nn.relu(tf.matmul(X, D_W2) + D_b2)
    X = tf.matmul(X, D_W3) + D_b3
    return X

def G(X):
    X = tf.nn.relu(tf.matmul(X, G_W1) + G_b1)
    X = tf.nn.relu(tf.matmul(X, G_W2) + G_b2)
    X = tf.nn.sigmoid(tf.matmul(X, G_W3) + G_b3)
    return X

real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
random_Y = G(random_X)

eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
X_inter = eps*real_X + (1. - eps)*random_Y
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

D_loss = tf.reduce_mean(D(real_X)) - tf.reduce_mean(D(random_Y)) + grad_pen
G_loss = tf.reduce_mean(D(random_Y))

D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=D_variables)
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=G_variables)
#ONLY USE CPU, PROVED no effect....
#config = tf.ConfigProto(device_count = {'GPU': 0})
loss_ops = [G_loss, D_loss]
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

image_options = {'image_size': width}
data_reader = myDataBatch.BatchDatset(inpath=data_path,inzone=zone,image_options=image_options)


for e in range(epochs):
    for i in range(5):
        real_batch_X = data_reader.next_batch(batch_size)
        real_batch_X = real_batch_X.reshape(batch_size, -1)
        random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
        _= sess.run([D_solver], feed_dict={real_X:real_batch_X, random_X:random_batch_X})
        #writer.add_summary(sum1)
    random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
    _,G_loss_,D_loss_,su = sess.run([G_solver]+loss_ops+[summary_op], feed_dict={real_X:real_batch_X, random_X:random_batch_X})
    writer.add_summary(su, e)
    if e % 100 == 0:
        print('epoch %s, D_loss: %s, G_loss: %s'%(e, D_loss_, G_loss_))
        n_rows = 1
        n_cols = 5
        check_imgs = sess.run(random_Y, feed_dict={random_X:random_batch_X}).reshape((batch_size, width, height, -1))
        #print check_imgs.shape
        check_imgs = check_imgs[:n_rows*n_rows]
        imgs = np.ones((height*n_rows, width*n_cols))
        #imgs = np.ones((1, 5 ))
        for i in range(5):
            #print(i,n_rows,width,height,5+5*(i%n_rows)+width*(i%n_rows),5+5*(i%n_rows)+width+width*(i%n_rows),5+5*int(i/n_rows)+height*int(i/n_rows),5+5*int(i/n_rows)+height+height*int(i/n_rows),check_imgs.shape)
            #imgs[(5+5*(i%n_rows)+width*(i%n_rows)):(5+5*(i%n_rows)+width+width*(i%n_rows)), (5+5*int(i/n_rows)+height*int(i/n_rows)):(5+5*int(i/n_rows)+height+height*int(i/n_rows))] = np.sum(check_imgs[i],axis=2)
#print imgs.shape
            #print check_imgs[0][:,:,i].shape
            imgs[ height*(i/n_cols):height*(i/n_cols)+height,width*(i%n_cols):width+width*(i%n_cols)] = check_imgs[0][:,:,i]

        misc.imsave('out/%s.png'%(e/100), imgs)
