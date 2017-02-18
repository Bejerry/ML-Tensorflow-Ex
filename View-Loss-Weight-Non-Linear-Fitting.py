import tensorflow as tf  
import numpy as np  
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):  
    # add one more layer and return the output of this layer  
    layer_name = 'layer%s' % n_layer  
    with tf.name_scope(layer_name):  
        with tf.name_scope('weights'):  
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  
            tf.summary.histogram(layer_name + '/weights', Weights)  
        with tf.name_scope('biases'):  
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  
            tf.summary.histogram(layer_name + '/biases', biases)  
        with tf.name_scope('Wx_plus_b'):  
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  
        if activation_function is None:  
            outputs = Wx_plus_b  
        else:  
            outputs = activation_function(Wx_plus_b, )  
        tf.summary.histogram(layer_name + '/outputs', outputs)  
    return outputs  
# Make up some real data  
x_data = np.linspace(-1,1,300)[:, np.newaxis]  
noise = np.random.normal(0, 0.05, x_data.shape)  
y_data = np.square(x_data) - 0.5 + noise  
# define placeholder for inputs to network  
with tf.name_scope('inputs'):  
    xs = tf.placeholder(tf.float32, [None, 1],name='input_x')  
    ys = tf.placeholder(tf.float32, [None, 1],name='input_y')  
  
# add hidden layer  
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  
# add output layer  
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)  
  
  
# the error between prediciton and real data  
with tf.name_scope('loss'):  
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),  
                                        reduction_indices=[1]))  
    tf.summary.scalar('loss', loss)  
with tf.name_scope('train'):  
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  
  
sess = tf.Session()  
merged = tf.summary.merge_all()  
# save the logs  
writer = tf.summary.FileWriter("logs/", sess.graph)  
sess.run(tf.global_variables_initializer())  
for i in range(1000):  
    # training  
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  
    if i % 50 == 0:  
        # to see the step improvement  
        result = sess.run(merged,  
                          feed_dict={xs: x_data, ys: y_data})  
        writer.add_summary(result, i)  


"""
打一下命令：
$ tensorboard --logdir='logs/'
再在网页中输入链接：127.0.1.1:6006 即可获得展示
"""
