import tensorflow as tf


# 1，介绍 Variable ：

state = tf.Variable(0, name='count')

print (state.name)

one = tf.constant(1)

value = tf.add(state, one)

update = tf.assign(state, value)

# init = tf.initialize_all_variables()  deprecated and will be removed after 2017-03-02.
init = tf.global_variables_initializer() # if define some variable

with tf.Session() as sess:
    sess.run(init)
    for X in range(3):
        sess.run(update)
        print(sess.run(state))


# 2，介绍 placeholder :

input1 = tf.placeholder(tf.float32)

input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    # sess.run(output)
    print(sess.run(output, feed_dict={input1:[7., 3.], input2:[2.2, .5]}))
    
Running-Result:
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0)
1
2
3
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0)
[ 15.40000057   1.5       ]
