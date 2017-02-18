import tensorflow as tf
import numpy as np

# Train recruisive function, formula:y = 0.1x +0.3

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#print(x_data,y_data)

# create tf struct begin #

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

Biases  = tf.Variable(tf.zeros([1]))

y = Weights*x_data + Biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# create tf struct end #


sess = tf.Session()
sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Biases))

w = sess.run(Weights)
b = sess.run(Biases)

print("Result is: Weight ----- Biases")
print(w, b)

"""
Running-Result：
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\core\common_runtime\gpu\gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0)
0 [ 0.15538068] [ 0.36024028]
20 [ 0.10616058] [ 0.29687488]
40 [ 0.10192242] [ 0.29902482]
60 [ 0.10059989] [ 0.2996957]
80 [ 0.10018721] [ 0.29990506]
100 [ 0.10005846] [ 0.29997036]
120 [ 0.10001825] [ 0.29999074]
140 [ 0.10000572] [ 0.29999712]
160 [ 0.10000179] [ 0.29999909]
180 [ 0.10000055] [ 0.29999971]

Result is: Weight ----- Biases
       [ 0.10000017] [ 0.29999992]
"""
