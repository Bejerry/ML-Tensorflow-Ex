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

print("Result: Weight ----- Biases")
print(w, b)

