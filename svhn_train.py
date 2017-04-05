# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 01:09:36 2017

@author: babburi
"""
import tensorflow as tf
import cnn_model
import svhn_input
import numpy as np

def train():
    
    X_train = svhn_input.X_train
    y_train = svhn_input.y_train
    x = cnn_model.x
    y = cnn_model.y
    batch_size = svhn_input.batch_size
    global_step = tf.Variable(0, name='global_step', trainable=False)
    initial_lr = 0.01
    
    prediction = cnn_model.convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    lr = tf.train.exponential_decay(initial_lr, global_step, decay_steps=1000, decay_rate=0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost, global_step)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    n_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
         
        for epoch in range(n_epochs):
            epoch_loss = 0
            i = 0
            while i <= X_train.shape[3]:
                start = i
                end = i + batch_size
                batch = {x: X_train[:,:,:,start:end], y: y_train[start:end,:]}
                _, c = sess.run([optimizer, cost], feed_dict = batch)
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, 'completed out of', n_epochs, 'loss:', epoch_loss)
            randsel = np.random.randint(1, X_train.shape[3], batch_size)
            train_accuracy = accuracy.eval(feed_dict={x: X_train[:,:,:,randsel], y: y_train[randsel,:]})
            print("Training accuracy", train_accuracy)
        
        #print('Test Accuracy:', accuracy.eval({x:X_test, y:y_test}))


def main():
    train()

if __name__ == "__main__":
    main()