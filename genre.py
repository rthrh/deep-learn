import os
import numpy as np
from python_speech_features import mfcc
import tensorflow as tf
import tensorflowvisu
import math
import sys


from matplotlib import pylab
# rt scipy.io.wavfile as wav

import sunau

genre_dict = {
    "blues" : 0,
    "classical" : 1,
    "country" : 2,
    "disco" : 3,
    "hiphop" : 4,
    "jazz" : 5,
    "metal" : 6,
    "pop" : 7,
    "reggae" : 8,
    "rock" : 9
}


flat = np.zeros((10,1),np.float32)

feed_dict = {}

XXX = np.zeros((1000,13),np.float32)
YYY = np.zeros((1000,10),np.chararray)

### READ AUDIO DATA FRAMES AND CALC MFCC
directory = 'C:/Users/adpa.MOBICAPL/Desktop/genrex/genres/'
for subdir in next(os.walk(directory))[1]:
    # print ('checking dir: ',subdir)

    subpath = directory + subdir + '/'
    for file in next(os.walk(subpath))[2]:
        
        file_path = subpath + file
        # print (file_path)
        # convert_dataset_to_wav(file_path)
        f=sunau.Au_read(file_path)
        audio_data = np.fromstring(f.readframes(10), dtype=np.float32)
        
        # FEATURES
        features = mfcc(audio_data)
        # print (features.shape)
        XXX = np.append(XXX,features)
        # print (XXX.shape)
        XXX = np.reshape(XXX,(13,-1))
        # print (XXX.shape)
        
        # LABELS
        label = file.split('.')[0]
        bit = genre_dict[label]
        label_score = np.zeros((10,1),np.float32)
        label_score[bit] = 1.0
        

        # print (label_score.shape)
        # break
        
        YYY = np.append(YYY,label_score)
        # print (YYY.shape)
        YYY = np.reshape(YYY,(10,-1))
        # print (YYY.shape)
        # YYY = np.reshape(YYY,(1,-1))
        # print (YYY)
        



        
print ('CREATING NN MODEL')
### CREATE NEURAL NETWORK MODEL
## INPUT
X = tf.placeholder(tf.float32, [13, 1, 1])

## LABELS
Y_ = tf.placeholder(tf.float32, [10,1])
## VARIABLES
W = tf.Variable(tf.zeros([13, 10]))

XX = tf.reshape(X, [-1, 13])

# biases b[10]

b = tf.Variable(tf.zeros([10]))
# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)



# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 100.0  # normalized for batches of 10 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualisation
# allweights = tf.reshape(W, [-1])
# allbiases = tf.reshape(b, [-1])
# I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
# It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
# datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)                       

# print (XXX.shape)
# print (YYY.shape)

print ("START LOOP")
for i in range(1000):
    batch_xs = XXX[:,i]
    batch_xs = np.reshape(batch_xs,(13,1,1))
    
    batch_ys = YYY[:,1]
    batch_ys = np.reshape(batch_ys,(10,1))
    

    sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys})
    

    
    # print ('i =',i)
    # print ("i = ",i,"    Accuracy = ",accuracy)
    
  
  
  
# batch_xs = tf.train.shuffle_batch_join(image, batch_size=batch_size, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)  
  
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
print(sess.run(accuracy, feed_dict={X: batch_xs, Y_: batch_ys}))
print ('PROGRAM EXIT')  
  
  
  
  
  
  
  

