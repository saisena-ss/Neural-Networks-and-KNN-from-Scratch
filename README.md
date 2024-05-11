#### Task 1:
##### Problem Formulation:
1. As given in the description, the problem is straight forward. I had to write euclidean and manhattan distances, fit function where in which we store the entire training data and predict to calculate distance between current input and training examples.
2. Euclidean distance = sqrt(sum((x1-x2)**2)) and manhattan distance = sum(|x1-x2|)

##### How the program works:
1. First an instance of knn is created with number of neighbors to check(k), weights to consider (uniform or distance) and distance type euclidean or manhattan.
2. After that, once the training data is passed to the fit function, it stores all of the data and also the labels
3. When the new data point is passed to predict, it calculates distance between new point and all the points in training data and checks which class is near most of the times out of k nearest points and predicts that as class label

##### Assumptios/Design Decision:
1. While predicting using distance as weights, some of the data points had distance as zero. So, to avoid run time warning - added some small distance (0.0001) to all distances 


#### Task 2:
##### Problem Formulation:
1. As given in the description, the problem is straight forward. I had to write 9 functions.
2. First completed the basic functions required in utils.py so that those can be used while 
getting the output for each layer.
3. Then completed remaining functions in multilayer_perceptron.py one by one.


##### How the program works:
1. First an instance of mlp is created with number of iterations, learning rate, hidden neuron, activation function for hidden layer.
2. After that, once the training data is passed to the fit function, _initialize is executed first to initiate some random weights for the hidden and output layer (including bias terms) and one hot encode the target labels and also to store unique target labels to decode one hot encoding while returning the prediction.
3. Next, neural network is feed forwarded using those weights till the output layer is reached, once the output layer is reached, back propagation starts.
4. In back propagation, gradient is calcualated for cross entropy loss with respect to the output weights and output bias first and then for the hidden layer.
5. Once the derivatives are calculated, all the weights are updated as w_new = w_old - learning rate * gradient.
6. Above 3,4,5 steps are repeated n_iteration times.
7. Now, to predict class label for a new data point, feed forward layer is run till the last layer in the network and the class with highest probability is predicted as it's label and one hot encoding is decoded back into true label.

##### Problems Faced/Assumptios/Design Decision:
1. Initially, was stuck in back propagating the network. But solved step by step to understand it properly, and implemented it finally.
