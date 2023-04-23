pj 1
==

Using python(start from scratch) to build a basic neural network in identificating handwriting digits
---
In this pj, I tried to build a 3layer neural network to identify the USPS dataset(United States Postal Service handwriting digits). This basic neural network uses sigmoid function as activation function and BP algorithm to accelerate sgd. Simply using this neural network I achieved an accuracy of 95%. Also I implemented several modifications including:
1. Changing the network structure(3->4layers, hidden units),
2. Implementing momentum.
3. Adding l2 regularization and implementing early stopping.
4. Using a softmax layer at the end of the network and implementing cross entropy error.
5. Implementing “dropout".
6. Implementing ‘fine-tuning’ of the last layer.
7. Artificially create more training examples by applying small transformations.

Finally I achievede the highest accuracy of 97%. Also there is still room for improvement, Such as implementing batch, 2D convolutional layer or doing some visulization. 
