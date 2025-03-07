`## digit-recognition ##`

This is my attempt to create a digit recognition
neural network from scratch (without machine learning
frameworks, only numpy + math).

----

`## Model info ##`

- It consists of one input layer (for the 28 * 28 grayscale
pixel values), 2 hidden layers (each with 10 nodes) and
one output layer (which corresponds which digit is the most likely).
- It uses relu activation layers for the hidden layers and
a softmax activation layer for the output layer.
- It uses the MSE as the loss function. The error in this case is the
model output subtracted by a one hot vector of the expected output.
- It uses standard backpropagation with a constant learning rate and
stochastic learning to update the weight matrix.
- The weights are initialized using a normal distribution, as described
in the research paper of Le Cun et al.
- I could get it accurate up to 96%.
- It is licensed under the MIT license.

----

`## Dependencies ##`

- numpy
- matplotlib

----

`## Training the model ##`

To start training pleas use `python3 main.py`.
This will initialize the model and start training. Periodically a matplotlib
window will be opened showing a sample of test images and the model prediction
to visualize the model's accuracy.

----

&copy; 2024 Lucas Birkert - All Rights Reserved
