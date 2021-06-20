# Hand written digit recognition using Neural Networks

Implementation is done using regular Deep Neural Networks(DNN) and Convolutional Neural Networks(CNN). First observe how DNN based model works and then start learning CNN based model to get a better understanding of our model.
> All concepts used in DNN model are explained on [medium](https://naveen-varma.medium.com/hand-written-digit-recognition-using-deep-neural-networks-with-mnist-dataset-p-4-635bf20cb5e1)

> CNN concepts can be found [here](https://naveen-varma.medium.com/convolutional-neural-networks-cnn-concepts-p-5-9abc6e733bcc)

> Detailed explanation of CNN model code is [here]()

- for DNN based image classification program [click here](digit_recognition_minist_deep_neural_network.ipynb)
- for CNN based image classification program [click here](digit_recognition_using_CNN_MNIST.ipynb)

### Remainder:
- All neural networks are referred as Artificial Neural Networks(ANN). The neural networks with more than one hidden layer are called Deep Neural Networks(DNN). Convolutional Neural Networks(CNN) are mainly used in image processing.

## Executing the code: (I recommend Google colabs)
> CNN based digit recognition requires computational power, so use google colab for execution.
- First install these files
- Open Google colabs and open these files
- Goto runtime->change runtime->select GPU here to use computational power.
- Now run all cells. No need to install any libraries as google colab has preinstalled most of the python libraries that we use, in our case all libraries are preinstalled.

# In CNN based Model:

> All the below topics are discussed in the medium post, if you did refer the medium then you must be familiar with the below theory.

- We will load the training and testing images from MNIST first.
- Then display them on to a grid so that we can have a chance to witness the vareity of digits in each class.
- After that we'll need to prepare our data for training by reshaping it into specific format and then normalizing it.
- Then we start training the data. "Remember that the training data set is used to train the neural network to obtain the approriate parameters."

### importing data:
- importing training data to obtain the parameters and test data to evaluate the performance of the neural network.
- mnist.load_data imports 60000 images with labels into training data and 10000 images into testing data.
- Each image in the dataset is 28px wide and 28px height i.e each image has 784 pixels.

### assert function:
- assert func takes in a single argument, the argument is just a condition that is either True or False.
- If the condition is true then the code runs smoothly otherwise print a string
- Using this func is a good practice as it helps debug a complex problem
- NOTE: The no.of training images must be equal to the no.of labels for consistency.

### Preparing our data to use it in training
NOTE:
- Previously in DNN model, we flattened our image to give it as input, but here we are leaving the image as it is, that is 28*28 also add a depth of 1. With regular Neural Networks the image had to be flattened into a 1-d array of pixel intensities.
- For CNN its different. First we add a depth, Why depth?, as CNN works by applying filter to the channels of the image that are being viewed, in case of gray-scale images there is one channel present, therefore our data must reflect the presence of the depth.
- By adding this depth of 1 our data will be in the desired shape to be used as an input for the convolutional layer.

> Add depth and Perform One hot encoding on train and test data, which is necessary for multi class classification.

### Normalize the data
- We choose to divide by 255 because we want to normalize our data to be in a range between 0 and 1.
- This ensures that the max pixel value 255 is normalized down to the max value of 1.
- This normalization process is important as it scales down our features to a uniform range and decreases variance among our data. We need to ensure that our data has low variance. Helps to learn more clearly and accurately.

### Creating the model(leNet Model)
- There are predefined CNN Architectures like LeNet, AlexNet, ZFNet, GoogleNet etc.
- We will be designing a LeNet based architecture for digit recognition.
- LeNet model contains 2 convolutional layers and 2 pooling layers

### We will be using "dropout layer" to reduce overfitting
- This layer functions by randomly setting a fraction rate of input units to 0 at each update during training, which helps prevent overitting. This process is implemented only on training data not on testing or new data. So while using new data all nodes are utilzed to provide a more efficient result.
- we will be using only 1 dropout layer. However, more than one dropout layer can be used to increase the performance.
- Remember that these layers are mostly placed inbetween the layers that have a high number of parameters because the high parameter layers are more likely to overfit.
- Dropout rate is the amount of input nodes the dropout layers drops during each update. where 0 indicates to drop 0 nodes and 1 indicates to drop all nodes, 0.5 is the recommended rate.

### Note: We split our training data into training and validation sets.
- where training set is used to tune the weights and bias
- validation set is used to tune the hyper parameters.
- When ever the validation error is more than training error that indicates the start of our model overfitting
