# MA_MLBD - Project Botanist

**Authors** : Eric Bousbaa, Dylan Canton, Thomas Robert

**School** : HES-SO

**Course** : MA_MLBD

**Date** : 24.01.2024

---

[TOC]

---

TODO:

- Augmenter le dataset neuchatel : Dylan
- Tester avec dataset neuchatel : Eric + Thomas
- code cleanup + marche à suivre : Eric
- Présentation : Tous (<https://docs.google.com/presentation/d/1UJyt7ywg6fFq_mlWROCbzvzzPN0oVw38ch_AZ46q7xQ/edit?usp=sharing>)

## 1. Context

The University of Neuchatel maintains an extensive collection of handwritten documents, primarily composed of field notes from botanists. The objective of this project is to develop a machine learning model capable of accurately identifying the authorship of these notes based on unique handwriting characteristics.

This project is undertaken as part of the 'Machine Learning and Big Data' course in the MSE formation at HES-SO.

## 2. Database description

We are working with two distinct datasets. The first, a public dataset, comprises pre-processed images of handwritten documents. These images, categorized into folders, include various types such as paragraphs, lines, sentences, and words. They have been pre-cropped and converted into a binary format (black and white), with each category accompanied by an associated metadata text file. However, it is important to note that the images are not all the same size.

The second dataset, known as the Neuchâtel dataset, consists of scanned document containing image, annotations, handwritten and typed text. These images are in their raw form: varying in size, uncropped, and in full color. The files are organized into folders by the botanist's name, including 'Chaillet', 'Douteux', and 'Other botanists'.

Below are two typical image examples from the Neuchâtel dataset:

![Alt text](<assets/data-neuchatel-1.png>)

![Alt text](<assets/data-neuchatel-2.png>)

## 3. Data pre-processing & feature extraction

The initial step involves normalizing the data from both the public and Neuchâtel datasets. Our aim is to ensure the writings are as similar as possible, thereby eliminating any bias and facilitating the extraction of pertinent features. We will commence with the Neuchâtel dataset, given its raw and unprocessed nature.

### OCR

Our initial task is to distinguish the handwritten text from the printed one.

While this could be done manually, we have opted for a more academic approach. We've chose to use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to identify and crop the handwritten text within the images. Prior to saving the results, a pre-processing step is necessary. We've used [OpenCV](https://opencv.org/) to convert the images to black and white and resize them to a uniform height (in this case: 50px).

Below is an example of a processed image, following by some output image examples.

![Alt text](<assets/process.png>)

### OCR CNN

TODO: Dylan

Unfortunately, PaddleOCR is not able to differentiate handwritten text from typed text. We tried to create another model to detect handwritten text vs printed one, but we decided to focus on the main goal of the project and do this part by hand.

### Data pre-processing & augmentation

We used a similar pre-processing approach for the public dataset. We resized the images to a uniform height (50px) and cropped them in the middle to a fixed width. This width was a parameter we could adjust to improve the model's performance, but value around 150px seems to work well. We didn't convert the images to black and white, as the dataset is already in this format.

We then augmented the dataset by altering the images in various ways via the keras `ImageDataGenerator` class. This included rotating, shifting, and zooming the images. We also flipped the images horizontally and vertically. This was done to increase the number of images available for training, thereby improving the model's performance.

## 4. Machine Learning Techniques

TODO: Eric

We decided to have two different approaches to this problem: one using a CNN and one using an autoencoder.

### CNN

Our model started with a simple architecture with just some layers, just engough to do some first tests. We decided to improve it by using a dual-path structure with different filter sizes (3x3 and 5x5), enabling it to capture diverse features.

We try to avoid overfitting throug L1/L2 regularization layers. After our max pooling layers, we merge the two paths then add dense and dropout layer with a value of 0.2.

```python
# Define modele
input_layer = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))

# Define L1 and L2 regularization
l1_l2 = keras.regularizers.l1_l2(l1=0, l2=1e-4)

# path 1
conv1_1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2)(input_layer)
pool1_1 = keras.layers.MaxPooling2D((2, 2))(conv1_1)
conv1_2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2)(pool1_1)
pool1_2 = keras.layers.MaxPooling2D((2, 2))(conv1_2)

# path 2
conv2_1 = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', kernel_regularizer=l1_l2)(input_layer)
pool2_1 = keras.layers.MaxPooling2D((2, 2))(conv2_1)
conv2_2 = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l1_l2)(pool2_1)
pool2_2 = keras.layers.MaxPooling2D((2, 2))(conv2_2)

# merge paths
merged = keras.layers.concatenate([pool1_2, pool2_2])

flat = keras.layers.Flatten()(merged)
dense1 = keras.layers.Dense(128, activation='relu', kernel_regularizer=l1_l2, name=FLATTEN_LAYER_NAME)(flat)
dropout = keras.layers.Dropout(0.2)(dense1)  # Consider experimenting with the dropout rate
output_layer = keras.layers.Dense(N_CLASSES, activation='softmax')(dropout)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Autoencoder

Our second model is the autoencoder. It uses three convolutional layers with decreasing filters (64, 32, 16).

```python
# Encoder
input_img = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)) # adapt this if using `channels_first` image data format

x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)

# Decoder
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Autoencoder model
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
encoder_model = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_layer').output)
```

### K-fold

To ensure the reliability of our results, we decided to use a k-fold cross validation. This method is used to validate the model's performance on unseen data. It is particularly useful as our dataset is relatively small, as it allows us to use all the data for training and testing.

We used a k-fold cross validation to train our models. We split the dataset into 5 folds, then train the model on 4 folds and test it on the remaining one.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

## 5. Experiments and results

TODO: Dylan

*Experiments and results: describe the experiments and the results. Explain the reasoning behind those experiments, e.g., what are the hypothesis ? Use performance measures to evaluate them and explain the results*

## 6. Analyse et conclusions

TODO: Dylan
