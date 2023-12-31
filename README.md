# MA_MLBD - Project Botanist

**Authors** : Eric Bousbaa, Dylan Canton, Thomas Robert

**School** : HES-SO

**Course** : MA_MLBD

**Date** : 24.01.2023

---

[TOC]

---

## 1. Context

The University of Neuchatel has a large database of handwritten documents. Thoses docuemnts are notes taken by botanists during their field trips. The goal of this project is to create a machine learning model that would be able to predict if a specific handwriting belongs to a specific botanist.

## 2. Database description

We have two datasets. The first one is the public dataset. It regroups pre-processed images of handwritten documents. Inside, folders contains different types of images: paragraphes, lines, sentences and words. The images are already cropped and in binary format (black and white). Each group has a metadata text file associated.

The second one is the Neuchâtel dataset. It regroups scanned images of documents, that contains handwritten and typed text. The images are raw: different sizem, uncropped and in color. The files are regrouped in folders by botanist: "Chaillet", "Douteux" and "Other botanists".

## 3. Data pre-processing & feature extraction

With Neuchâtel dataset, we have to pre-process the images so they are close to the public dataset. We have to crop them, convert them to black and white and resize them to the same size as the public dataset. We could do that by hand, but we chose a more academic approach. We decided to use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to detect written text in the images and crop them. We then use [OpenCV](https://opencv.org/) to convert them to black and white and resize them.

Unfortunately, PaddleOCR is not able to differentiate handwritten text from typed text.

## 4. Machine Learning Techniques

*A brief description mentioning the ML techniques used and explaining why you chose them. Present the parameters of your model and explain how you selected them (e.g., in the case of an ANN: topology, activation functions, number of layers, number of hidden neurones per layer, etc). Present the parameters of the learning algorithm and explain how you selected them. (1 page)*

## 5. Experiments and results

*Experiments and results: describe the experiments and the results. Explain the reasoning behind those experiments, e.g., what are the hypothesis ? Use performance measures to evaluate them and explain the results*

## 6. Analyse et conclusions

###
