**Motorcycle Classification with Convolutional Neural Networks**

Summary
=======

It always starts the same way. The forum post is always accompanied by a
photo of some random motorcycle. What bike is this? Experts and novices
alike scramble to find the make, model and year of the motorcycle
pictured. Wouldn't it be nice to easily classify a motorcycle from an
image? This project seeks to do just that.

Using the power of pre-trained convolutional neural networks, we can
customize the models to classify the year, make and model of motorcycle
images. What kind of performance can we expect? Is it even possible?
Motorcycles can be very similar between various models and years. We
will find that we can routinely achieve around 70% top-3 accuracy with a
relatively small data set. We will also find that to increase accuracy,
we would likely need to greatly increase the size of the data set
collected in this project. Along the way, we will look at methods to
collect and process data, while building a suitable model for
classification.

In the end, we will find that model tuning, data transformation, and
data augmentation have, at best, incremental benefits on the model. In
fact, the best time I spent on this project involved performance tuning
Pytorch itself to achieve faster modeling times.

![](./media/image1.png){width="6.5in" height="2.91875in"}

Data
====

Obtaining Data
--------------

Pre-processing 
---------------

EDA
---

Data Tuning
===========

Transforms
----------

Balance
-------

Modeling
========

Model Selection
---------------

### Resnet-34

### Batchnorm vs. Dropout

Model tuning
------------

### Learning rate

### Batch Size

Further Research
================

Conclusion
==========
