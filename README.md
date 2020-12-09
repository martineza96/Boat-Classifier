# Boat-Classifier

While I was enlisted in the Navy, my job involved harbor security operations. We had two main responsibilities, escort submarines in and out of the port and identify and classify vessels in the area of responsibility. I had always wondered if the second part of the job could be done with machine learning. However, now that I am a Data Scientist, I decided to try to create a machine learning model that could classify common ships and boats that I would see every day.


### Resources

- The Images used were scraped from Google images
- Transfer learned from 2 different models:
  - VGG16 :
    -13 convolutional layers, 5 pooling layers and 3 dense layers.
     134m total parameters. 16m trainable
  -EfficentNetB0 :
    -230 layers that include 49 convolutional layers
     68m total parameters . 64m trainable parameters

The final model used for fitting and training was EfficentNetB0. EfficentNetB0 involves a method called Compound Scaling and uses a compound coefficient to uniformly scale width, depth, and resolution in a principled way. The user-specified coefficient controls resources (e.g. Floating Point Operations (FLOPs)) available for model scaling.

#### EfficentNetB0 structure 

![](Plots_and_images/efficenetb0.png)
