# Boat-Classifier

While I was enlisted in the Navy, my job involved harbor security operations. We had two main responsibilities, escort submarines in and out of the port and identify and classify vessels in the area of responsibility. I had always wondered if the second part of the job could be done with machine learning. However, now that I am a Data Scientist, I decided to try to create a machine learning model that could classify common ships and boats that I would see every day.


### Resources

- The Images used were scraped from Google images
- Transfer learned from 2 different models:
  - VGG16 :13 convolutional layers, 5 pooling layers and 3 dense layers and 134m total parameters. 16m trainable
  
  - EfficentNetB0 : 230 layers that include 49 convolutional layers and 68m total parameters . 64m trainable parameters

The final model used for fitting and training was EfficentNetB0. EfficentNetB0 involves a method called Compound Scaling and uses a compound coefficient to uniformly scale width, depth, and resolution in a principled way. The user-specified coefficient controls resources (e.g. Floating Point Operations (FLOPs)) available for model scaling.

#### EfficentNetB0 structure:

![](Plots_and_images/efficenetb0.png)


### How a CNN:

Convolutional Neural Networks (CNN) are used in image processing and machine learning to effectively detect features and details. An image is passed through neurons with trainable and changeable weights that are all interconnected where at the end there’s an output. This output is than assessed and information is sent backwards to change the weights of the CNN.


![](Plots_and_images/howconvworks.png)

Above you can see an example of how the weights associated with the nodes affect how the computer "sees" the image. These weights help the computer identify edges and details that help in the overall classification of the image.

### Images:

In this project I decided to use 9 different classes of boats. These classes derived from real world experiences I had working as a Harbor Security Boat Coxswain. The 9 different classes that model was trained on were Cruise ships, Destroyers, Security Boats, Fishing Boats, Submarines, Carriers, Sailboats, Kayaks and Tugs.
The total number of images used was 1,090, all split into 3 groups, a Training, Testing, and Validation group. The training and validation group were used to fit the model and the test group was used to test the model.

![](Plots_and_images/9classes.jpg)


### Performance of model.

![](Plots_and_images/resultsofbestmodel.png)

The hyperparameters that were important in the building of this model was the activation function(softmax), the weights(noisy-student) , and the optimizer(RMSprop). As seen in the notebook, with trial and error , the model was able to reach 0.94 accuracy on the validation set with a loss of 0.554.

From the confusion matrix you can see that most of the categories are accurately identified except fishing boats.

These are the predictions of the model with their actual class label. Most of the classes are predicted correctly except the security boat predicted as a fishing boat.
![](Plots_and_images/predictions.jpg)


These are also some of the other predictions the model got wrong. These misclassifications are not out of the ordinary based on the size and features of the classes being mixed up.
![](Plots_and_images/wrongclassification.png)


### Utility Boats

After making the best model I could with the 9 different classes I wanted to see if combining the two classes that were the most similar (fishing and security) would increase the performance of the model. I combined the images of the fishing boats and security boats and re-split the images into a training, testing, and validating set. 

![](Plots_and_images/UtilityBoatCM.jpg)

From the confusion matrix above you can see that the overall accuracy of the model is actually lower and that more boats are being confused with the nee 'Utility Boats' class.

After rerunning the model and getting lower accuracy I wanted to see how a model would perform with just the fishing boats and security boats.
 
### Fishing and Security matrix

![](Plots_and_images/TwoBoats.jpg)

The overall accuracy from the model was 84.17% and from the confusion matrix it shows that the security boats were being misidentified 15.83% of the time as fishing boats.

### Conclusion

In conclusion the model used (EfficientNetB0) worked very well with the google images provided. To make the model better I will need to gather more accurate images of the different classes and see if there are any more hyperparameters that can be tweaked for better performance.

I would also like to implement an Identification Model that would be able to identify weather something was a boat or not and then classify that boat identified.
