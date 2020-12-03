



# Plots the images with labels
# rows= how man rows of pictures to display
def image_plot(rows, batch, cols=bs):
    """ 
    Plots the images with labels 
  
    Used in begining of notbook to show images that model will
    be trained on
  
    Parameters: 
    rows (int): number of rows for the images displayed
    batch (ImageDataGenerator): training batch of pictures that will be passed into model
    cols (int) : number of columns displayed ( set to batch size )
  
    Returns: 
    plot: Plot of images that will model will be fitted on 
  
    """
    fig, axs = plt.subplots(rows,cols,figsize=(cols*3,rows*3))
    for i in range(rows):
        images, labels = next(batch)
        for j, pic in enumerate(images):
            if rows > 1:
                axs[i,j].imshow(pic)
                axs[i,j].set_title(types[list(labels[j]).index(1)])
            else:
                axs[j].imshow(pic)
                axs[j].set_title(classlist[list(labels[j]).index(1)])
             
def image_plot_predict(rows, batch, model, cols=bs):  
    
    """ 
    Plots images with thier predicted class and actual class  
  
    Used after model has been fit to see results of test data
  
    Parameters: 
    rows (int): number of rows for the images displayed
    batch (ImageDataGenerator): test batch of pictures that will be predicted in model
    model : the model training images have been trained on to be used to predict test images
    cols (int) : number of columns displayed ( set to batch size )
    
    
    Returns: 
    plot: Plot of images with thier predicted class and thier actual class 
  
    """
    fig, axs = plt.subplots(rows,cols,figsize=(cols*3,rows*4))
    for i in range(rows):
        images, labels = next(batch)
        predictions = model.predict(images)
        for j, pic in enumerate(images):
            title = 'predict' + ' ' + \
            types[list(predictions[j]).index(predictions[j].max())] + ' ' + \
            '\n' + \
            'actual' + ' ' + \
            types[list(labels[j]).index(1)]
            if rows > 1:
                axs[i,j].imshow(pic)
                axs[i,j].set_title(title)
            else:
                axs[j].imshow(pic)
                axs[j].set_title(title)

                
def con_fu(rows, test_batches, model):
    """ 
    Collects data for confusion_matrix 
  
    Used in conjunction with sklearn.metrics confusion_matrix function
  
    Parameters: 
    rows (int): (rows * 6) number of pictures wil be predicted on
    test_batches (ImageDataGenerator) : the pictures that will be predicted
    model : the model training images have been trained on to be used to predict test images
    
    
    
    Returns: 
    l (list) : list of actual class per image
    o (list) : list of predicted class per image
    miss_class (list) : list of images that were missclassified
    """
    
    
    
    l = []
    o = []
    miss_class = []
    for i in range(rows):
        images, labels = next(test_batches)
         predictions = model.predict(images)
        for j, h in enumerate(labels):
            act_ = list(h).index(1.)
            pred_ = list(predictions[j]).index(predictions[j].max())
            l.append(act_)
            o.append(pred_)

            if act_ != pred_:
            miss_class.append(images[j])
  
    return l, o, miss_class


