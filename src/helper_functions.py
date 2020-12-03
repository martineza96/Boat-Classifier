




def image_plot(rows, batch, cols=bs):   
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


