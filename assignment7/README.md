The simple ResNet model is placed in the models/ dir in the repo and the util functions are collected in various files inside the models/utils/ dir, including img_utils.py (for plotting and the like), data_utils.py (Datasets and DataLoaders), and train_loop.py for train and test loops.

To execute the assignment-7 code, please refer to the assignment7/ dir and inside that dir, assignment7_notebook.ipynb which is jupyter notebook (not a main.py script, but a notebook). 

The misclassified images and the gradcam images are also present in the assignment7/ dir (output.png and output_gradcam.png respectively). For ease of checking, 

![output](https://user-images.githubusercontent.com/14867819/219497546-ed61f497-5b90-460d-adb1-b542afae5747.png)

the above is the plot with a sample of 20 misclassified images and applying gradcam, we have the below image:

![output_gradcam](https://user-images.githubusercontent.com/14867819/219497719-4eadd962-5fa9-4f05-ba50-6d87c51db149.png)
