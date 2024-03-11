# ADS_Project
**Lip Reading Using CNN and LSTM with an Emphasis on Low Light and Low-Resolution Video**

**Project Description**:

While there have been some projects devoted to Lip reading using CNN and LSTMS in the past, they have stayed true to the difficult but vanilla task of Lip Reading under optimal conditions
Even though there is still much room for improvement for this task (highest accuracy of .76), we decided to give attention to the less optimal task. We decided to go this route because users of such a technology
likely fall into two core groups. those using it for incredibly general tasks, ie government use, wide smartphone adoption, etc. The second group comprises the less fortunate, primarily deaf individuals who could 
use this technology to aid their lives. Because of these two target groups, it makes sense to attempt a model for an unideal case, low light and low resolution video. While we were not able to record our
our data set in these conditions, we were able to apply transformations to an existing dataset to mimic them. 

**Dataset**

The data used for this project was the M1RICLvc1 dataset.
The data set contains 10 speakers each saying a collection of 10 words, 10 times each. The data set is cited below with a downloadable kaggle access as well. 

Rekik, Ahmed, Achraf Ben-Hamadou, and Walid Mahdi. "A New Visual Speech Recognition Approach for RGB-D Cameras." Image Analysis and Recognition - 11th International Conference, ICIAR 2014, Vilamoura, Portugal, October 22-24, 2014, 2014, pp. 21-28.

https://www.kaggle.com/datasets/apoorvwatsky/miraclvc1

The downloaded dataset in Google Drive shared to all NYU users can be found here:
https://drive.google.com/drive/folders/1agFqF-bhEIVA_5WXMHlNdqPbI8kj8RVZ?usp=drive_link


**Repository Guide**

This Repository containts 4 files other than the README. 

Crop.ipynb which contains all logic for cropping

Data_Loader.ipynb which is used to load the dataset and perform transformations. 
  It is recommended to use the other files within this file

Tuning.ipynb which contains code used for hyperparameter testing 

Model.ipynb which contains our best-performing model.



**Additional References**

Data loading and cropping are significantly influenced by this post
https://www.kaggle.com/code/boisbois/final-final-code-clean





