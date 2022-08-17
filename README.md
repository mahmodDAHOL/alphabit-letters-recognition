
# Recognize alphabit letters from shape of hand

in this project we train machine learning algorithms on hand images from A to Y except J


## How to use this repo
first of all, you have to download the dataset from this link: https://drive.google.com/drive/folders/1SY67sDO2ROoOoBhTBIIDn17gStS0AvCB

then extract file to outside source code and name it "dataset" as follow:

    project folder

        src

            ....

            ....    

        dataset

after that run this command :

python split_data.py --data_path YOUR_DATA_PATH --test_data_path_to_save YOUR_DATA_PATH/test --train_ratio 0.7

notice that in dataset folder, there is test folder and inside it test data taht is 0.3 of whole dataset
,then you have to move rest of data to new foler that it has name "train",
now your dataset are ready.

Now to train on your dataset you have to run main.py file, this file is responsible for create virtual environment and install all required packages for this project, and run running training.py, test.py files or both.


## what training.py do?
in this file we are extracting sift features from all images and clustering them to form bag of visual word, then we train following algorithms on them:

1- support vector machine (SVM)

2- naive bayes

3- logistic regression

4- k-nearest neighbors


and training convolutional neural network (CNN) on images directly .


## what test.py do?
this file turn on your camera and find a hand, then it draw rectangle around it, finally it use the model that has trained by cnn to predict class of that hand from A to Y except J.

hint: we used cnn model because it give best accuracy relative to other methods. 