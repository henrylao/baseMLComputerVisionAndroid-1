<b><h2>baseMLComputerVisionAndroid repository</h2></b>

baseMLComputerVisionAndroid  repository is a copy of an early version of [Covid ID repository](https://github.com/grewe/covid_id/wiki) is a project out of [iLab](http://borg.csueastbay.edu/~grewe/ilab/index.html) associated with [Computer Science at California State University East Bay](https://www.csueastbay.edu/cs/) to explore how Computer Vision and Machine Learning can be used to assist the general public with situation awareness as it relates to Covid-19.  This code repostiory was created as a BASE Android application using Computer Vision and Tensorflow Machine Learning.

This app in the current state simply loads up a google map with a set of buttons on top for logging in (you can ignore this --if you wish you can add later your own authentication code), and two buttons - one for classification and the other for detection.   The first launches the [ClassifierActivity class](https://github.com/grewe/baseMLComputerVisionAndroid/blob/master/app/src/main/java/edu/ilab/covid_id/classification/ClassifierActivity.java) which takes every frame captured by Android camera and uses a Flower trained classifier ML model on it to perform predictions and displays the results.   The second  launches the [DetectorActivity class](https://github.com/grewe/baseMLComputerVisionAndroid/blob/master/app/src/main/java/edu/ilab/covid_id/localize/DetectorActivity.java) which again takes every frame captured by Anadroid and puses a Object Detector (general purpose trained on COCO dataset) ML model on it to perform detections and displays the results.   

Additionally there are some supporting Data classes that could be used to store records on the backend using Google Firestore --at this point you will not have access to store the data on MY backend so you should turn this feature off (comment out this code) in the ClassifierActivity and DetecorActivity classes but, I wanted to share with you the logic/code to make this happen.  If you wish and you understand or are willing to learn Google Firestore on your own you have the option of modifying as necessary to point to your Firestore so the records can be stored.   The data files are located in the [data package](https://github.com/grewe/baseMLComputerVisionAndroid/tree/master/app/src/main/java/edu/ilab/covid_id/data).

Note the code that uses Google Auntehtication for authenticating a user based on the Login button being hit on the main activity interface can be found in the [auth package](https://github.com/grewe/baseMLComputerVisionAndroid/tree/master/app/src/main/java/edu/ilab/covid_id/auth)

developed by [![image](https://user-images.githubusercontent.com/11790686/82628915-0db28800-9ba3-11ea-817d-a0dcfe447ad7.png)](http://borg.csueastbay.edu/~grewe/ilab/index.html)         contact: [Professor Lynne Grewe](mailto:lynne.grewe@csueastbay.edu)

#### Important 1: you will need to supply your own Google Maps API key for the starting map activity to display a map properly

#### Important 2: you will not be able to store things in MY firestore as your application has not been granted permission ---you need to create your OWN firebase backend and follow the instructions to [set it up so YOUR android application will be able to communicate to it](https://firebase.google.com/docs/android/setup)







