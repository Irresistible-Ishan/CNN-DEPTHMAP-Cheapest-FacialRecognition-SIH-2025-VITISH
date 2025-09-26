# CNN-DEPTHMAP-Cheapest-FacialRecognition-SIH-2025-VITISH
Problem Statement : SIH25012 , Automated Attendance System for Rural Schools by Govt. Of Punjab , India;  

## Brief on how this is working : (Please also refer to the PPTX files given with the files) :
### Training : The code first stores the Images into a 128 Dimensional vector space , then Model trains to put similar images nearby in that vector space what model learns is to convert an image into the vector embedding with precise and accurate positioning 
### Usage : when someone enters a face image , it converts the image into the embedding then searches for the similar embedding in the vector space using mathematical vector search methods. and if it exists then it compares the similarity of all the axis features and gives a score and also returns the label ID through which it can extract other data regarding that person from any data handling methods such as a csv that is encrypted with military grade encryptions and with implementation of blockchain technology to rectify false inputs and data manipulation and also to safeguard data of students.
### Future updates : the cropping of the face image is manual from a full portrait image , a face detection model is to be integrated such that only the closeup face image is cropped and passed into the CNN . Also its argued that anyone can put up a fake image or a video of the person to make system think its the person , to fix this issue its very simple and its by implementing the DEPTH mapping , to use the recent models on hugging face released by many researchers which converts IMAGE into a DEPTH map while a real face would have a depth similar to a face , a picture wouldnt have a depth and would be 2d , a model can be further trained on this to match depth as well as using CNN to know the exact identity. 

DEPTH mapping example : 

# RESULT:
### when compared with recent two images of a friend
- > 99.3% accuracy (with crop) (close up face pic)

### when i took my current pic and stored it in embedding
- > then compared my 4 years old pic : 94.8% (without crop) (not a close up face pic)
- > then compared my childhood pic : 99.5% (without crop) (not a close up face pic)
  
Threshold decision 92% or more: for prototype (can be taken not necessary)



## FUTURE PLAN : 
### 1. Adding a depth mapping , IMAGE -> DEPTH MAP. to make system foolproof
### 2. Facedetection : to crop the close up of the face 
### 3. Adding live classroom detection 24/7 with relative classroom movement and student behaviour 







## DATASET USED : https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset
## Nikit Periwal · Updated 4 years ago - DATASET AUTHOR KAGGLE

About Dataset
Welcome to Face Recognition Dataset, a database of face photographs designed for the creation of face detection and recognition models. This dataset has been derived from the Labeled Faces in the Wild Dataset.
This dataset is a collection of JPEG pictures of famous people collected on the internet. All details are available on the official website:
http://vis-www.cs.umass.edu/lfw/

The Dataset
Each picture is centered on a single face, and every image is encoded in RGB. The original images are of the size 250 x 250.
The dataset contains 1680 directories, each representing a celebrity.
Each directory has 2-50 images for the celebrity.
Extracted Faces
Faces extracted from the original image using Haar-Cascade Classifier (cv2)
encoded in RGB and size of image is 128, 128
Acknowledgement
We wouldn't be here without the help of others.
I would like to thank Computer Vision Laboratory, University of Massachusetts for providing us with such an excellent database.

Inspiration
I wanted to build a one-shot model for face recognition. This was the dataset I ended up using for my work. I'm tweaking it and posting it here in Kaggle, so that others can easily use this dataset to build similar models


license: 
CC0 1.0 Universal 
CC0 1.0 Deed

No Copyright
The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission. See Other Information below.

Other Information
In no way are the patent or trademark rights of any person affected by CC0, nor are the rights that other persons may have in the work or in how the work is used, such as publicity or privacy rights.
Unless expressly stated otherwise, the person who associated a work with this deed makes no warranties about the work, and disclaims liability for all uses of the work, to the fullest extent permitted by applicable law.
When using or citing the work, you should not imply endorsement by the author or the affirmer.


