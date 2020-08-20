# C2TSR-Concurrent-Canada-based-Traffic-Signpost-Recognition-System
C2TSR as abbreviated for Concurrent Canada-based Traﬃc Signposts Recognition is a proposed deep learning model to classify and detect real-time traﬃc signs and signals on the streets of Canada.



Folders and Files:

A.C2TSR
  a. Data
    1)Raw       ----- Contains recorded videos (I have not uploaded any videos due to space limit)
      	i)Frames  ----- Contains frames extracted from the recorded videos (For ex. I have uploaded only two frames)
      	ii)final  ----- Contains finalised frames and their corresponding annotation files 
    2)Test      ----- Contains Vidoes/images to test
      	i)Output  ----- Contains output of the model on test vidoes (I have uploaded output tested on unseen data)
    3)Dataset
      	i)FinalC2TSR.csv  ----- Contains annotation details of all the frames combined a CSV format
      	ii)image dataset  ----- Includes link to google drive where obj.zip (image dataset) is stoded (could not upload here due to space limit)-
                              	https://drive.google.com/file/d/1UF0FfQ9zyp90ZeDizcc6nzPVx54H3qSQ/view?usp=sharing
      
  b. Model
        i)trainedModel  ----- includes link to last trained weight/model i.e. at iteration 80,0000 (due to size, I could not upload on github) 
                               https://drive.google.com/file/d/1CgA_v2lL6bt_UKBGg-TazYa-h8F9jrXR/view?usp=sharing
                              
  c. Scripts
        i)Create_CSV_Dataframe_From_TextFiles.pdf  ----- to create the csv file using all annotation files and automate some manual tasks
        ii)Deploy_UI.pdf                           ----- to create UI for uploading the videos/images from video or capture video from camera and run C2TSR model on it
        iii)Enhancing_Image_Dataset.pdf            ----- to augment image dataset
        iv)Extract_Frames.pdf                      ----- to extract frames from the recorded videos
        v)TrainC2TSR.pdf                           ----- to train C2TSR model
        vi)VisualizeDataset.pdf                    ----- to visualize the dataset
        vii)test_on_images.pdf                     ----- code to test C2TSR model on images
        viii)test_on_videos.pdf                    ----- code to test C2TSR model on videos
        ix)generate_train.py                       ----- to create training list of images
        
  d. miscFiles
        i)Count.txt               ----- to keep the count of frames
        ii)FinalC2TSR.names       ----- includes classes names and used while annotating the images
        iii)obj.data              ----- includes details about training the model such as where to store the trained weight, etc
        iv) obj.names             ----- includes classes names and used while training the C2TSR model
        v)yolov3_custom.cfg       ----- Yolo v3 configuration file
        
  e. Self-contained Setup guide
        i)Setup guide for C2TSR.pdf ----- setup guide
        ii)requirement.txt          ----- installing dependent modules 
        iii)versions.txt            ----- list the different modules and their version
        
B.Project delivery documents
  a. Proposal_200434194_SubySingh.pdf  ----- Proposal document
  b. C2TSR project proposal.pdf        ----- Proposal presentation
  
  Link to google drive using which I trained C2TSR model, it includes all the files/scripts to train the model
  https://drive.google.com/drive/folders/1S7OStCARCD9yDvwgYEUNHp9H9qh55mOq?usp=sharing
  
  Final presentation recording on Youtube at https://youtu.be/GcAlXeLciZc

  C2TSR output video on youtube at  https://youtu.be/l7xEClga5yI

  C2TSR deployment video on youtube at https://youtu.be/jlay6j7ZRp4
