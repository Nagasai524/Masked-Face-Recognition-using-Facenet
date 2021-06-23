# Masked-Face-Recognition-using-MTCNN-and-FaceNet (End-to-End-Deploymnet-using-Streamlit)
Existing Face Recognition systems fail in recognizing the persons when they are masked but I have developed a face recognition system that is capable of recognizing persons even when they are masked with different kinds of occlusion such as Facemasks, templates, Handkerchiefs, and so on. I have even deployed my model using Streamlit.

## Entire Workflow of the project
  ![flowhart](https://user-images.githubusercontent.com/40739974/123039188-4d374d80-d40f-11eb-852d-886a05cfd8d8.jpg)

## How to use the code in the repository
### 1. Collect images of persons that are to be recognized using the model. The images that you collect should contain various kinds of occlusions. For good accuray, make sure that the images that you collect are as follows <br/>
     ```
     a. Non masked face images
     b. Non masked face images with Spectacles or Goggles
     c. Masked face with any kind of masks such as N-95, Cloth or surgical masks
     d. Masked face with any kind of mask such as N-95, Cloth or surgical mask along with spectacles or goggles.
     e. Masked face with a handkerchief or any cloth or scarf.
     f. Masked face with a handkerchief or any cloth or scarf when you are wearing either Spectacles or Goggles.
     ```
 ![Picture1](https://user-images.githubusercontent.com/40739974/123040082-cd11e780-d410-11eb-9df2-39fa90eb8f6e.png)
  The images of every individual person should be placed in a seperate directory. The directory structure of the images that you have collected should be as follows
  ![image](https://user-images.githubusercontent.com/40739974/123040172-f3378780-d410-11eb-8f90-5bb917f98a51.png)

### 2. Run the file *'embeddings_generator.py'*.</br>
   <ul> 
   <li> Make sure that you have installed the MTCNN package before running this file. You can download the package by using the command 'pip install mtcnn'. </li>
   <li> Downlaod the Facenet model file from this link https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn and place it in the same directory that contains
     'embeddings_generator.py' file and rename the downloaded file as 'facenet.h5'. </li>
   <li> The modification to be done to the code in 'embeddings_generator.py' file is providing the path to the directory that contain directories of images of each individual. </li> 
   <li> Change the value of the variable 'base_dir' in the first line of the file with your local location of the dataset directory. </li>
   </ul> 
 
### 3. Run the streamlit file *face_recognizer.py* using command line. </br>
To run the streamlit file, move the directory location in the command prompt and enter the command **streamlit run face_recognizer.py** which will open a tab in your default browser by default.
The inteface of the webapp will be as follows
![pro1](https://user-images.githubusercontent.com/40739974/123042894-1fed9e00-d415-11eb-9d60-04b9de7bbcb1.PNG)

Now upload a image into the Webapp.<br/> 
If the image uploaded contains face of a trained person then the name of the person will be displayed if not the person will be recognized as Unknown. <br/>
The result of the model on the image of a trained person is as follows
![pro2](https://user-images.githubusercontent.com/40739974/123124313-33285a00-d465-11eb-8570-e657e1c6015c.PNG)



