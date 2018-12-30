# FEUP Monument Recognizer

This Project Recognizes Porto Monuments with deep learning technics: Ponte da Arrabida, Casa da Musisca, Casa de Serralves, Torre dos Clerigos e Camara Municipal

## Setup Windows
#### Might work for ubuntu with adaptation
1. Install python version 3.6.7 - [here](https://www.python.org/downloads/release/python-367/)
2. Download latest graphic card drivers [here](https://www.nvidia.com/Download/index.aspx?lang=en-us) and Nvidia GeForce Experience  [here](https://www.geforce.com/geforce-experience/download)
3. Install Cuda Toolkit 9.0 - [here](https://developer.nvidia.com/cuda-90-download-archive)
4. Download and extract cudaCNN to somewhere in the pc- [cudaCNN](https://developer.nvidia.com/cudnn)
5. Add to PATH environment variable the link to bin of cudaCNN extracted folder - ``...\cuda\bin``
6. ```python -m pip install --upgrade pip```
7.  ```pip install keras ```
8. ```pip install keras ```
9. ```pip install tensorflow```
10.  ```pip install tensorflow-gpu```
11.  ```pip install pillow```
12. ```pip install matplotlib``` 
13. ```pip install scikit-learn```
14. ```sudo apt-get install python-opencv```

## Build / Run
1. Download and extract the dataset repository from this [link](https://drive.google.com/file/d/1DXmnP-Cl2E0a4B1qFzxtC8YaS2ebH5xA/view), and place the 'data' folder in this project root directory
1. Run 'python train.py', this will compile, train and save the weights.
1. Run 'python validate.py <image path>' this will evaluate an image with the model, displaying the class and probability
  
## Pre Processing

Before running any of the bellow options, a folder with the necessary data containing a subfolder for annotations and images. Each of these subfolders must contain a subfolder for each class and inside the necessary files. The annotation files must be in xml with information about xmin, ymin, width and height of the bounding box of a specific image. The pair (image,annotation) files must have the same name (without considering the extension).
  
An example of the folder 'data' structure:

```
data
├── annotations
│   ├── arrabida
│   ├── camara
│   ├── clerigos
│   ├── musica
│   ├── serralves
└── images
    ├── arrabida
    ├── camara
    ├── clerigos
    ├── musica
    └── serralves

12 directories
```

To join information across data and to allow a generic implementation it is necessary to run the following command:

```python3 generate_dataset_info.py [path_to_data_folder]```

This will join information for each image with its respective annotation. For each class, a ```[class_name]_information.txt``` will be created with the oint information from the image and its annotation in the format: "image_path xmin,ymin,width,height,class_index".These .txt files will be created inside the data folder.

Example: ```data/images/musica/musica-0068.jpg 11,40,259,167,0```

The command will also create a python file ```data_resources.py``` with an array with the classes names, another with the paths to the mentioned .txt files and another with the number of instances in each .txt file. This way, it will be easy to access the data and the classes names.

Example: 

```
classes = ["musica", "arrabida", "clerigos", "camara", "serralves"]

data_info = ["data/musica_information.txt", "data/arrabida_information.txt", "data/clerigos_information.txt", "data/camara_information.txt", "data/serralves_information.txt"]

n_instances_info = ["326", "521", "573", "453", "206"]
```

