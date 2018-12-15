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

## Build / Run
1. Download and extract the dataset repository from this [link](https://drive.google.com/file/d/1DXmnP-Cl2E0a4B1qFzxtC8YaS2ebH5xA/view), and place the 'data' folder in this project root directory
1. Run 'python train.py', this will compile, train and save the weights.
1. Run 'python validate.py <image path>' this will evaluate an image with the model, displaying the class and probability