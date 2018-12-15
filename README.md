# FEUP Monument Recognizer

This Project Recognizes Porto Monuments with deep learning technics: Ponte da Arrabida, Casa da Musisca, Casa de Serralves, Torre dos Clerigos e Camara Municipal

## Setup
1. Download and extract the dataset repository from this [link](https://drive.google.com/file/d/1DXmnP-Cl2E0a4B1qFzxtC8YaS2ebH5xA/view), and place the 'data' folder in this project root directory
1. Run 'python train.py', this will compile, train and save the weights.
1. Run 'python validate.py <image path>' this will evaluate an image with the model, displaying the class and probability