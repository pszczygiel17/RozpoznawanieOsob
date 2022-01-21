# RozpoznawanieOsob

#### 1) Install miniconda on your pc:
- https://docs.conda.io/en/latest/miniconda.html

#### 2) Open 'Anaconda Prompt (miniconda3)'

#### 3) If you want to use another terminal, enter this command in anaconda prompt:
- conda init <name of shell>

#### 4) Create environment for your project and name it 'envFaceRecognition':
- conda create --name envFaceRecognition python=3.6

#### 5) Start env:
- conda activate envFaceRecognition

#### 6) Install tenserflow gpu:
- conda install tensorflow-gpu==2.1.0

#### 7) Install required packages with conda or pip
- pip install <package>
- conda install <package>

#### 8) Run script with command:
- python ./main.py


#### Another commands:

To remove environment:
- conda env remove --name <project-env>

To list created environments:
- conda env list

To list packages installed in selected environment:
- conda activate <name-env>
- conda list
