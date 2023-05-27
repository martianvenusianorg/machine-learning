Step1: Install Xcode Command Line Tools
```
xcode-select --install
```

Step2: Install Miniforge
```
conda config --set auto_activate_base false
```

Step3: Create a virtual environment
```
conda create --name mlp python=3.8
conda activate mlp
```

Step4: 

Installing Tensorflow-MacOS
```
conda install -c apple tensorflow-deps
```

Install base TensorFlow:
```
pip install tensorflow-macos
```

Install metal plugin:
```
pip install tensorflow-metal
```


Step5: Install Jupyter Notebook & Pandas
```
conda install -c conda-forge -y pandas jupyter
```

Step6: Run a Benchmark by training the MNIST dataset

Let’s install Tensorflow Datasets
```
pip install tensorflow_datasets
```

