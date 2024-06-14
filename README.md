# Automated Mitosis Detection

This repository contains a program that performs automated detection of mitoses in histological images using OpenCV and image processing techniques. The program processes all images in a specified directory, segments the images to detect brown regions (which represent mitoses), and analyzes the results.

## Prerequisites

To run this program, you need to have the following installed:
- Python 3.9 or later
- Miniconda or Anaconda

## Installation

Follow these steps to set up your environment and install the necessary packages.

### Step 1: Install Miniconda

Download and install Miniconda for your platform from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

### Step 2: Create a Conda Environment

Open your terminal and create a new Conda environment with Python 3.9:

```
conda create --name mitosis_env python=3.9
conda activate mitosis_env
```

### Step 3: Install Required Packages

Install the required packages using ```conda``` and ```pip```:

```
conda install numpy matplotlib
pip install imutils opencv-python
```

### Step 4: Add the Conda Environment to Jupyter

To use the Conda environment in Jupyter Notebook, install ```ipykernel``` and add the environment:

```
conda install ipykernel
python -m ipykernel install --user --name=mitosis_env --display-name "Python (mitosis_env)"
```

### Step 5: Run Jupyter Notebook

Start Jupyter Notebook:

```
jupyter notebook
```

## Usage:

- Place your images in the ```mitosis_data_set/``` directory.
- Run the Jupyter Notebook and execute the cells to process the images and display the results.


