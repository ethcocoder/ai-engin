# Google Colab Setup & Training Instructions

Since you have pushed this project to your Git repository, the easiest way to run the training on Google Colab is to clone it directly there. Follow these steps:

## Step 1: Open Google Colab and Setup GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to `Runtime` -> `Change runtime type` in the top menu.
4. Select **T4 GPU** (or A100/V100 if available) from the Hardware accelerator dropdown and click **Save**. This drasticaly speeds up training.

## Step 2: Clone Your Repository in Colab
In the first cell of your notebook, run the following command to clone your repository (replace `YOUR_GIT_REPO_URL` with your actual repo link):
```bash
!git clone YOUR_GIT_REPO_URL
```

## Step 3: Navigate and Install Dependencies
In the next cell, navigate into the downloaded folder and install the dependencies. Assuming your repo folder name is `ai-engin`:
```bash
%cd ai-engin
!pip install -r requirements.txt
```

## Step 4: Run the Training Script
We have added a `train.py` script that utilizes GPU. Execute your training script by running:
```bash
!python src/train.py
```
This will automatically download the CIFAR10 dataset, start the training process for 10 epochs (showing a progress bar), and save checkpoints to a new `checkpoints/` folder.

## Step 5: Save your Checkpoints
After training, you can download the weights from the Colab file browser (on the left sidebar, click the Folder icon) under `ai-engin/checkpoints/` or configure Google Drive to save them directly.
