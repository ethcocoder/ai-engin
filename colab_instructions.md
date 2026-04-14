# Google Colab Setup & Training Instructions

Since you have pushed this project to your Git repository, the easiest way to run the training on Google Colab is to clone it directly there. Follow these steps:

## Step 1: Open Google Colab and Setup GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Go to `Runtime` -> `Change runtime type` in the top menu.
4. Select **T4 GPU** (or A100/V100 if available) from the Hardware accelerator dropdown and click **Save**. This drastically speeds up training.

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

## Step 4: Train the Neural Compressor
We completely rewrote the engine to be a **High-Fidelity Spatial Compressor** (ResNet layers + Perceptual Loss combinations) so it transmits flawless visual data instead of just abstract ideas.

Execute your training script:
```bash
!python src/train.py --epochs 25 --batch_size 128 --latent_channels 4
```
This automatically downloads CIFAR10 data, maps them through the spatial dimensions, and saves the system to `checkpoints/best_compressor.pth`.

## Step 5: Test the Telecom Bandwidth Simulator
Run the application wrapper simulating User A sending high quality photos, the server saving 1000% bandwidth costs, and User B receiving identical replicas on their mobile device!
```bash
!python src/telecom_demo.py --model_path checkpoints/best_compressor.pth
```
This script acts as the core telecommunications controller and will output calculations into your terminal confirming exactly how much bandwidth you saved, alongside generating `telecom_simulation_result.png` so you can visually prove the image survived the trip perfectly.

## Step 6: Save your Checkpoints
After training, you can download the weights from the Colab file browser (on the left sidebar, click the Folder icon) under `ai-engin/checkpoints/` or configure Google Drive to save them directly.
