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

## Step 4: Run the Training Script
We have added a comprehensive `train.py` script. Execute your training script by running:
```bash
!python src/train.py --epochs 20 --batch_size 64 --lr 0.001 --latent_dim 128
```
This will automatically download the CIFAR10 dataset, start the training process, and save the best checkpoint to `checkpoints/best_autoencoder.pth`.

## Step 5: The Paradox Cognitive Engine Demonstration
The main paradigm shift for this project is treating latent vectors as a **cognitive space**. We have integrated the `LatentMemory` and `ReasoningEngine` into a unified pipeline. You can run the entire Paradox Engine demonstration via:
```bash
!python src/paradox_demo.py --model_path checkpoints/best_autoencoder.pth
```
This script acts as the core controller and will:
1. Load dataset samples.
2. Mathematically compress them via the Encoder and store them into `LatentMemory`.
3. Use the `ReasoningEngine` to **blend** two concepts (e.g., Object A + Object B) and **imagine** a new variation of a concept by exploring the latent space.
4. Decode these newly synthesized vector math spaces back into images! 

Open `paradox_engine_demo.png` when it is generated to view the AI's abstract thought process.

## Step 6: Save your Checkpoints
After training, you can download the weights from the Colab file browser (on the left sidebar, click the Folder icon) under `ai-engin/checkpoints/` or configure Google Drive to save them directly.
