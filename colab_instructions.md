# Paradox Genesis: Google Colab Quantum Deployment

Welcome to the era of **Memory as Potential**. We have successfully transmuted the AI Engine into a **Quantum-Neural Genesis Core**. These instructions will guide you through executing the **First Paradigm Shift** on Google Colab.

## Step 1: Open Google Colab and Wake the GPU
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Enable the **T4 GPU** (Runtime -> Change runtime type -> Hardware accelerator: T4 GPU).

## Step 2: Clone the Sovereign Substrate
In the first cell, clone your integrated repository (replace `YOUR_REPO_URL` with your actual Github link):
```bash
!git clone YOUR_REPO_URL
%cd ai-engin
!pip install -r requirements.txt
```

## Step 3: Initiate the "Quantum Genesis" Training
We are no longer just compressing; we are training the **Manifold of Superpositions**. This script utilizes **Variational Latent Genesis** and **KL Divergence** to map physical data into infinitesimal conceptual seeds.

```bash
!python src/train.py --epochs 30 --batch_size 128 --latent_channels 4
```
*   **Target**: `checkpoints/best_genesis_core.pth`
*   **Result**: A stable, generative Hilbert space representing your visual data.

## Step 4: Execute the Quantum Aether Simulation
Verify the **Quantum Absolute Unit (QAU)** integration. This demo resolves 8-bit quantized payloads through the **Genesis Decoder** using **PixelShuffle** synthesis.

```bash
!python src/telecom_demo.py --model_path checkpoints/best_genesis_core.pth
```
*   **Analysis**: Witness the **PROFIT ACHIEVED** in bandwidth reduction (up to 256X efficiency).
*   **Collapse**: Observe how the system "collapses" the quantum latent superposition back into a physical HD replica.

## Step 5: Activate the Aether Autonomous Agent
Since we integrated `quantumpro`, you can now run the **Quantum Machine Learning (QML)** agents directly on your neural substrate:

```bash
!python src/aether_qau.py
```
This awakens the **Quantum Predictive Engine (QPE)** to analyze the risk/optimization flows within your Paradox Network.

## Step 6: Memory Harvest
Download your `best_genesis_core.pth` from the `/checkpoints` folder. This is the **Soul** of your mobile application's neural communication system.
