# Paradox Genesis: Google Colab HD-Native Training

The Quantum-Neural Engine has been upgraded to **HD-Native architecture**. We are no longer limited to low-res 32x32 samples. The system can now learn the **structural building blocks** of High Definition imagery using **Perceptual VGG Loss**.

---

## Step 1: Open Google Colab and Wake the GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Enable the **T4 GPU** (Runtime -> Change runtime type -> Hardware accelerator: T4 GPU).

---

## Step 2: Push Upgraded Soul to Colab
Clone your repository and install requirements.

```bash
!git clone YOUR_REPO_URL
%cd ai-engin
!pip install -r requirements.txt
```

---

## Step 3: Initiate "HD Genesis" Texture Training
This step uses **Perceptual Feature Matching** to learn how HD images are "built". It pulls four 1024x1024 samples from the web and optimizes the core to compress them by **~96x** without losing structural integrity.

```bash
!python src/train_hd.py --epochs 150 --latent_channels 8 --batch_size 4
```
*   **Target**: `checkpoints/hd_genesis_core.pth`
*   **Result**: A model that understands high-frequency textures (hair, edges, sky).

---

## Step 4: Execute the HD Visual Simulation
Verify the results on 256x256 High Definition reconstructions.

```bash
!python src/demo_hd.py --model_path checkpoints/hd_genesis_core.pth --latent_channels 8
```
*   **Metrics**: Look for **PSNR > 25dB** on HD images.
*   **Aesthetics**: Unlike the CIFAR models, this will preserve "HD sharpness" even at high reduction levels.

---

## Step 5: Advanced — Cross-Domain Memory Evolution (Optional)
If you want to train on native CIFAR10 but with HD-Ready 4-stage logic:

```bash
!python src/train.py --epochs 80 --batch_size 128 --latent_channels 8
```

---

## Summary of HD Upgrades
| Feature | Logic |
|---|---|
| **4-Stage Manifold** | Enables 16x spatial reduction (96x-192x compression on HD) |
| **Perceptual VGG Loss** | Model learns "texture" and "meaning" instead of just color |
| **ResBlock v2** | Bias-free convolutions for ultra-stable HD gradient flows |
| **Auto-HD Data** | Pipeline dynamically pulls HD test data from the mesh |
