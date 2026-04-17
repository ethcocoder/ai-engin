# Paradox Genesis: Google Colab Quantum Deployment — v2 (Upgraded Engine)

Welcome to the era of **Memory as Potential**. The AI Engine has been upgraded to the **Paradox Genesis Core v2** with:
- ✅ **SSIM Perceptual Loss** — structural fidelity, not just pixel averages
- ✅ **KLD Annealing** — prevents posterior collapse in early training
- ✅ **AdamW + Cosine LR** — better generalization, no plateau oscillations
- ✅ **Wider Genesis Decoder** — 256-channel manifold for richer reconstruction
- ✅ **Gradient Clipping** — stable training from epoch 1
- ✅ **Resolution Fix** — model trains and evaluates on native 32×32 (no mismatch)
- ✅ **PSNR Quality Metric** — objective measurement of reconstruction quality

---

## Step 1: Open Google Colab and Wake the GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Enable the **T4 GPU**: Runtime → Change runtime type → Hardware accelerator: **T4 GPU**

---

## Step 2: Clone the Sovereign Substrate
```bash
!git clone YOUR_REPO_URL
%cd ai-engin
!pip install -r requirements.txt
```

---

## Step 3: Initiate the Upgraded "Quantum Genesis" Training

> **Why 50 epochs?** The upgraded model has a wider decoder (256 channels vs 128 before).
> It needs more epochs to converge — but the result will be dramatically sharper images.

```bash
!python src/train.py \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --latent_channels 4 \
  --kld_warmup 10 \
  --kld_max 0.01
```

**What to watch:**
- `batch_loss` should drop from ~0.5 to ~0.15 over 50 epochs  
- `kld_w` climbs from 0 → 0.01 over the first 10 epochs (annealing)  
- `LR` smoothly decays every epoch (cosine schedule)  
- **Target**: Genesis Loss ≈ 0.12–0.15, Validation Fidelity ≈ 0.13–0.16  
- **Checkpoint saved**: `checkpoints/best_genesis_core.pth`

---

## Step 4: Execute the Quantum Aether Simulation

```bash
!python src/telecom_demo.py --model_path checkpoints/best_genesis_core.pth
```

**What to look for:**
- **PSNR > 25 dB** = good quality reconstruction ✅  
- **PSNR > 28 dB** = excellent ✅✅  
- Decoded images should be **recognizable** with high structural fidelity  
- Compression ratio remains ~192x with 8-bit quantization  

The key fix: the demo now evaluates on native **32×32 CIFAR** images (same resolution as training). The previous version was upscaling to 256×256 at test time, causing catastrophic distribution shift.

---

## Step 5: Activate the Aether Autonomous Agent
```bash
!python src/aether_qau.py
```

---

## Step 6: Memory Harvest
Download `checkpoints/best_genesis_core.pth` — this is the **Soul** of your mobile telecom compression system.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Decoded images are color blobs / noise | You're using old weights — retrain from scratch |
| `CUDA out of memory` | Reduce `--batch_size` to 64 |
| `PSNR < 20 dB` after 50 epochs | Increase epochs to 80 or reduce `--kld_max` to 0.005 |
| Loss not decreasing after epoch 20 | The cosine scheduler keeps going — check epoch 40+ |
