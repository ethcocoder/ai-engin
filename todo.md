# TODO: AI Latent Communication Engine

## Setup

- [x] Create project folder
- [x] Initialize Git repository
- [x] Setup Python environment
- [x] Install dependencies (torch, torchvision, numpy)

---

## Data

- [x] Download dataset (CIFAR10 / custom)
- [x] Preprocess images (resize, normalize)
- [x] Create dataloader

---

## Model - Encoder

- [x] Build CNN encoder
- [x] Add convolution layers
- [x] Output latent vector

---

## Model - Decoder

- [x] Build CNN decoder
- [x] Reconstruct image
- [x] Match input size

---

## Training

- [ ] Define loss function (MSE)
- [ ] Setup optimizer (Adam)
- [ ] Train loop
- [ ] Save model checkpoints

---

## Evaluation

- [ ] Visualize reconstructed images
- [ ] Compare original vs output
- [ ] Measure loss

---

## Compression

- [ ] Reduce latent size
- [ ] Test multiple dimensions
- [ ] Store latent as file
- [ ] Measure size in KB

---

## Communication

- [ ] Serialize latent vector
- [ ] Send via file or API
- [ ] Decode on receiver

---

## Optimization

- [ ] Add quantization
- [ ] Reduce model size
- [ ] Speed up inference

---

## Advanced

- [ ] Implement VAE
- [ ] Add perceptual loss
- [ ] Try different architectures

---

## Deployment

- [ ] Export to ONNX
- [ ] Convert to C/C++
- [ ] Build simple CLI tool

---

## Stretch Goals

- [ ] Video encoding
- [ ] Text-image hybrid encoding
- [ ] AI-to-AI protocol system

---

## Notes

- Focus on working system first
- Do not over-optimize early
- Test frequently
