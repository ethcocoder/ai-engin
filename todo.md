# TODO: AI Latent Communication Engine

## Setup

- [ ] Create project folder
- [ ] Initialize Git repository
- [ ] Setup Python environment
- [ ] Install dependencies (torch, torchvision, numpy)

---

## Data

- [ ] Download dataset (CIFAR10 / custom)
- [ ] Preprocess images (resize, normalize)
- [ ] Create dataloader

---

## Model - Encoder

- [ ] Build CNN encoder
- [ ] Add convolution layers
- [ ] Output latent vector

---

## Model - Decoder

- [ ] Build CNN decoder
- [ ] Reconstruct image
- [ ] Match input size

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
