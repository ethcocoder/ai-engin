import torch
import matplotlib.pyplot as plt
from model import Autoencoder
from memory import LatentMemory
from reasoning import ReasoningEngine
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import argparse

def unnorm(img):
    """Reverts normalization for visualization"""
    img = img * 0.5 + 0.5
    return torch.clamp(img, 0, 1)

def run_paradox_engine(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Paradox Engine on {device}")

    # 1. Initialize Communication & Compression Layers
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Initialize Cognitive Layers
    memory_bank = LatentMemory(latent_dim=args.latent_dim)
    reasoning_engine = ReasoningEngine(latent_dim=args.latent_dim)

    # 3. Get sample data (simulating physical object inputs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(loader))
    images = images.to(device)
    classes = dataset.classes

    print("\n[*] Executing Latent Compression -> Memory Storage")
    latents = []
    with torch.no_grad():
        for i in range(5):
            latent = model.encoder(images[i].unsqueeze(0))
            concept_name = f"{classes[labels[i]]}_{i}"
            # Store in LatentMemory 
            memory_bank.store(item_id=concept_name, vector=latent, metadata={'label': classes[labels[i]]})
            latents.append(latent)
            print(f"    [+] Stored abstract concept: {concept_name}")
            
    print(f"\n[*] Executing Latent Reasoning (Concept Blending)...")
    concept_a = latents[0]
    concept_b = latents[1]
    name_a = f"{classes[labels[0]]}"
    name_b = f"{classes[labels[1]]}"
    
    # Blending two concepts together mathematically
    blended_latent = reasoning_engine.blend(concept_a, concept_b, weight_a=0.5)
    print(f"    [+] Merged {name_a} and {name_b} into a single vector representation")

    print(f"\n[*] Executing Latent Reasoning (Imagination)...")
    # Imagination (Generating variations via structured exploration)
    # Lowered noise_scale to 0.1 so it doesn't shatter the vector completely
    imagined_latent = reasoning_engine.imagine(concept_a, noise_scale=0.1)
    print(f"    [+] Applied imagination noise to base concept {name_a}")
    
    print("\n[*] Reverting Cognitive Latent Space to Visual Representation (Decoding)...")
    with torch.no_grad():
        decoded_blend = model.decoder(blended_latent).cpu().squeeze(0)
        decoded_imagine = model.decoder(imagined_latent).cpu().squeeze(0)
    
    # 4. Generate Visualization showing the Engine's Thought Process
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(unnorm(images[0].cpu()).permute(1, 2, 0).numpy())
    axes[0].set_title(f"Concept A ({name_a})")
    axes[0].axis('off')

    axes[1].imshow(unnorm(images[1].cpu()).permute(1, 2, 0).numpy())
    axes[1].set_title(f"Concept B ({name_b})")
    axes[1].axis('off')

    axes[2].imshow(unnorm(decoded_blend).permute(1, 2, 0).numpy())
    axes[2].set_title(f"Blended Latent Space\n{name_a} + {name_b}")
    axes[2].axis('off')

    axes[3].imshow(unnorm(decoded_imagine).permute(1, 2, 0).numpy())
    axes[3].set_title(f"Imagined Output\n{name_a} Vector + Noise")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('paradox_engine_demo.png', dpi=300)
    print("\n[*] SUCCESS: Paradox Engine demonstration output saved to 'paradox_engine_demo.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate the full Paradox Cognitive Engine pipeline")
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_autoencoder.pth model weights")
    parser.add_argument('--latent_dim', type=int, default=128, help="Size of the latent vector")
    args = parser.parse_args()
    
    run_paradox_engine(args)
