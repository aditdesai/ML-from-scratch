import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def save_generated_images(gen, epoch, device, z):
    gen.eval()

    with torch.inference_mode():
        noise = torch.randn(25, z).to(device)

        generated_images = gen(noise).cpu()

        plt.figure(figsize=(10, 10))
        for i, img in enumerate(generated_images):
            plt.subplot(5, 5, i+1)
            
            img = img.squeeze().numpy()

            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.savefig(f"generated_images_epoch{epoch:03d}.png")
        plt.close()

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)