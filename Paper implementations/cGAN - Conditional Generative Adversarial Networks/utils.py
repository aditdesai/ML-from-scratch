import matplotlib.pyplot as plt
import torch

def save_generated_images(gen, epoch, device, z):
    gen.eval()

    with torch.inference_mode():
        noise = torch.randn(25, z, 1, 1).to(device)
        labels = torch.randint(1, 10, size=(25, 1)).to(device)
        generated_images = gen(noise, labels).cpu()

        plt.figure(figsize=(10, 10))
        for i, img in enumerate(generated_images):
            plt.subplot(5, 5, i+1)

            img = img.squeeze().numpy()

            plt.imshow(img, cmap='gray')
            plt.title(labels[i][0])
            plt.axis('off')

        plt.savefig(f"generated_images_epoch{epoch:03d}.png")
        plt.close()