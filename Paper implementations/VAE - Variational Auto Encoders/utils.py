import matplotlib.pyplot as plt
import torch


def save_generated_images(model, epoch, device, z_dim):
    model.eval()

    with torch.inference_mode():
        z = torch.randn(25, z_dim).to(device)

        dec = model.decoder_fc(z)
        dec = dec.reshape(dec.size(0), 64, 7, 7)
        generated_images = model.decoder(dec).cpu()

        plt.figure(figsize=(12, 12))
        for i, img in enumerate(generated_images):
            plt.subplot(5, 5, i+1)

            img = img.squeeze().numpy()

            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.savefig(f'generated_images_epoch{epoch:03d}.png')
        plt.close()