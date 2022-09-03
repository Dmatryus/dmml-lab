import pickle
import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


class NN:
    def __init__(self, model, optimazer, init_function: Callable = None):
        self.model = model
        self.optimazer = optimazer
        self.init_function = init_function

    def init_weights(self):
        if self.init_function:
            self.model.apply(self.init_function)
        else:
            warnings.warn("Init function is None. Weights was not init.")


class GanProcessor:
    def __init__(self, generator: NN, discriminator: NN, criterion, len_latent_vector: int, device: torch.device,
                 dataloader: DataLoader):
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = criterion
        self.len_latent_vector = len_latent_vector
        self.device = device
        self.dataloader = dataloader

        self.img_list = []
        self.generator_losses = []
        self.discriminator_losses = []
        self.iters = 0

    def train(self, num_epoch):
        print("Starting Training Loop...")
        fixed_noise = torch.randn(64, self.len_latent_vector, 1, 1, device=self.device)
        for epoch in range(num_epoch):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.model.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator.model(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.len_latent_vector, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.discriminator.model(noise)
                label.fill_(0)
                # Classify all fake batch with D
                output = self.discriminator.model(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.discriminator.optimazer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.model.zero_grad()
                label.fill_(1)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator.model(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.generator.optimazer.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epoch, i, len(self.dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.generator_losses.append(errG.item())
                self.discriminator_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 500 == 0) or ((epoch == num_epoch - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator.model(fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                self.iters += 1

    def train_plot(self, figsize=(10, 5)):
        plt.figure(figsize=figsize)
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.generator_losses, label="Generator")
        plt.plot(self.discriminator_losses, label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def result_evolution(self, figsize=(8, 8), interval=1000, repeat_delay=1000):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.img_list]
        return animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    def to_picle(self, save_path: Path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)


