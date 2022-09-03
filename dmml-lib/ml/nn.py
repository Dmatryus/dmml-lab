from typing import Callable

import torch
from torch.utils.data import DataLoader


class NN:
    def __init__(self, model, optimazer, init_function: Callable = None):
        self.model = model
        self.optimazer = optimazer
        self.init_function = init_function

    def init_weights(self):
        if self.init_function:
            self.model.apply(self.init_function)
        else:
            # TODO: warning
            pass


class GanProcessor:
    def __init__(self, generator, discriminator, criterion, len_latent_vector: int, device: torch.device,
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
        def train_discriminator(data):
            self.discriminator.model.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = self.discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            err_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            err_real.backward()
            x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.len_latent_vector, 1, 1, device=device)
            # Generate fake image batch with G
            fake = self.discriminator.model(noise)
            label.fill_(0)
            # Classify all fake batch with D
            output = self.discriminator.model(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            err_fake = self.criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            err_fake.backward()
            d_g_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            err = err_real + err_fake
            # Update D
            self.discriminator.optimazer.step()

        print("Starting Training Loop...")
        for epoch in range(num_epoch):
            for i, data in enumerate(self.dataloader, 0):
                train_discriminator(data)
