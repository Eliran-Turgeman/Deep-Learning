import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        cin = in_size[0]
        cout = in_size[1] * 2
        #channel_list = [in_channels] + [128] + [256]  + [512] + [1024]

        for ci in range(4):
            modules.append(nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=5, stride=2, padding=2))
            modules.append(nn.BatchNorm2d(cout))
            modules.append(nn.LeakyReLU(negative_slope=0.05))
            cin = cout
            cout = cout *2
        modules.append(nn.Sigmoid())
        self.feature_extractor = nn.Sequential(*modules)
        self.classifier = nn.Linear(4 * in_size[1] * in_size[2],1)
        # ========================
    

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        feats = self.feature_extractor.forward(x).view(x.shape[0], -1)
        y = self.classifier(feats)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        cin = 1024
        cout = 1024 // 2
        self.feat_size = featuremap_size

        for ci in range(4):
            modules.append(nn.ConvTranspose2d(in_channels=cin, out_channels=cout, kernel_size=5, stride=2, padding=2, output_padding=1))
            modules.append(nn.BatchNorm2d(cout))
            modules.append(nn.LeakyReLU(negative_slope=0.05))
            cin = cout
            if cin < (1024 // 4):
                modules.append(nn.ConvTranspose2d(in_channels=cin, out_channels=out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
                #modules.append(nn.LeakyReLU(negative_slope=0.05))
                modules.append(nn.Tanh())
                break
            cout = cout // 2
             
        self.reconstructor = nn.Sequential(*modules)
        self.feats = nn.Linear(z_dim, 1024 * (featuremap_size **2))
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            samples = self(torch.randn(n, self.z_dim, device=device))
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        feats =  self.feats(z).reshape(-1, 1024, self.feat_size, self.feat_size)
        x = self.reconstructor(feats)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    noise_half_range = label_noise /2
   
    #(r1 - r2) * torch.rand(a, b) + r2
    noisy_label_data = (label_noise) * torch.rand_like(y_data, device=y_data.device) + data_label - noise_half_range
   # noisy_label_data = torch.nn.init.normal_(y_data.clone(), mean=data_label, std=label_noise)
    noisy_label_gen = (label_noise) * torch.rand_like(y_generated, device=y_generated.device) + 1 - data_label - noise_half_range
    #noisy_label_gen = torch.nn.init.normal_(y_generated.clone(), mean=1-data_label, std=label_noise)
    loss_func = nn.BCEWithLogitsLoss()
    loss_data = loss_func(y_data, noisy_label_data)
   # loss_data = -loss_data
    loss_generated = loss_func(y_generated, noisy_label_gen)
    #loss_generated = -loss_generated
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    label = data_label + torch.zeros_like(y_generated)
    loss_fn = nn.BCEWithLogitsLoss()
    loss =loss_fn(y_generated, label)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    sampels = gen_model.sample(x_data.shape[0])
    data_label = dsc_model(x_data)
    gen_label = dsc_model(sampels.detach())
    dsc_loss = dsc_loss_fn(data_label, gen_label)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    sampels = gen_model.sample(x_data.shape[0], with_grad=True)
    gen_label = dsc_model(sampels)
    gen_loss = gen_loss_fn(gen_label)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ====== 
    from statistics import mean
    early_stopping = False
    if len(dsc_losses) > 3 and mean([dsc_losses[-2], dsc_losses[-3], dsc_losses[-3]]) > dsc_losses[-1] and \
    len(gen_losses) > 3 and mean([gen_losses[-2], gen_losses[-3], gen_losses[-3]]) > gen_losses[-1]:
        early_stopping = True
    
#     if early_stopping:
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved



# def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
#     """
#     Saves a checkpoint of the generator, if necessary.
#     :param gen_model: The Generator model to save.
#     :param dsc_losses: Avg. discriminator loss per epoch.
#     :param gen_losses: Avg. generator loss per epoch.
#     :param checkpoint_file: Path without extension to save generator to.
#     """

#     saved = False
#     checkpoint_file = f"{checkpoint_file}.pt"

#     # TODO:
#     #  Save a checkpoint of the generator model. You can use torch.save().
#     #  You should decide what logic to use for deciding when to save.
#     #  If you save, set saved to True.
#     # ====== YOUR CODE: ======
#     if checkpoint_file is None:
#         print("check1")
#         return saved
#     if not len(gen_losses) > 2
#         print("gen_losses")
#     if not dsc_losses[-1] > dsc_losses[-2]
#         print("dsc_losses")

#     if len(gen_losses) > 2 and  dsc_losses[-1] > dsc_losses[-2] and dsc_losses[-1] > dsc_losses[-2]:
#         print("check2")
#         torch.save(gen_model, checkpoint_file)
#         saved=True
        
#     print("check3")
#     # ========================

#     return saved
