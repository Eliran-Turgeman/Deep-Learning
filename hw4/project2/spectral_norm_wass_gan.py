import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import project.vanilla_gan as gan_vanila
import project.wass_gan as w_gan
import project.spectral_norm_gan as sn_gan

""" we jsut going 2 use the same class we already rwite"""
class Discriminator(sn_gan.Discriminator):
    def __init__(self, in_size):
        super(Discriminator,self).__init__(in_size)


class Generator(sn_gan.Generator):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        super(Generator,self).__init__(z_dim, featuremap_size, out_channels)

        
def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    return w_gan.discriminator_loss_fn(y_data, y_generated, data_label, label_noise)

def generator_loss_fn(y_generated, data_label=0):
    return w_gan.generator_loss_fn(y_generated, data_label)

def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    return w_gan.train_batch(dsc_model, gen_model, dsc_loss_fn, gen_loss_fn, dsc_optimizer, gen_optimizer, x_data)


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    return gan_vanila.save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file)
