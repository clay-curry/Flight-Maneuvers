import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.data_module import MANEUVERS

class SE2_ResNetBlock(nn.Module):
    def __init__(self, num_blk, c_in, k_size, c_out=-1):
        super().__init__()
        # Network representing F
        # conv -> bnorm -> lrelu -> conv -> ... -> bnorm -> lrelu
        self.residual_map = nn.Sequential([
            nn.Conv1d(c_in, c_out, kernel_size=k_size, padding='same', bias=False),
            *[m for _ in range(num_blk - 1) for m in (
            nn.BatchNorm1d(c_out), nn.LeakyReLU(), 
            nn.Conv1d(c_out, c_out, kernel_size=k_size, padding='same', bias=False))],
            nn.BatchNorm1d(c_out), nn.LeakyReLU()]) 
            
    def forward(self, x):
        residuals = self.residual_map(x.reshape(1, -1, x.shape[-1])).reshape(-1, 3, x.shape[-1])
        output = F.leaky_relu(residuals + x)
        return output


class SE2_PreActResNetBlock(nn.Module):
    def __init__(self, n_blk, c_in, k_size, c_out=-1):
        super().__init__()
        # Network representing F
        self.net = nn.Sequential(nn.BatchNorm1d(c_in), nn.LeakyReLU(),
            nn.Conv1d(c_in, c_out, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(c_out), nn.LeakyReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=k_size, padding='same', bias=False))

    def forward(self, x):
        z = self.net(x.reshape(1, -1, x.shape[-1])).reshape(-1, 3, x.shape[-1])
        return z + x


resnet_block_types = {
    "ResNetBlock": SE2_ResNetBlock,
    "PreActResNetBlock": SE2_PreActResNetBlock
}

class SE2_ResNet(LightningModule):

    def __init__(self, num_blocks=[3,3,3], k_size=[3,3,3], c_hidden=[3,3,3], block_type="ResNetBlock", **kwargs):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        assert block_type in block_type
        super().__init__()
        self.save_hyperparameters()
        self._create_network()
        self._init_params()

    def _create_network(self):
        # Creating the ResNet block sc_hidden = self.hparams.c_hidden
        c_out_list = [3 * c for c in self.hparams.c_hidden]
        c_in_list = c_out_list[:-1]
        c_in_list.insert(0, 3)
        h = self.hparams
        self.input_bn = nn.BatchNorm1d(3)
        
        modules = []
        for n_blk, ch_i, k_size, ch_o in zip(h.num_blocks, c_in_list, h.k_size, c_out_list):
            modules.append(resnet_block_types[h.block_type](n_blk, ch_i, k_size, ch_o))
        
        self.conv_blocks = nn.Sequential(*modules)
        self.output_net = nn.Sequential(
            nn.Linear(c_out_list[-1], len(MANEUVERS)))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def common_step(self, batch):
        trajectory, maneuver = batch
        avg_crossentropy_loss = torch.tensor(0.0, dtype=torch.float32)
        for x, target in zip(trajectory, maneuver):
            x = x[..., 3:]
            x = torch.transpose(x, -1, -2).unsqueeze(0).contiguous()
            x = self.input_bn(x)
            x = self.conv_blocks(x)
            x = torch.flatten(x, end_dim=1)
            x = torch.transpose(x, -1, -2)
            logit = self.output_net(x).squeeze(0).log_softmax(-1)
            avg_crossentropy_loss += F.nll_loss(logit, target) / target.shape[0] / 100
        return avg_crossentropy_loss

    def forward(self, x):
        x = torch.transpose(x, -1, -2).unsqueeze(0).contiguous()
        x = self.input_bn(x)
        x = self.conv_blocks(x)
        x = torch.flatten(x, end_dim=1)
        x = torch.transpose(x, -1, -2)
        return self.output_net(x).squeeze(0).softmax(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.lr_scheduler,
            "monitor": "cross entropy"
        }
            
    def training_step(self, batch, batch_idx):
        ce_loss = self.common_step(batch)
        self.log("cross entropy", ce_loss.item(), prog_bar=True)
        return ce_loss
        
    def validation_step(self, batch, batch_idx):
        ce_loss = self.common_step(batch)
        self.log("cross entropy (v)", ce_loss.item(), prog_bar=True)
        return ce_loss
    
    def predict(self, trajectory_df):
        import pandas as pd

        trajectory = torch.from_numpy(trajectory_df[['vx', 'vy', 'vz']].to_numpy())
        joint_dist = self.forward(trajectory.to(torch.float32)).detach().numpy()
        joint_df = pd.DataFrame({
            'takeoff': joint_dist[:, 0],
            'line': joint_dist[:, 1],
            'turn': joint_dist[:, 2],
            'orbit': joint_dist[:, 3],
            'landing': joint_dist[:, 4]
        })
        joint_df['maneuver'] = joint_df.idxmax(axis="columns")
        return joint_df