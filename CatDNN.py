import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset.QMsymex_dataset import QMdataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.gat import GAT
from model.gcn import GCN
from model.ggnn import GGNN
from model.gin import GIN
from model.mpnn import MPNN
from model.rgcn import RGCN
from model.chebynet import ChebyNet
from model.SchNet import SchNet

from script.tensor_info import *
from script.optimizer import Adam_multitask
from script.atomref import atomref

import wandb

# Sweep parameters
hyperparameter_defaults = dict(
    lr=1e-3,
    epoch=300,
    batchsize_train=32,
    batchsize_val=32,
    model='MPNN',
    catalyst=1,
)

all_property = ['gap', 'homo', 'lumo', 'mu', 'alpha', 'r2', 'zpve_kcal', 'U0', 'U', 'H', 'G', 'Cv',
                'sing_E1', 'trip_E1', 'fission', 'delta']
num_target = len(all_property)
target_l = all_property[:num_target]

wandb.init(project='catalyst_QMsymex_MTL_large',
           config=hyperparameter_defaults)

# Config parameters are automatically set by W&B sweep agent
config = wandb.config
if config.catalyst == 1:
    wandb.config.update({'let_catalyst': 4,
                         'gap_catalyst': 1,
                         'ene_catalyst': 1,
                         'alpha_catalyst': 0,
                         'delta_catalyst': 1,
                         })
else:
    wandb.config.update({'let_catalyst': 0,
                         'gap_catalyst': 0,
                         'ene_catalyst': 0,
                         'alpha_catalyst': 0,
                         'delta_catalyst': 0,
                         })

wandb.run.save()


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class LightningNet(pl.LightningModule):
    def __init__(self, config):
        super(LightningNet, self).__init__()
        # self.save_hyperparameters(config)
        if config.model == 'GGNN':
            self.transform = T.Compose([Complete()])
        elif config.model == 'MPNN':
            self.transform = T.Compose([Complete(), T.Distance(norm=False)])
        elif config.model == 'RGCN':
            self.transform = T.Compose([Complete()])
        elif config.model == 'SchNet':
            self.transform = T.Compose([Complete()])
        else:
            self.transform = None

        # dataset
        self.train_dataset = QMdataset(root='data-bin', mode='dev', transform=self.transform).shuffle()
        self.train_dataset.data.y = self.train_dataset.data.y[:, :num_target]
        self.atomref = atomref
        y = self.train_dataset.data.y.clone()
        self.register_buffer('val_mean', y.mean(dim=0))
        self.register_buffer('val_std', y.std(dim=0))
        if config.ene_catalyst == 0:
            y[:, target_l.index('U0'): target_l.index('U0') + 4] -= self.train_dataset.data.energy_ref
        else:
            y[:, target_l.index('U0')] -= self.train_dataset.data.energy_ref[:, 0]
            y[:, target_l.index('U')] = self.train_dataset.data.y[:, target_l.index('U')] - self.train_dataset.data.y[:, target_l.index('U0')]
            y[:, target_l.index('H')] = self.train_dataset.data.y[:, target_l.index('H')] - self.train_dataset.data.y[:, target_l.index('U')]
            y[:, target_l.index('G')] = self.train_dataset.data.y[:, target_l.index('H')] - self.train_dataset.data.y[:, target_l.index('G')]

        self.register_buffer('mean', y.mean(dim=0))
        self.register_buffer('std', y.std(dim=0))
        self.valid_dataset = QMdataset(root='data-bin', mode='valid', transform=self.transform)
        self.valid_dataset.data.y = self.valid_dataset.data.y[:, :num_target]

        # number of NN output
        output_dim = num_target
        if config.let_catalyst in [0, 1]:  # singlet的见paper
            output_dim += 0
        elif config.let_catalyst in [2, 3]:
            output_dim += 1
        elif config.let_catalyst in [4, 5]:
            output_dim += 2

        self.idx_sing = target_l.index('sing_E1')
        if config.model == 'GAT':
            self.model = GAT(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                             node_hidden_dim=128,
                             num_step_prop=6,
                             num_step_set2set=6)
        elif config.model == 'GCN':
            self.model = GCN(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                             node_hidden_dim=128,
                             num_step_prop=6,
                             num_step_set2set=6)
        elif config.model == 'GGNN':
            self.train_dataset.data.edge_attr = self.train_dataset.data.edge_attr.argmax(dim=1) + 1  # one-hot -> scalar, e.g., [1,0,0,0] -> 1
            self.valid_dataset.data.edge_attr = self.valid_dataset.data.edge_attr.argmax(dim=1) + 1
            self.model = GGNN(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                              node_hidden_dim=64,
                              num_step_prop=3)
        elif config.model == 'GIN':
            self.model = GIN(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                             node_hidden_dim=128,
                             num_step_prop=6,
                             num_step_set2set=6)
        elif config.model == 'MPNN':
            self.model = MPNN(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                              node_hidden_dim=48,
                              edge_hidden_dim=48,
                              num_step_message_passing=3,
                              num_step_set2set=6)
        elif config.model == 'ChebyNet':
            self.model = ChebyNet(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                                  node_hidden_dim=128,
                                  polynomial_order=5,
                                  num_step_prop=6,
                                  num_step_set2set=6)
        elif config.model == 'RGCN':
            self.train_dataset.data.edge_attr = self.train_dataset.data.edge_attr.argmax(dim=1) + 1  # one-hot -> scalar, e.g., [1,0,0,0] -> 1
            self.valid_dataset.data.edge_attr = self.valid_dataset.data.edge_attr.argmax(dim=1) + 1
            self.model = RGCN(node_input_dim=self.train_dataset.num_features, output_dim=output_dim,
                              node_hidden_dim=128,
                              num_basis=-1,
                              num_step_prop=6,
                              num_step_set2set=6)
        elif config.model == 'SchNet':
            self.model = SchNet(output_dim=output_dim, hidden_channels=128, num_filters=128, num_interactions=6, readout='mean')

        if config.let_catalyst in [3, 4, 5]:
            self.register_parameter('singlet_C', nn.Parameter(torch.rand([1])))
        if config.alpha_catalyst in [1]:
            self.register_parameter('alpha_C', nn.Parameter(torch.rand([1])))
            self.register_parameter('alpha_k', nn.Parameter(torch.rand([1])))

    def forward(self, data):
        out = self.model(data)

        out_dict = {}
        # singlet triplet catalyst
        if config.let_catalyst == 4:
            A_ = out[:, self.idx_sing + 1]  # replace (1 - (A - 1) ** 2 / A ** 2) by A_
            m_e = out[:, self.idx_sing + 2]
            m_h = out[:, self.idx_sing + 3]
            y_triplet = out[:, self.idx_sing + 0] + A_ * m_e * m_h / (m_e + m_h) * self.singlet_C
            out_dict['trip_E1'] = y_triplet * self.std[target_l.index('trip_E1')] + self.mean[target_l.index('trip_E1')]
        else:
            out_dict['trip_E1'] = out[:, target_l.index('trip_E1')] * self.std[target_l.index('trip_E1')] + self.mean[
                target_l.index('trip_E1')]

        # singlet delta fission catalyst
        if config.delta_catalyst == 1:
            out_dict['delta'] = (
                    out[:, target_l.index('delta')] * self.std[target_l.index('delta')] + self.mean[target_l.index('delta')]).clamp(0)
            out_dict['sing_E1'] = out_dict['delta'] + out_dict['trip_E1']
            out_dict['fission'] = out_dict['delta'] - out_dict['trip_E1']

        # energy catalyst
        if config.ene_catalyst == 1:
            out_dict['U0'] = out[:, target_l.index('U0')] * self.std[target_l.index('U0')] + self.mean[target_l.index('U0')] + data.energy_ref[:, 0]
            out_dict['U'] = out_dict['U0'] + (out[:, target_l.index('U')] * self.std[target_l.index('U')] + self.mean[target_l.index('U')]).clamp(0)
            out_dict['H'] = out_dict['U'] + (out[:, target_l.index('H')] * self.std[target_l.index('H')] + self.mean[target_l.index('H')]).clamp(0)
            out_dict['G'] = out_dict['H'] - (out[:, target_l.index('G')] * self.std[target_l.index('G')] + self.mean[target_l.index('G')]).clamp(0)
        else:
            out_dict['U0'] = out[:, target_l.index('U0')] * self.std[target_l.index('U0')] + self.mean[target_l.index('U0')] + data.energy_ref[:, 0]
            out_dict['U'] = out[:, target_l.index('U')] * self.std[target_l.index('U')] + self.mean[target_l.index('U')] + data.energy_ref[:, 1]
            out_dict['H'] = out[:, target_l.index('H')] * self.std[target_l.index('H')] + self.mean[target_l.index('H')] + data.energy_ref[:, 2]
            out_dict['G'] = out[:, target_l.index('G')] * self.std[target_l.index('G')] + self.mean[target_l.index('G')] + data.energy_ref[:, 3]
        # alpha catalyst
        if config.alpha_catalyst == 1:
            eps = torch.exp(out[:, target_l.index('alpha')])  # represent epsilon - 1
            out_dict['r2'] = (out[:, target_l.index('r2')] * self.std[target_l.index('r2')] + self.mean[target_l.index('r2')]).clamp(0)
            out_dict['alpha'] = ((torch.exp(self.alpha_k)) * eps) * out_dict['r2'] ** (3 / 2) + torch.exp(self.alpha_C)

        # band gap catalyst
        if config.gap_catalyst == 1:
            out_dict['gap'] = out[:, target_l.index('gap')] * self.std[target_l.index('gap')] + self.mean[target_l.index('gap')]
            out_dict['lumo'] = out[:, target_l.index('lumo')] * self.std[target_l.index('lumo')] + self.mean[target_l.index('lumo')]
            out_dict['homo'] = out_dict['gap'] + out_dict['lumo']

        out_ = []
        for target in target_l:
            if target in out_dict:
                out_.append(out_dict[target])
            else:
                out_.append(out[:, target_l.index(target)] * self.std[target_l.index(target)] + self.mean[target_l.index(target)])
        out_ = torch.column_stack(out_)

        return out_

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss_MSE = []
        for i in range(num_target):
            loss_MSE.append(F.mse_loss((pred[:, i] - self.val_mean[i]) / self.val_std[i], (batch.y[:, i] - self.val_mean[i]) / self.val_std[i]))
            self.log(f'loss_{target_l[i]}', loss_MSE[-1], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss_MSE', F.mse_loss((pred - self.val_mean) / self.val_std, (batch.y - self.val_mean) / self.val_std),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss_MSE[batch_idx % num_target]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer=None, optimizer_idx: int = None, optimizer_closure=None,
                       on_tpu: bool = None, using_native_amp: bool = None, using_lbfgs: bool = None):
        optimizer: Adam_multitask
        optimizer.step(loss_idx=batch_idx % num_target, closure=optimizer_closure)

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        val_MAE = F.l1_loss((pred - self.val_mean) / self.val_std, (batch.y - self.val_mean) / self.val_std)
        self.log('val_MAE', val_MAE, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        val_MSE = F.mse_loss((pred - self.val_mean) / self.val_std, (batch.y - self.val_mean) / self.val_std)
        self.log('val_MSE', val_MSE, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for i in range(num_target):
            self.log(f'val_{target_l[i]}', F.l1_loss(pred[:, i], batch.y[:, i]), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return val_MAE

    def configure_optimizers(self):
        optimizer = Adam_multitask(self.model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler, }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=config.batchsize_train, shuffle=True,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=config.batchsize_val,
                          num_workers=0)


if __name__ == '__main__':
    model = LightningNet(config=config)

    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(save_top_k=1,
                        save_last=True,
                        filename='{epoch}-{step}',
                        monitor='val_MSE')]

    logger = WandbLogger(log_model=True)

    trainer = pl.Trainer(logger=logger,
                         max_epochs=config.epoch,
                         gpus=1 if torch.cuda.is_available() else None,
                         callbacks=callbacks,
                         gradient_clip_val=0.4,
                         terminate_on_nan=True,
                         )

    trainer.fit(model)
