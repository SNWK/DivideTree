import numpy as np
import os
import time
import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from utilsGAN import *
from models import Generator, Discriminator
from dataGAN.sparse_molecular_dataset import SparseMolecularDataset
from rewardUtils import calConnectivityReward, getTreeReward
from tqdm import tqdm

printModel = False
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim # 16
        self.m_dim = self.data.atom_num_types # 4
        self.b_dim = self.data.bond_num_types # 2
        self.g_conv_dim = config.g_conv_dim # [256, 512, 1024]
        self.d_conv_dim = config.d_conv_dim # [[128, 64], 128, [128, 64]]
        self.g_repeat_num = config.g_repeat_num # 6
        self.d_repeat_num = config.d_repeat_num # 6
        self.lambda_cls = config.lambda_cls # 1
        self.lambda_rec = config.lambda_rec # 1
        self.lambda_gp = config.lambda_gp # 10
        self.post_method = config.post_method 

        self.metric = 'validity,sas'

        # Training configurations.
        self.batch_size = config.batch_size # 16
        self.num_iters = config.num_iters # 200000
        self.num_iters_decay = config.num_iters_decay # 100000
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout # 0.0
        self.n_critic = config.n_critic # 5
        self.beta1 = config.beta1 # 0.5
        self.beta2 = config.beta2 # 0.99
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        # self.data.vertexes
        self.G = Generator(self.g_conv_dim, self.z_dim, # 16
                           20,
                           self.data.bond_num_types, # edges type 
                           self.data.atom_num_types, # 6-d vector
                           self.dropout)
        self.D = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)
        self.V = Discriminator(self.d_conv_dim, self.m_dim, self.b_dim, self.dropout)

        self.g_optimizer = torch.optim.Adam(list(self.G.parameters())+list(self.V.parameters()),
                                            self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        if printModel:
            self.print_network(self.G, 'G')
            self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_( len(out.size())-1, labels.unsqueeze(-1), 1.)
        # scatter_(input, dim, index, src) 将src中数据根据index中的索引按照dim的方向填进input
        # len(out.size())-1 = 3
        # labels.unsqueeze(-1): 0->1,0, 1->0,1
        return out

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def postprocess(self, inputs, method, temperature=1.):

        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1,e_logits.size(-1))
                       / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]


        return delistify([delistify(e) for e in (softmax)])

    def reward(self, A, X):
        rr = 0.
        A_copy = A.cpu().detach().numpy().copy()
        X_copy = X.cpu().detach().numpy().copy()
        
        if A_copy.ndim == 2:
            A_copy = [A_copy]

        rr += 0.6*getTreeReward(A_copy, X_copy)

        rr += 0.4*calConnectivityReward(A_copy)

        return rr.reshape(-1, 1)

    def train(self):

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in tqdm(range(start_iters, self.num_iters)):
            if (i+1) % self.log_step == 0:
                mols, a, x, _, _, _ = self.data.next_validation_batch()
                z = self.sample_z(a.shape[0])
                # print('[Valid]', '')
            else:
                mols, a, x, _, _, _ = self.data.next_train_batch(self.batch_size)
                z = self.sample_z(self.batch_size) # 2*16

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()            # Adjacency.
            x = torch.from_numpy(x).to(self.device).float()            # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = x
            z = torch.from_numpy(z).to(self.device).float()

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images. features_real 16*64
            logits_real, features_real = self.D(a_tensor, None, x_tensor)
            d_loss_real = - torch.mean(logits_real)

            # Compute loss with fake images.
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat) = self.postprocess((edges_logits), self.post_method)
            # to be symetric
            edges_hat = (edges_hat + edges_hat.permute(0,2,1,3))/2
            # nodes_hat = nodes_logits
            logits_fake, features_fake = self.D(edges_hat, None, nodes_logits)
            d_loss_fake = torch.mean(logits_fake)

            # Compute loss for gradient penalty.
            eps = torch.rand(logits_real.size(0),1,1,1).to(self.device)
            x_int0 = (eps * a_tensor + (1. - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1. - eps.squeeze(-1)) * nodes_logits).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, None, x_int1)
            # print(grad0.shape, x_int0.shape, grad1.shape, x_int1.shape)
            # print(x_int0)
            d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1)

            # Backward and optimize.
            d_loss = d_loss_fake + d_loss_real + self.lambda_gp * d_loss_gp
            self.reset_grad()
            
            d_loss.backward() 
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Z-to-target
                edges_logits, nodes_logits = self.G(z)
                # Postprocess with Gumbel softmax
                (edges_hat) = self.postprocess((edges_logits), self.post_method)
                logits_fake, features_fake = self.D(edges_hat, None, nodes_logits)
                g_loss_fake = - torch.mean(logits_fake)

                # Real Reward
                rewardR = torch.from_numpy(self.reward(a, x)).to(self.device)
                # Fake Reward
                (edges_hard) = self.postprocess((edges_logits), self.post_method)
                edges_hard = (edges_hard + edges_hard.permute(0,2,1,3))/2
                edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], nodes_logits
                # mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
                #         for e_, n_ in zip(edges_hard, nodes_hard)]
                rewardF = torch.from_numpy(self.reward(edges_hard, nodes_hard)).to(self.device)
                # rewardF = rewardR
                # Value loss
                value_logit_real,_ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
                value_logit_fake,_ = self.V(edges_hat, None, nodes_logits, torch.sigmoid)
                g_loss_value = torch.mean((value_logit_real.float() - rewardR.float()) ** 2 + (
                                           value_logit_fake.float() - rewardF.float()) ** 2)
                # g_loss_value = torch.mean((value_logit_real.float() ) ** 2 + (
                #                            value_logit_fake.float() ) ** 2)
               
                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_value
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_value'] = g_loss_value.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)

                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                V_path = os.path.join(self.model_save_dir, '{}-V.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                torch.save(self.V.state_dict(), V_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / 10) #float(self.num_iters_decay))
                d_lr -= (self.d_lr / 10) #float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)

        with torch.no_grad():
            # mols, a, x, _, _, _ = self.data.next_test_batch()
            z = self.sample_z(1)
            z = Variable(torch.from_numpy(z)).to(self.device).float()
            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            edges_hat = (edges_hard + edges_hard.permute(0,2,1,3))/2
            A = torch.max(edges_hat, -1)[1]
            # print(A.data.cpu().numpy())
            # print(nodes_logits.data.cpu().numpy())
            drawTree(A.data.cpu().numpy(), nodes_logits.data.cpu().numpy()[0])
            # logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
            # g_loss_fake = - torch.mean(logits_fake)

            # # Fake Reward
            # (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), 'hard_gumbel')
            # edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
            # mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            #         for e_, n_ in zip(edges_hard, nodes_hard)]

            # # Log update
            # m0, m1 = all_scores(mols, self.data, norm=True)     # 'mols' is output of Fake Reward
            # m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
            # m0.update(m1)
            # for tag, value in m0.items():
            #     log += ", {}: {:.4f}".format(tag, value)

import matplotlib as mpl
import matplotlib.pyplot as plt

def drawTree(edges, nodes):
    X = []
    Y = []
    length = 20
    for i in range(length):
        for j in range(length):
            if edges[i][j] == 1:
                X.append([nodes[i][0], nodes[j][0]])
                Y.append([nodes[i][1], nodes[j][1]])

    for i in range(len(X)):
        plt.plot(X[i], Y[i], color='r')   

    plt.savefig('test.png')
