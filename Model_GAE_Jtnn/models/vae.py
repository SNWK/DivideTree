import torch
import torch.nn as nn
import torch.nn.functional as F
from tree import DTree
from nnutils import create_var, flatten_tensor, avg_pool
from encoder import Encoder
from decoder import Decoder
import sys, os
o_path = os.getcwd()
sys.path.append(o_path)
from helpers.datautils import tensorize

import copy, math

class VAE(nn.Module):

    def __init__(self, hidden_size, latent_size, depthT):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size = latent_size / 2 #Tree and Mol has two vectors
        # embedding = FC: 4->hidden_size
        self.jtnn = Encoder(hidden_size, depthT, nn.Linear(4, hidden_size))
        self.decoder = Decoder(hidden_size, latent_size, nn.Linear(4, hidden_size))

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)

    def encode(self, jtenc_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        return tree_vecs, tree_mess
    
    def encode_from_paths(self, path_list):
        # root, radius
        tree_batch = [DTree(s[0], s[1]) for s in path_list]
        _, jtenc_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _ = self.encode(jtenc_holder)
        return torch.cat([tree_vecs], dim=-1)

    def encode_latent(self, jtenc_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        tree_mean = self.T_mean(tree_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        return torch.cat([tree_mean], dim=1), torch.cat([tree_var], dim=1)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, prob_decode)

    def forward(self, x_batch, beta):
        x_batch, x_jtenc_holder = x_batch
        x_tree_vecs, x_tree_mess = self.encode(x_jtenc_holder)
        z_tree_vecs,tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)

        kl_div = tree_kl 
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)

        return word_loss + topo_loss + beta * kl_div, kl_div.item(), word_acc, topo_acc


    def decode(self, x_tree_vecs, prob_decode):
        #currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0: return None
        elif len(pred_nodes) == 1: return pred_root.feature

        #Mark nid & is_leaf & atommap
        for i,node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)

        scope = [(0, len(pred_nodes))]
        jtenc_holder,mess_dict = Encoder.tensorize_nodes(pred_nodes, scope)
        _,tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict) #Important: tree_mess is a matrix, mess_dict is a python dict

        return None
        
    def dfs_assemble(self, y_tree_mess, x_mol_vecs, all_nodes, cur_mol, global_amap, fa_amap, cur_node, fa_node, prob_decode, check_aroma):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = children
        #todo
        cur_amap = [(fa_nid,a2,a1) for nid,a1,a2 in fa_amap if nid == cur_node.nid]
        cands,aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles,cand_amap = zip(*cands)
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            probs = F.softmax(scores.view(1,-1), dim=1).squeeze() + 1e-7 #prevent prob = 0
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _,cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in xrange(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id,ctr_atom,nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap) #father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(y_tree_mess, x_mol_vecs, all_nodes, cur_mol, new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma)
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol

