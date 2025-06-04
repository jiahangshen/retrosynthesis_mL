import argparse
import json
import numpy as np
import copy
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import DataLoader

from gnn import GNN_graphpred
from prepare_mol_graph import MoleculeDataset
from chemutils import cano_smiles

from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
class DOMLGNN(GNN_graphpred):
    def __init__(self, num_domains,fusion_function, **kwargs):
        super().__init__(fusion_function=fusion_function,**kwargs)
        self.num_domains = num_domains
        self.fusion_function=fusion_function

        self.domain_projector = nn.Sequential(
            nn.Linear(self.emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        domain_classifier_input_dim = 128 
        dc_hidden_dim1 = 128  
        dc_hidden_dim2 = 64   
        dc_dropout_rate = 0.3 
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(domain_classifier_input_dim, dc_hidden_dim1),
                nn.ReLU(),
                nn.Dropout(dc_dropout_rate),
                nn.Linear(dc_hidden_dim1, dc_hidden_dim2),
                nn.ReLU(),
                nn.Dropout(dc_dropout_rate),
                nn.Linear(dc_hidden_dim2, 2)
            ) for _ in range(int(num_domains*(num_domains-1)//2))
        ])
        
        for classifier in self.domain_classifiers:
            nn.init.xavier_uniform_(classifier[0].weight)
            nn.init.zeros_(classifier[0].bias)
        self.grl = GradientReversal.apply
        self.adv_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        #self.domain_temp = nn.Parameter(torch.tensor(0.5)) 
        self.prod_head = self.graph_embedding
        self.react_head = self.graph_embedding
        self.register_buffer('domain_weights', torch.ones(num_domains))


    def get_domain_idx(self, src, tgt):
        i, j = sorted([src, tgt])
        return i * (2 * self.num_domains - i - 1) // 2 + j - i - 1
    
    def forward(self, batch, typed=False, decode=False, beam_size=50):
        node_representation, graph_representation = self.run_gnn(batch, typed)
        
        if decode:
            beam_search_results = super().forward(batch, typed, decode, beam_size)
            outputs = {'beam_nodes': beam_search_results}
        else:
            loss_prod, loss_react,prod_pred_max, react_pred = self.rnn_model(batch, node_representation, graph_representation)
            outputs = {
                'prod_pred_max': prod_pred_max,
                'react_pred': react_pred,
                'loss_prod':loss_prod,
                'loss_react':loss_react
            }
        reversed_rep = self.grl(graph_representation, self.adv_alpha)
        domain_feat = self.domain_projector(reversed_rep)
        outputs['domain_feat'] = domain_feat
        
        if self.training and hasattr(batch, 'domain_labels'):
            assert batch.domain_labels.min() >= 1, "Domain labels should be 1-based"
            batch.domain_labels = batch.domain_labels - 1
            domain_loss = 0
            active_domains = torch.unique(batch.domain_labels).tolist()
            domain_pairs = list(combinations(active_domains, 2))
            if not domain_pairs:
                return outputs
            
            max_pairs = min(40, len(domain_pairs))
            sampled_indices = torch.randperm(len(domain_pairs))[:max_pairs]
            
            for idx in sampled_indices:
                src, tgt = domain_pairs[idx]
                mask = (batch.domain_labels == src) | (batch.domain_labels == tgt)
                if mask.sum() < 2:
                    continue
                
                pair_idx = self.get_domain_idx(src, tgt)
                classifier = self.domain_classifiers[pair_idx]
                logits = classifier(domain_feat[mask])
                labels = (batch.domain_labels[mask] == tgt).long()
                loss = F.cross_entropy(logits, labels)
                weight = 1.0 / (mask.sum().float().sqrt() + 1e-6)
                domain_loss += loss * weight * self.domain_weights[src] * self.domain_weights[tgt]

            
            if len(sampled_indices) > 0:
                domain_loss = domain_loss / len(sampled_indices)
                outputs['domain_loss'] = domain_loss * torch.clamp(self.adv_alpha, 0.0, 1.0)
            else:
                outputs['domain_loss'] = 0.0
                
        return outputs




def train(args, model, device, loader, optimizer=None, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    domain_loss = 0.0
    total_loss_batch=0.0
    prod_pred_res_max, react_pred_res, react_pred_res_each = [], [], []

    for batch in tqdm(loader, desc="Iteration"):
        if batch is None:
            continue
        batch = batch.to(device)
        outputs = model(batch, typed=args.typed)        
        loss_prod = outputs['loss_prod']
        loss_react =outputs['loss_react']
        main_loss = loss_prod + loss_react
        total_loss_batch = main_loss
        if model.training and 'domain_loss' in outputs:
            total_loss_batch += outputs['domain_loss']
            domain_loss += outputs['domain_loss'].item()
        
        prod_pred_max = outputs['prod_pred_max']
        react_pred=outputs['react_pred']
        prod_pred_res_max.extend(prod_pred_max)
        react_pred_res_each.extend(react_pred.reshape(-1, ).tolist())
        for react in react_pred:
            react_pred_res.append(False not in react)        

        if model.training:
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += total_loss_batch.item()
        else:
            total_loss += total_loss_batch.item()
    prod_pred_acc_max = np.mean(prod_pred_res_max)
    react_pred_acc_each = np.mean(react_pred_res_each)
    react_pred_acc = np.mean(react_pred_res)
    
    metrics = {
        'total_loss': total_loss/ len(loader),
        'domain_loss': domain_loss / len(loader) if model.training else 0,
        'prod_pred_acc_max': prod_pred_acc_max,
        'react_pred_acc_each': react_pred_acc_each,
        'react_pred_acc': react_pred_acc
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description='DOML for Chemical Reaction Prediction')
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=6,
                        help='number of GNN message passing layers (default: 6).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="concat",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='data/USPTO50K',
                        help='root directory of dataset.')
    parser.add_argument('--atom_feat_dim', type=int, default=45,
                        help="atom feature dimension.")
    parser.add_argument('--bond_feat_dim', type=int, default=12,
                        help="bond feature dimension.")
    parser.add_argument('--onehot_center', action='store_true', default=False,
                        help='reaction center encoding: onehot or subgraph')
    parser.add_argument('--center_loss_type', type=str, default='ce',
                        help='loss type (bce or ce) for reaction center prediction')
    parser.add_argument('--typed', action='store_true', default=False,
                        help='if given reaction types')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='only evaluate on test data')
    parser.add_argument('--multiprocess', action='store_true', default=False,
                        help='train a model with multi process')
    parser.add_argument('--num_process', type=int, default=4,
                        help='number of processes for multi-process training')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='uspto50k',
                        help='output filename')
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_split', type=str, default='test',
                        help='evaluation test/valid/train dataset')
    parser.add_argument('--beam_size', type=int, default=50,
                        help='beam search size for rnn decoding')
    parser.add_argument('--num_domains', type=int, default=10,
                       help='Number of domains for adversarial training')
    parser.add_argument('--adv_weight', type=float, default=0.3,
                       help='Weight for domain adversarial loss')
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    if args.test_only:
        assert args.eval_split in ['test', 'valid', 'train']
        test_dataset = MoleculeDataset(args.dataset, split=args.eval_split, load_mol=True)
        prod_word_size = len(test_dataset.prod_smarts_fp_list)
        react_word_size = len(test_dataset.react_smarts_list)
    else:
        train_dataset = MoleculeDataset(args.dataset, split='train', load_mol=True)
        valid_dataset = MoleculeDataset(args.dataset, split='valid', load_mol=False)
        prod_word_size = len(train_dataset.prod_smarts_fp_list)
        react_word_size = len(train_dataset.react_smarts_list)

    if args.typed:
        args.atom_feat_dim += 10
        args.filename += '_typed'

    model = DOMLGNN(
        num_domains=args.num_domains,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        atom_feat_dim=args.atom_feat_dim,
        bond_feat_dim=args.bond_feat_dim,
        center_loss_type=args.center_loss_type,
        fusion_function=0,
        prod_word_size=prod_word_size,
        react_word_size=react_word_size,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        graph_pooling=args.graph_pooling
    )

    param_groups = [
        {'params': model.gnn.parameters(), 'lr': args.lr, 'weight_decay': args.decay},
        {'params': model.gnn_diff.parameters(), 'lr': args.lr, 'weight_decay': args.decay},
        {'params': model.domain_projector.parameters(), 'lr': args.lr*0.1, 'weight_decay': args.decay},
        {'params': model.domain_classifiers.parameters(), 'lr': args.lr, 'weight_decay': args.decay},
        #{'params': [model.adv_alpha, model.domain_temp], 'lr': args.lr*0.01, 'weight_decay': 0}
    ]

    optimizer = optim.AdamW(param_groups)
    del model.gnn_diff
    del model.scoring
    model.to(device)

    dataset = os.path.basename(args.dataset)
    args.filename = os.path.join('logs', dataset, args.filename)
    os.makedirs(args.filename, exist_ok=True)
    if not args.input_model_file == "":
        input_model_file = os.path.join(args.filename, args.input_model_file)
        model.from_pretrained(input_model_file, args.device)
        print("load model from:", input_model_file)

    if args.test_only:
        print("evaluate on test data only")
        if args.multiprocess:
            eval_multi_process(args, model, device, test_dataset)
        else:
            eval_decoding(args, model, device, test_dataset)
        exit(1)

    if args.multiprocess:
        mp.set_start_method('spawn', force=True)
        model.share_memory() 
        processes = []
        output_model_files = []
        for rank in range(args.num_process):
            output_model_files.append(os.path.join(args.filename, 'model_{}.pt'.format(rank)))
            p = mp.Process(
                target=train_multiprocess,
                args=(rank, args, model, device, train_dataset, valid_dataset)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    else:
        current_time = datetime.now()
        #time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_model_file = os.path.join(args.filename, "last_model_1.pt")
        print('output_model_file:', output_model_file)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print(optimizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch))
            ret = train(args, model, device, train_loader, optimizer)
            print(ret)
            scheduler.step()
            torch.save(model.state_dict(), output_model_file)
def eval_decoding(args, model, device, dataset, save_res=True, k=0):
    model.eval()
    pred_results = {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            beam_nodes = model(batch, typed=args.typed, decode=True, beam_size=args.beam_size)
            cano_pred_mols = {}
            for node in beam_nodes['beam_nodes']:
                batch_idx = node.index
                data_idx = batch.index[batch_idx]
                if data_idx not in cano_pred_mols:
                    cano_pred_mols[data_idx] = set()
                if data_idx not in pred_results:
                    pred_results[data_idx] = {
                        'rank': 1000,
                        'product': batch.product[batch_idx],
                        'reactant': batch.reactant[batch_idx],
                        'cano_reactants': batch.cano_reactants[batch_idx],
                        'type': batch.type[batch_idx].item(),
                        'seq_gt': batch.sequences[batch_idx],
                        'templates': batch.templates[batch_idx],
                        'templates_pred': [],
                        'templates_pred_log_prob': [],
                        'reactants_pred': [],
                        'seq_pred': [],
                    }
                product = pred_results[data_idx]['product']
                seq_pred = node.targets_predict
                prod_smarts_list = []
                for i, cand in enumerate(batch.reaction_center_cands[batch_idx]):
                    if cand == seq_pred[0]:
                        prod_smarts_list.extend(batch.reaction_center_cands_smarts[batch_idx][i])
                prod_smarts_list = set(prod_smarts_list)
                assert len(prod_smarts_list)
                seq_pred[1:] = [tp for tp in seq_pred[1:] if tp < len(dataset.react_smarts_list)]
                decoded_results = dataset.decode_reactant_from_seq(product, seq_pred, prod_smarts_list, keep_mapnums=True)
                for decoded_result in decoded_results:
                    pred_tmpl, pred_mols = decoded_result
                    for pred_mol in pred_mols:
                        cano_pred_mol = cano_smiles(pred_mol)
                        if cano_pred_mol not in cano_pred_mols[data_idx]:
                            cano_pred_mols[data_idx].add(cano_pred_mol)
                            pred_results[data_idx]['templates_pred_log_prob'].append(node.log_prob.item())
                            pred_results[data_idx]['templates_pred'].append(pred_tmpl)
                            pred_results[data_idx]['reactants_pred'].append(pred_mol)
                            pred_results[data_idx]['seq_pred'].append(seq_pred)
                            if pred_results[data_idx]['cano_reactants'] == cano_pred_mol:
                                pred_results[data_idx]['rank'] = min(pred_results[data_idx]['rank'], len(pred_results[data_idx]['seq_pred']))

            beam_nodes.clear()
    domain_results = defaultdict(lambda: {'total':0, 'correct':0})
    for data_idx, res in pred_results.items():
        true_domain = dataset.domain_labels[int(data_idx)]
        domain_results[true_domain]['total'] += 1
        if res['rank'] == 1:
            domain_results[true_domain]['correct'] += 1
    
    print("\nDomain-wise Accuracy:")
    for dom in sorted(domain_results.keys()):
        acc = domain_results[dom]['correct']/domain_results[dom]['total']
        print(f"Domain {dom}: {acc:.2%} ({domain_results[dom]['correct']}/{domain_results[dom]['total']})")



def eval_multi_process(args, model, device, test_dataset):
    model.eval()
    data_chunks = []
    chunk_size = len(test_dataset.processed_data_files) // args.num_process + 1
    print('total examples to evaluate:', len(test_dataset.processed_data_files))
    for i in range(0, len(test_dataset.processed_data_files), chunk_size):
        data_chunks.append(test_dataset.processed_data_files[i:i + chunk_size])
        print('chunk size:', len(data_chunks[-1]))

    mp.set_start_method('spawn')
    model.share_memory()
    processes = []
    results = []
    for k, data_files in enumerate(data_chunks):
        test_dataset.processed_data_files = data_files
        res_file = os.path.join(args.filename, 'beam_result_{}.json'.format(k))
        results.append(res_file)
        p = mp.Process(
            target=eval_decoding,
            args=(args, model, device, test_dataset, True, k)
        )
        # We first train the model across `num_process` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pred_results = {}
    for res_file in results:
        pred_results.update(json.load(open(res_file)))
    with open(os.path.join(args.filename, 'beam_result_{}.json'.format(args.eval_split)), 'w') as f:
        json.dump(pred_results, f, indent=4)

def train_multiprocess(rank, args, model, device, train_dataset, valid_dataset):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    epochs = args.epochs // args.num_process
    output_model_file = os.path.join(args.filename, 'model.pt')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset_ = copy.deepcopy(valid_dataset)
    valid_dataset_.processed_data_files = valid_dataset_.processed_data_files_valid
    val_loader = DataLoader(valid_dataset_, batch_size=args.batch_size, shuffle=False)
    for epoch in range(1, epochs + 1):
        print("====rank and epoch: ", rank, epoch)
        train_res = train(args, model, device, train_loader, optimizer)
        log = "rank: %d epoch: %d train_loss: %f loss_prod: %f loss_react: %f " \
              "prod_pred_acc_max: %f react_pred_acc_each: %f react_pred_acc: %f" % (rank, epoch, *train_res)
        print(log)
        scheduler.step()
        if rank == 0:
            torch.save(model.state_dict(), output_model_file)
        print("====evaluation")
        val_res = train(args, model, device, val_loader, train=False)
        log = "rank: %d epoch: %d val_loss: %f loss_prod: %f loss_react: %f " \
              "prod_pred_acc_max: %f react_pred_acc_each: %f react_pred_acc: %f" % (rank, epoch, *val_res)
        print(log)
if __name__ == "__main__":
    main()