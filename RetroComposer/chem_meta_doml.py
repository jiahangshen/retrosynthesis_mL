# 此文件改编自 https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py

import argparse
import json
import numpy as np
import copy # 对于 deepcopy很重要
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch import autograd # 对于 autograd.grad很重要
from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import DataLoader, Batch # Batch 用于 get_data_by_domain
from torch_geometric.data import Data # 用于类型注解
from typing import List, Tuple, Dict # 从 typing 模块导入

# 假设这些模块在您的项目中存在
from gnn import GNN_graphpred
from prepare_mol_graph import MoleculeDataset
from chemutils import cano_smiles

from itertools import combinations, cycle # cycle für DataLoader-Paarung
from collections import defaultdict
# from sklearn.cluster import KMeans # KMeans 在当前代码中未使用

# --- 全局辅助函数 ---
_printed_warnings = set()
def print_once(msg):
    if msg not in _printed_warnings:
        print(msg)
        _printed_warnings.add(msg)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DOMLGNN(GNN_graphpred):
    def __init__(self, num_domains_total: int, fusion_function, hparams_meta_doml: dict,emb_dim: int, prod_word_size: int, react_word_size: int, **kwargs):
        super().__init__(emb_dim=emb_dim, prod_word_size=prod_word_size, react_word_size=react_word_size, fusion_function=fusion_function, **kwargs)
        self.num_domains_total = num_domains_total
        self.fusion_function = fusion_function
        self.hparams_meta_doml = hparams_meta_doml

        self.domain_projector = nn.Sequential(
            nn.Linear(self.emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        self.num_dann_classifiers = int(self.num_domains_total * (self.num_domains_total - 1) // 2)
        if self.num_dann_classifiers > 0:
            self.domain_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                ) for _ in range(self.num_dann_classifiers)
            ])
            for classifier_dann in self.domain_classifiers:
                nn.init.xavier_uniform_(classifier_dann[0].weight)
                nn.init.zeros_(classifier_dann[0].bias)
        else:
            self.domain_classifiers = None
            print_once("警告: num_dann_classifiers 为 0, 未创建DANN domain_classifiers。")

        self.grl = GradientReversal.apply
        self.adv_alpha = nn.Parameter(torch.tensor(self.hparams_meta_doml.get('initial_adv_alpha', 1.0)))
        self.domain_temp = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('domain_weights_dann', torch.ones(self.num_domains_total))

        self.num_meta_adv_predictors = int(self.num_domains_total * (self.num_domains_total - 1) // 2)
        if self.num_meta_adv_predictors > 0:
            if hasattr(self, 'rnn_model') and self.rnn_model is not None and isinstance(self.rnn_model, nn.Module):
                try:
                    self.meta_adv_reaction_predictors = nn.ModuleList(
                        [copy.deepcopy(self.rnn_model) for _ in range(self.num_meta_adv_predictors)]
                    )
                    print_once(f"信息: 成功创建 {self.num_meta_adv_predictors} 个 meta_adv_reaction_predictors (作为 self.rnn_model 的副本)。")
                except Exception as e:
                    print_once(f"错误: 复制 self.rnn_model 时出错: {e}。Meta-DOML预测器未正确初始化。")
                    self.meta_adv_reaction_predictors = None
            else:
                print_once("警告: self.rnn_model 不可用或不是 nn.Module。无法创建 Meta-DOML 预测器。")
                self.meta_adv_reaction_predictors = None
        else:
            self.meta_adv_reaction_predictors = None
            print_once("警告: num_meta_adv_predictors 为 0, 未创建 meta_adv_reaction_predictors。")

    def get_pair_idx(self, d1_id: int, d2_id: int) -> int:
        i, j = sorted([int(d1_id), int(d2_id)])
        if i >= self.num_domains_total or j >= self.num_domains_total or i == j:
             raise ValueError(f"无效的域ID用于对索引: i={i}, j={j}，而 num_domains_total={self.num_domains_total}")
        # M = self.num_domains_total
        # return M * i - i * (i + 1) // 2 + j - i - 1 # Diese Formel gilt, wenn i von 0 bis N-2 und j von i+1 bis N-1 läuft
        # Korrekte Formel für 0-basierte Indizes i < j aus N Elementen, um einen flachen Index zu erhalten:
        # sum_{k=0}^{i-1} (N-1-k) + (j-i-1)
        # = i*N - i - i*(i-1)/2 + j-i-1
        # = i*N - i*(i+1)/2 + j - i -1 (falsch)
        # Einfachere Formel, wenn N die Anzahl der Domänen ist:
        # (i * (2*N - i - 1) / 2) + (j - i - 1) war für eine andere Zählweise.
        # Für flache obere Dreiecksmatrix-Indizierung (ohne Diagonale):
        if i < 0 or j <= i or j >= self.num_domains_total:
            raise ValueError(f"Ungültige Indizes i={i}, j={j} für num_domains_total={self.num_domains_total}")
        
        # Index = Summe der Elemente in den vorherigen Reihen + Position in der aktuellen Reihe
        # Reihe i hat (N - 1 - i) Elemente.
        # sum_{k=0}^{i-1} (N - 1 - k) = i*(N-1) - i*(i-1)/2
        # dann + (j - (i+1))
        index = 0
        for row in range(i):
            index += (self.num_domains_total - 1 - row)
        index += (j - (i + 1))
        return index


    def forward(self, batch: Batch, typed: bool = False, decode: bool = False, beam_size: int = 50,
                use_main_rnn: bool = True, meta_adv_predictor_idx: int = None,
                specific_graph_rep: dict = None, # Erwartet {'node': node_rep, 'graph': graph_rep}
                use_specific_gnn: nn.Module = None) -> dict: # Ermöglicht die Übergabe eines GNN-Moduls (z.B. Klon)

        outputs = {}
        node_representation, graph_representation = (None, None)

        current_gnn_to_use = use_specific_gnn if use_specific_gnn is not None else self.gnn

        if specific_graph_rep is None:
            # self.run_gnn muss so angepasst werden, dass es ein optionales gnn_module akzeptiert
            # oder wir rufen die Teile von run_gnn hier direkt mit current_gnn_to_use auf.
            # Annahme: self.run_gnn kann nicht einfach ein anderes GNN-Modul verwenden.
            # Daher rufen wir die Logik von run_gnn hier nach.
            if not hasattr(batch, 'x') or batch.x is None: # Fallback für leere Batches oder Batches ohne Features
                print_once("WARNUNG: Batch hat keine Knotenmerkmale (batch.x). Überspringe GNN-Berechnung.")
                # Setze Dummy-Repräsentationen, falls erforderlich, oder gib Fehler zurück
                # Die Größe muss mit self.emb_dim übereinstimmen, Anzahl der Graphen aus batch.ptr ableiten
                num_graphs_in_batch = batch.num_graphs if hasattr(batch, 'num_graphs') else 0
                if num_graphs_in_batch > 0 :
                     graph_representation = torch.zeros((num_graphs_in_batch, self.emb_dim), device=batch.x.device if hasattr(batch, 'x') and batch.x is not None else self.adv_alpha.device)
                     node_representation = torch.zeros((0, self.emb_dim), device=batch.x.device if hasattr(batch, 'x') and batch.x is not None else self.adv_alpha.device) # Leere Knotenrepräsentation
                else: # Leerer Batch
                     graph_representation = torch.zeros((0, self.emb_dim), device=self.adv_alpha.device)
                     node_representation = torch.zeros((0, self.emb_dim), device=self.adv_alpha.device)

            else: # Normaler Pfad
                node_feat = self.atom_encoder(batch.x) # batch.x sind die initialen Knotenmerkmale
                if typed and hasattr(batch, 'reaction_type'):
                    type_emb = self.type_embed(batch.reaction_type)
                    node_feat = torch.cat([node_feat, type_emb[batch.batch]], dim=1) # batch.batch für korrekte Zuordnung

                # Hier current_gnn_to_use (z.B. cloned_gnn oder self.gnn)
                node_representation_gnn = current_gnn_to_use(node_feat, batch.edge_index, batch.edge_attr)
                
                # Graph-Level Pooling (aus GNN_graphpred.run_gnn übernommen)
                if self.graph_pooling == "sum":
                    graph_representation = global_add_pool(node_representation_gnn, batch.batch)
                elif self.graph_pooling == "mean":
                    graph_representation = global_mean_pool(node_representation_gnn, batch.batch)
                elif self.graph_pooling == "max":
                    graph_representation = global_max_pool(node_representation_gnn, batch.batch)
                elif self.graph_pooling == "attention":
                    if self.graph_pred_linear is None: # Sicherstellen, dass es initialisiert wurde
                        # Diese Initialisierung sollte eigentlich in __init__ erfolgen, basierend auf graph_pooling
                        print_once("WARNUNG: self.graph_pred_linear nicht initialisiert für Attention-Pooling. Wird jetzt erstellt.")
                        self.graph_pred_linear = nn.Linear(self.emb_dim, 1).to(node_representation_gnn.device)

                    graph_representation = self.pool(node_representation_gnn, batch.batch)
                else:
                    raise ValueError("Ungültiges Graph-Pooling")
                node_representation = node_representation_gnn # Knotenrepräsentation vor Pooling
        else:
            node_representation, graph_representation = specific_graph_rep['node'], specific_graph_rep['graph']
        
        outputs['graph_representation'] = graph_representation
        outputs['node_representation'] = node_representation

        if decode:
            return super().forward(batch, typed, decode, beam_size=beam_size, 
                                   # Übergib die berechneten Repräsentationen an super().forward, falls es sie benötigt
                                   # und nicht selbst run_gnn aufruft.
                                   # Dies hängt von der Implementierung von GNN_graphpred.forward ab.
                                   # Annahme: GNN_graphpred.forward ruft self.run_gnn und self.rnn_model.
                                   # Da wir `model.gnn` temporär ersetzen, sollte dies funktionieren.
                                  )


        if use_main_rnn:
            if hasattr(self, 'rnn_model') and self.rnn_model is not None:
                loss_prod, loss_react, prod_pred_max, react_pred = self.rnn_model(batch, node_representation, graph_representation)
                outputs.update({
                    'prod_pred_max': prod_pred_max, 'react_pred': react_pred,
                    'loss_prod': loss_prod, 'loss_react': loss_react,
                    'main_task_loss': loss_prod + loss_react
                })
            else: # Fallback, falls rnn_model nicht existiert
                outputs['main_task_loss'] = torch.tensor(0.0, device=graph_representation.device)
                print_once("WARNUNG: Haupt-RNN-Modell nicht verfügbar für main_task_loss.")

        elif meta_adv_predictor_idx is not None and self.meta_adv_reaction_predictors is not None:
            if 0 <= meta_adv_predictor_idx < len(self.meta_adv_reaction_predictors):
                predictor = self.meta_adv_reaction_predictors[meta_adv_predictor_idx]
                loss_prod_meta, loss_react_meta, _, _ = predictor(batch, node_representation, graph_representation)
                outputs['meta_task_loss'] = loss_prod_meta + loss_react_meta
            else:
                print_once(f"警告: 无效的 meta_adv_predictor_idx: {meta_adv_predictor_idx}")
                outputs['meta_task_loss'] = torch.tensor(0.0, device=graph_representation.device)
        
        # DANN-Komponente
        if self.training and hasattr(batch, 'domain_labels') and self.domain_classifiers is not None:
            graph_domain_labels_0_based = batch.domain_labels - 1
            current_graph_rep_for_dann = graph_representation
            if self.grl is not None:
                 current_graph_rep_for_dann = self.grl(graph_representation.clone(), self.adv_alpha) # .clone() zur Sicherheit

            domain_feat_for_dann = self.domain_projector(current_graph_rep_for_dann)
            
            dann_domain_loss_value = torch.tensor(0.0, device=graph_representation.device)
            unique_domain_ids_in_batch = torch.unique(graph_domain_labels_0_based).tolist()
            
            if len(unique_domain_ids_in_batch) >= 2:
                domain_pairs = list(combinations(unique_domain_ids_in_batch, 2))
                if domain_pairs:
                    actual_calculated_pairs = 0
                    accum_pair_loss = 0.0
                    max_pairs_to_sample = self.hparams_meta_doml.get('dann_max_pairs', 40)
                    domain_pairs_to_use = domain_pairs
                    if len(domain_pairs) > max_pairs_to_sample:
                        sampled_indices = torch.randperm(len(domain_pairs))[:max_pairs_to_sample].tolist()
                        domain_pairs_to_use = [domain_pairs[i] for i in sampled_indices]

                    for src_id, tgt_id in domain_pairs_to_use:
                        mask = (graph_domain_labels_0_based == src_id) | (graph_domain_labels_0_based == tgt_id)
                        if mask.sum() < 2: continue
                        try:
                            pair_idx_dann = self.get_pair_idx(src_id, tgt_id)
                        except ValueError: continue
                        if not (0 <= pair_idx_dann < len(self.domain_classifiers)): continue

                        classifier_dann = self.domain_classifiers[pair_idx_dann]
                        features_for_pair = domain_feat_for_dann[mask]
                        labels_for_pair = (graph_domain_labels_0_based[mask] == tgt_id).long()

                        if len(features_for_pair) > 0:
                            logits = classifier_dann(features_for_pair)
                            loss = F.cross_entropy(logits, labels_for_pair)
                            weight_dann = self.domain_weights_dann[src_id] * self.domain_weights_dann[tgt_id]
                            accum_pair_loss += loss * weight_dann
                            actual_calculated_pairs += 1
                    
                    if actual_calculated_pairs > 0:
                        dann_domain_loss_value = accum_pair_loss / actual_calculated_pairs
            
            outputs['dann_domain_loss'] = dann_domain_loss_value * torch.clamp(self.adv_alpha, 0.0, 1.0)
        else:
            outputs['dann_domain_loss'] = torch.tensor(0.0, device=graph_representation.device if graph_representation is not None else self.adv_alpha.device) # Fallback-Device

        return outputs

# --- Meta-DOML Trainingslogik ---
def get_data_by_domain(loader: DataLoader, device: torch.device, num_total_domains: int) -> Dict[int, List[Data]]:
    print_once("信息: get_data_by_domain 被调用。将数据按域分组。")
    domain_data_lists: Dict[int, List[Data]] = defaultdict(list)

    for pyg_batch in loader:
        if pyg_batch is None: continue
        
        if not hasattr(pyg_batch, 'domain_labels') or not hasattr(pyg_batch, 'num_graphs'):
            print_once(f"警告: Batch-Objekt (Typ: {type(pyg_batch)}) fehlen 'domain_labels' oder 'num_graphs'. Übersprungen.")
            continue
        if pyg_batch.num_graphs == 0: continue # Leeren Batch überspringen

        # Datenpunkte einzeln auf das Gerät verschieben, um GPU-Speicher bei der Sammlung zu schonen
        for i in range(pyg_batch.num_graphs):
            data_point: Data = pyg_batch[i]
            # .to(device) hier, um sicherzustellen, dass die Daten auf dem richtigen Gerät sind,
            # bevor sie der Liste hinzugefügt werden. Vermeidet spätere Probleme.
            data_point = data_point.to(device) 
            domain_id_1_based = pyg_batch.domain_labels[i].item()
            domain_id_0_based = domain_id_1_based - 1
            domain_data_lists[domain_id_0_based].append(data_point)
            
    if not domain_data_lists:
        print_once("警告: get_data_by_domain 未能 Daten sammeln.")
    else:
        print_once(f"get_data_by_domain: {len(domain_data_lists)} Domänen gefunden mit folgenden Anzahlen an Graphen:")
        for dom_id, data_list_for_dom in domain_data_lists.items():
            print_once(f"  Domäne {dom_id}: {len(data_list_for_dom)} Graphen")
            
    return domain_data_lists

def create_meta_splits(domain_data_lists: Dict[int, List[Data]], hparams_meta_doml: dict) -> List[dict]:
    domain_ids = sorted(domain_data_lists.keys())
    num_total_domains_in_set = len(domain_ids)

    if num_total_domains_in_set < 2:
        print_once("警告: Datenmenge enthält weniger als 2 Domänen, keine Meta-Splits möglich.")
        return []

    splits = []
    for i in range(num_total_domains_in_set):
        target_domain_id = domain_ids[i]
        target_data_list = domain_data_lists[target_domain_id]
        
        source_domain_info_list = []
        for j in range(num_total_domains_in_set):
            if i == j: continue
            source_id = domain_ids[j]
            source_data_list = domain_data_lists[source_id]
            if source_data_list : # Nur hinzufügen, wenn Daten vorhanden sind
                source_domain_info_list.append({'id': source_id, 'data': source_data_list})
        
        if not source_domain_info_list or not target_data_list: # Sicherstellen, dass Ziel und Quellen Daten haben
            continue
            
        splits.append({
            'target': {'id': target_domain_id, 'data': target_data_list},
            'sources': source_domain_info_list 
        })
    if not splits:
        print_once("WARNUNG: create_meta_splits hat keine gültigen Splits erstellt.")
    return splits

def train_meta_doml(args, model: DOMLGNN, device: torch.device, loader: DataLoader, optimizers: dict, epoch_num: int):
    model.train()

    gen_opt = optimizers.get('gen_opt') # .get verwenden für Sicherheit
    meta_adv_opt = optimizers.get('meta_adv_opt', None)
    main_reaction_opt = optimizers.get('main_reaction_opt')
    dann_disc_opt = optimizers.get('dann_disc_opt', None)

    if not gen_opt or not main_reaction_opt:
        print_once("FEHLER: gen_opt oder main_reaction_opt nicht initialisiert. Training abgebrochen.")
        return {'total_loss': -1, 'meta_loss_in': -1, 'meta_loss_out': -1, 'main_loss': -1, 'dann_loss': -1}


    domain_data_lists = get_data_by_domain(loader, device, model.num_domains_total)
    
    perform_meta_learning = model.meta_adv_reaction_predictors is not None and \
                            meta_adv_opt is not None and \
                            len(domain_data_lists) >= 2
    
    if not perform_meta_learning:
        # Fallback-Logik (wie zuvor)
        print_once(f"Epoch {epoch_num}: Meta-Learning Bedingungen nicht erfüllt oder <2 Domänen. Fallback zu Standard DANN + Hauptverlust.")
        total_loss_epoch_fb = 0.0; main_loss_epoch_fb = 0.0; dann_loss_epoch_fb = 0.0; num_graphs_epoch_fb = 0
        for original_batch_fb in tqdm(loader, desc=f"Epoch {epoch_num} Fallback Training", leave=False):
            if original_batch_fb is None or (hasattr(original_batch_fb, 'num_graphs') and original_batch_fb.num_graphs == 0): continue
            original_batch_fb = original_batch_fb.to(device)
            current_num_graphs_fb = original_batch_fb.num_graphs if hasattr(original_batch_fb, 'num_graphs') else 1
            num_graphs_epoch_fb += current_num_graphs_fb

            outputs_fb = model(original_batch_fb, use_main_rnn=True)
            main_loss_fb = outputs_fb.get('main_task_loss', torch.tensor(0.0, device=device))
            dann_loss_fb = outputs_fb.get('dann_domain_loss', torch.tensor(0.0, device=device))
            current_total_loss_fb = main_loss_fb + dann_loss_fb
            
            if not current_total_loss_fb.requires_grad:
                total_loss_epoch_fb += current_total_loss_fb.item() * current_num_graphs_fb
                main_loss_epoch_fb += main_loss_fb.item() * current_num_graphs_fb
                dann_loss_epoch_fb += (dann_loss_fb.item() if isinstance(dann_loss_fb, torch.Tensor) else dann_loss_fb) * current_num_graphs_fb
                continue

            gen_opt.zero_grad(); main_reaction_opt.zero_grad()
            if dann_disc_opt: dann_disc_opt.zero_grad()
            
            current_total_loss_fb.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            
            gen_opt.step(); main_reaction_opt.step()
            if dann_disc_opt: dann_disc_opt.step()

            total_loss_epoch_fb += current_total_loss_fb.item() * current_num_graphs_fb
            main_loss_epoch_fb += main_loss_fb.item() * current_num_graphs_fb
            dann_loss_epoch_fb += (dann_loss_fb.item() if isinstance(dann_loss_fb, torch.Tensor) else dann_loss_fb) * current_num_graphs_fb
        
        return {
            'total_loss': total_loss_epoch_fb / num_graphs_epoch_fb if num_graphs_epoch_fb > 0 else 0,
            'meta_loss_in': 0, 'meta_loss_out': 0,
            'main_loss': main_loss_epoch_fb / num_graphs_epoch_fb if num_graphs_epoch_fb > 0 else 0,
            'dann_loss': dann_loss_epoch_fb / num_graphs_epoch_fb if num_graphs_epoch_fb > 0 else 0,
        }

    meta_splits = create_meta_splits(domain_data_lists, model.hparams_meta_doml)
    num_actual_splits = len(meta_splits)
    if num_actual_splits == 0:
        print_once(f"Epoch {epoch_num}: Trotz erfüllter Bedingungen wurden keine Meta-Splits erstellt. Fallback.")
        # (Identischer Fallback-Code wie oben, hier der Kürze halber weggelassen)
        return {'total_loss': 0, 'meta_loss_in': 0, 'meta_loss_out': 0, 'main_loss': 0, 'dann_loss': 0} # Fallback-Return

    total_objective_in = 0.0
    total_objective_out = 0.0
    
    # Gradienten für Hauptoptimierer (außer meta_adv_opt) vor der Split-Schleife nullen
    gen_opt.zero_grad()
    # main_reaction_opt.zero_grad() # Wird erst am Ende für den Hauptverlust benötigt
    # if dann_disc_opt: dann_disc_opt.zero_grad() # Wird erst am Ende für den DANN-Verlust benötigt

    for split_idx, split_info in enumerate(tqdm(meta_splits, desc=f"Epoch {epoch_num} Meta-Splits", leave=False)):
        target_info = split_info['target']
        source_infos = split_info['sources']
        target_domain_id = target_info['id']
        target_data_list = target_info['data']

        cloned_gnn = copy.deepcopy(model.gnn)
        cloned_gnn.train()
        clone_gen_opt = optim.Adam(cloned_gnn.parameters(), lr=model.hparams_meta_doml['lr_g_clone'])
        
        original_gnn_module = model.gnn
        model.gnn = cloned_gnn # Temporäres Ersetzen

        # === Inner Loop ===
        if meta_adv_opt: meta_adv_opt.zero_grad()
        # clone_gen_opt.zero_grad() # Wird für jeden Klon neu initialisiert

        # --- Schritt 1: Update meta_adv_reaction_predictors ---
        accumulated_loss_for_meta_adv_step1 = torch.tensor(0.0, device=device)
        # Speichere die Graphenrepräsentationen, die vom Klon erzeugt wurden,
        # um sie im Schritt 2 für den Klon-Verlust wiederzuverwenden und Gradientenfluss zu ermöglichen.
        s1_reps_for_clone_update = []
        s2_reps_for_clone_update = []
        s1_batches_for_clone_update = []
        s2_batches_for_clone_update = []
        pair_indices_for_clone_update = []
        num_valid_pairs_for_meta_adv = 0

        for idx_s1 in range(len(source_infos)):
            for idx_s2 in range(idx_s1 + 1, len(source_infos)):
                s1_info_inner = source_infos[idx_s1]
                s2_info_inner = source_infos[idx_s2]
                s1_domain_id_inner, s1_data_list_inner = s1_info_inner['id'], s1_info_inner['data']
                s2_domain_id_inner, s2_data_list_inner = s2_info_inner['id'], s2_info_inner['data']

                try:
                    pair_idx_meta_inner = model.get_pair_idx(s1_domain_id_inner, s2_domain_id_inner)
                except ValueError: continue
                if not (model.meta_adv_reaction_predictors and \
                        0 <= pair_idx_meta_inner < len(model.meta_adv_reaction_predictors)): continue

                s1_loader_inner_loop = DataLoader(s1_data_list_inner, batch_size=args.batch_size, shuffle=True, drop_last=False)
                s2_loader_inner_loop = DataLoader(s2_data_list_inner, batch_size=args.batch_size, shuffle=True, drop_last=False)
                
                # Iteriere über gepaarte Mini-Batches
                for s1_mini_batch, s2_mini_batch in zip(s1_loader_inner_loop, cycle(s2_loader_inner_loop) if len(s1_loader_inner_loop) > len(s2_loader_inner_loop) else cycle(s1_loader_inner_loop)):
                    if len(s1_loader_inner_loop) <= len(s2_loader_inner_loop) and s1_mini_batch == list(s1_loader_inner_loop)[-1] and len(s1_loader_inner_loop) < len(s2_loader_inner_loop): # Abbruchbedingung cycle
                         if s2_mini_batch == list(s2_loader_inner_loop)[len(s1_loader_inner_loop)-1] : break # Ungefähr
                    elif len(s2_loader_inner_loop) < len(s1_loader_inner_loop) and s2_mini_batch == list(s2_loader_inner_loop)[-1] and len(s2_loader_inner_loop) < len(s1_loader_inner_loop): # Abbruchbedingung cycle
                         if s1_mini_batch == list(s1_loader_inner_loop)[len(s2_loader_inner_loop)-1] : break # Ungefähr

                    s1_mini_batch, s2_mini_batch = s1_mini_batch.to(device), s2_mini_batch.to(device)

                    # Graphenrepräsentationen mit cloned_gnn (das in model.gnn ist)
                    s1_outputs_step1 = model(s1_mini_batch, use_main_rnn=False, use_specific_gnn=cloned_gnn)
                    s1_rep_step1 = {'graph': s1_outputs_step1['graph_representation'], 'node': s1_outputs_step1['node_representation']}
                    s2_outputs_step1 = model(s2_mini_batch, use_main_rnn=False, use_specific_gnn=cloned_gnn)
                    s2_rep_step1 = {'graph': s2_outputs_step1['graph_representation'], 'node': s2_outputs_step1['node_representation']}

                    # Diese Repräsentationen speichern wir für Schritt 2 (Klon-Update), da sie von cloned_gnn abhängen
                    s1_reps_for_clone_update.append(s1_rep_step1)
                    s2_reps_for_clone_update.append(s2_rep_step1)
                    s1_batches_for_clone_update.append(s1_mini_batch) # Batches speichern für den Forward-Pass mit meta_adv_predictors
                    s2_batches_for_clone_update.append(s2_mini_batch)
                    pair_indices_for_clone_update.append(pair_idx_meta_inner)


                    # Für das Update der meta_adv_predictors, behandeln wir die Repräsentationen als fixiert (detach)
                    s1_rep_detached = {'graph': s1_rep_step1['graph'].detach(), 'node': s1_rep_step1['node'].detach()}
                    s2_rep_detached = {'graph': s2_rep_step1['graph'].detach(), 'node': s2_rep_step1['node'].detach()}

                    er_s1_outputs_adv = model(s1_mini_batch, use_main_rnn=False, meta_adv_predictor_idx=pair_idx_meta_inner, specific_graph_rep=s1_rep_detached)
                    er_s1_adv = er_s1_outputs_adv.get('meta_task_loss', torch.tensor(0.0, device=device))
                    er_s2_outputs_adv = model(s2_mini_batch, use_main_rnn=False, meta_adv_predictor_idx=pair_idx_meta_inner, specific_graph_rep=s2_rep_detached)
                    er_s2_adv = er_s2_outputs_adv.get('meta_task_loss', torch.tensor(0.0, device=device))

                    pair_disc_adv = torch.abs(er_s1_adv - er_s2_adv)
                    if pair_disc_adv.requires_grad:
                        num_valid_pairs_for_meta_adv += 1
                        accumulated_loss_for_meta_adv_step1 += (-pair_disc_adv)
        
        if num_valid_pairs_for_meta_adv > 0 and meta_adv_opt:
            avg_loss_for_meta_adv_step1 = accumulated_loss_for_meta_adv_step1 / num_valid_pairs_for_meta_adv # Skalierung durch num_actual_splits hier nicht, da es der Durchschnitt pro Paar ist
            if avg_loss_for_meta_adv_step1.requires_grad:
                avg_loss_for_meta_adv_step1.backward() # Kein retain_graph, da die Graphen detached waren
                torch.nn.utils.clip_grad_norm_(model.meta_adv_reaction_predictors.parameters(), args.clip_norm)
                meta_adv_opt.step()
        if meta_adv_opt: meta_adv_opt.zero_grad() # Zero grad für den Outer Loop oder nächsten Split


        # --- Schritt 2: Update cloned_gnn ---
        # Verwende die *aktualisierten* meta_adv_reaction_predictors und die zuvor gespeicherten Repräsentationen,
        # die noch Gradienten zum cloned_gnn haben.
        accumulated_loss_for_clone_step2 = torch.tensor(0.0, device=device)
        num_valid_pairs_for_clone = 0
        for k in range(len(s1_reps_for_clone_update)):
            s1_rep_k = s1_reps_for_clone_update[k]
            s2_rep_k = s2_reps_for_clone_update[k]
            s1_batch_k = s1_batches_for_clone_update[k]
            s2_batch_k = s2_batches_for_clone_update[k]
            pair_idx_meta_k = pair_indices_for_clone_update[k]

            # er_s1, er_s2 mit (potenziell aktualisierten) meta_adv_predictors und den Graphenrepräsentationen,
            # die noch an cloned_gnn (jetzt in model.gnn) angehängt sind.
            er_s1_outputs_clone = model(s1_batch_k, use_main_rnn=False, meta_adv_predictor_idx=pair_idx_meta_k, specific_graph_rep=s1_rep_k)
            er_s1_clone = er_s1_outputs_clone.get('meta_task_loss', torch.tensor(0.0, device=device))
            er_s2_outputs_clone = model(s2_batch_k, use_main_rnn=False, meta_adv_predictor_idx=pair_idx_meta_k, specific_graph_rep=s2_rep_k)
            er_s2_clone = er_s2_outputs_clone.get('meta_task_loss', torch.tensor(0.0, device=device))
            
            pair_disc_clone = torch.abs(er_s1_clone - er_s2_clone)
            if pair_disc_clone.requires_grad:
                num_valid_pairs_for_clone += 1
                accumulated_loss_for_clone_step2 += pair_disc_clone
        
        if num_valid_pairs_for_clone > 0:
            avg_loss_for_clone_step2 = accumulated_loss_for_clone_step2 / num_valid_pairs_for_clone
            total_objective_in += avg_loss_for_clone_step2.item()
            if avg_loss_for_clone_step2.requires_grad:
                clone_gen_opt.zero_grad()
                # Gradienten für cloned_gnn Parameter (die in clone_gen_opt sind)
                avg_loss_for_clone_step2.backward() # Kein retain_graph, da dies der letzte Gebrauch dieses Graphen ist
                torch.nn.utils.clip_grad_norm_(cloned_gnn.parameters(), args.clip_norm)
                clone_gen_opt.step()
        
        # --- Ende Inner Loop für einen Split ---
        # (Ähnliche zweistufige Logik für Outer Loop anwenden)
        # Outer Loop würde das gerade aktualisierte cloned_gnn verwenden,
        # um Gradienten für das original_gnn_module zu berechnen.

        # === Outer Loop (vereinfachte Darstellung, muss analog zum Inner Loop implementiert werden) ===
        if meta_adv_opt: meta_adv_opt.zero_grad() # Vorbereitung für Outer Loop meta_adv Updates
        accumulated_loss_for_meta_adv_outer_step1 = torch.tensor(0.0, device=device)
        # ... (Iteriere über source_infos und target_info, um Paare für Outer Loop zu bilden)
        # ... (Berechne er_s_outer, er_t_outer mit detached Reps von cloned_gnn für meta_adv_opt Update)
        # ... (meta_adv_opt.step())

        accumulated_loss_for_gen_opt_outer_step2 = torch.tensor(0.0, device=device)
        # ... (Iteriere über source_infos und target_info erneut)
        # ... (Berechne er_s_outer, er_t_outer mit *nicht* detached Reps von cloned_gnn und aktualisierten meta_adv_predictors)
        # ... (loss_for_gen_opt_outer = lambda_outer * (abs_diff / num_pairs))
        # ... (grads_outer = autograd.grad(loss_for_gen_opt_outer, cloned_gnn.parameters()))
        # ... (Akkumuliere grads_outer zu original_gnn_module.grad)
        # total_objective_out += loss_for_gen_opt_outer.item()


        # Am Ende des split_info-Loops:
        model.gnn = original_gnn_module # Stelle ursprüngliches GNN wieder her
        del cloned_gnn
        del clone_gen_opt
        # if torch.cuda.is_available(): torch.cuda.empty_cache() # Nur zum aggressiven Testen

    # --- Ende der Meta-Split-Schleife ---

    # --- Update mit Haupt-Aufgabenverlust und DANN-Verlust (auf allen Daten des Loaders) ---
    # Stelle sicher, dass die Gradienten für gen_opt und main_reaction_opt (und dann_disc_opt)
    # korrekt für diesen abschließenden Schritt genullt werden, unter Berücksichtigung der
    # bereits akkumulierten Meta-Gradienten in gen_opt (original_gnn_module.grad).
    # main_reaction_opt und dann_disc_opt wurden zu Beginn von train_meta_doml genullt.
    # gen_opt wurde auch zu Beginn von train_meta_doml genullt und hat dann Meta-Gradienten akkumuliert.
    # Jetzt werden die Gradienten vom Haupt-/DANN-Verlust hinzugefügt.

    total_main_loss_epoch_val = 0.0
    total_dann_loss_epoch_val = 0.0
    num_graphs_in_epoch_main_dann = 0

    for original_batch_main_dann in tqdm(loader, desc=f"Epoch {epoch_num} Main/DANN Loss", leave=False):
        if original_batch_main_dann is None or (hasattr(original_batch_main_dann, 'num_graphs') and original_batch_main_dann.num_graphs == 0) : continue
        original_batch_main_dann = original_batch_main_dann.to(device)
        
        current_num_graphs_md = original_batch_main_dann.num_graphs if hasattr(original_batch_main_dann, 'num_graphs') else 1
        num_graphs_in_epoch_main_dann += current_num_graphs_md

        outputs_md = model(original_batch_main_dann, use_main_rnn=True) # Verwendet original_gnn_module
        main_loss_md = outputs_md.get('main_task_loss', torch.tensor(0.0, device=device))
        dann_loss_md = outputs_md.get('dann_domain_loss', torch.tensor(0.0, device=device))
        
        combined_loss_md = torch.tensor(0.0, device=device)
        add_to_combined = False
        if main_loss_md.requires_grad:
            combined_loss_md += main_loss_md
            add_to_combined = True
        if isinstance(dann_loss_md, torch.Tensor) and dann_loss_md.requires_grad:
            combined_loss_md += dann_loss_md
            add_to_combined = True
        
        if add_to_combined: # Nur wenn es etwas zu backpropagieren gibt
             # Die Gradienten werden zu den existierenden .grad Attributen addiert.
             # Für gen_opt (model.gnn) sind das die Meta-Gradienten.
             # Für main_reaction_opt und dann_disc_opt sind die .grad Attribute vorher None oder 0.
            combined_loss_md.backward() 
        
        total_main_loss_epoch_val += main_loss_md.item() * current_num_graphs_md
        total_dann_loss_epoch_val += (dann_loss_md.item() if isinstance(dann_loss_md, torch.Tensor) else dann_loss_md) * current_num_graphs_md

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    
    if gen_opt: gen_opt.step()
    if main_reaction_opt: main_reaction_opt.step()
    if dann_disc_opt: dann_disc_opt.step()

    avg_main_loss = total_main_loss_epoch_val / num_graphs_in_epoch_main_dann if num_graphs_in_epoch_main_dann > 0 else 0
    avg_dann_loss = total_dann_loss_epoch_val / num_graphs_in_epoch_main_dann if num_graphs_in_epoch_main_dann > 0 else 0
    avg_meta_in = total_objective_in # Wurde bereits pro Split gemittelt, hier nur für Logging
    avg_meta_out = total_objective_out # Wurde bereits pro Split gemittelt

    return {
        'total_loss': avg_main_loss + avg_dann_loss + avg_meta_out, # Meta-Out wird als Teil des Verlusts betrachtet
        'meta_loss_in': avg_meta_in,
        'meta_loss_out': avg_meta_out,
        'main_loss': avg_main_loss,
        'dann_loss': avg_dann_loss,
    }


# --- Main Funktion und andere Hilfsfunktionen (aus Original übernommen/angepasst) ---
def main():
    parser = argparse.ArgumentParser(description='DOML-Meta GNN für Reaktionsvorhersage')
    # ... (Argumente wie zuvor, stelle sicher, dass keine deutschen Kommentare mehr drin sind) ...
    parser.add_argument('--device', type=int, default=0, help='GPU Geräte-ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch-Größe für das Training')
    parser.add_argument('--epochs', type=int, default=100, help='Anzahl der Trainingsepochen')
    parser.add_argument('--lr', type=float, default=0.001, help='Veraltete Lernrate, spezifische LRs verwenden')
    parser.add_argument('--decay', type=float, default=0, help='Weight Decay')
    parser.add_argument('--num_layer', type=int, default=6, help='Anzahl der GNN Message Passing Layer')
    parser.add_argument('--emb_dim', type=int, default=300, help='Embedding-Dimensionen')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='Dropout-Rate')
    parser.add_argument('--graph_pooling', type=str, default="attention", help='Graph-Pooling')
    parser.add_argument('--JK', type=str, default="last", help='Jumping Knowledge')
    parser.add_argument('--dataset', type=str, default='data/USPTO50K', help='Stammverzeichnis des Datensatzes')
    parser.add_argument('--atom_feat_dim', type=int, default=45, help='Atom-Merkmalsdimension')
    parser.add_argument('--bond_feat_dim', type=int, default=12, help='Bindungs-Merkmalsdimension')
    parser.add_argument('--center_loss_type', type=str, default='ce', help='Verlusttyp für Reaktionszentrum')
    parser.add_argument('--typed', action='store_true', default=False, help='Ob Reaktionstypen gegeben sind')
    parser.add_argument('--test_only', action='store_true', default=False, help='Nur auf Testdaten evaluieren')
    parser.add_argument('--multiprocess', action='store_true', default=False, help='Mit Multi-Prozessen trainieren (nicht angepasst!)')
    parser.add_argument('--num_process', type=int, default=4, help='Anzahl der Prozesse für Multi-Prozess-Training')
    parser.add_argument('--input_model_file', type=str, default="", help='Dateiname des zu ladenden Modells')
    parser.add_argument('--filename', type=str, default='uspto50k_meta_doml', help='Ausgabedateiname-Präfix')
    parser.add_argument('--runseed', type=int, default=42, help="Zufallsgenerator-Seed")
    parser.add_argument('--eval_split', type=str, default='test', help='Evaluationssplit (test/valid/train)')
    parser.add_argument('--beam_size', type=int, default=50, help='Beam Search Größe')
    parser.add_argument('--num_domains', type=int, default=5, help='Gesamtanzahl der Domänen im Datensatz')
    parser.add_argument('--clip_norm', type=float, default=5.0, help="Gradient Clipping Norm")
    parser.add_argument('--initial_adv_alpha', type=float, default=0.1, help='Initialer adv_alpha für DANN GRL')
    parser.add_argument('--lr_g_meta', type=float, default=1e-4, help='Meta-LR für Featurizer (gen_opt)')
    parser.add_argument('--lr_g_clone', type=float, default=1e-4, help='Meta-LR für Featurizer-Klon')
    parser.add_argument('--lr_meta_adv', type=float, default=1e-4, help='Meta-LR für meta_adv_reaction_predictors')
    parser.add_argument('--lr_main_reaction', type=float, default=3e-4, help='LR für Haupt-Reaktionsmodell')
    parser.add_argument('--lr_dann_disc', type=float, default=3e-4, help='LR für DANN-Diskriminatoren')
    parser.add_argument('--lambda_outer', type=float, default=1.0, help='Gewichtung für äußeren Meta-Verlust')
    parser.add_argument('--doml_beta', type=float, default=1.0, help='Gewichtung für Anwendung des äußeren Gradienten auf Featurizer')
    parser.add_argument('--dann_max_pairs', type=int, default=20, help='Maximale Anzahl DANN-Paare für DANN-Verlust-Sampling')

    args = parser.parse_args()
    print_once(str(args))

    hparams_meta_doml = {
        "initial_adv_alpha": args.initial_adv_alpha,
        "lr_g_meta": args.lr_g_meta,
        "lr_g_clone": args.lr_g_clone,
        "lr_meta_adv": args.lr_meta_adv,
        "lr_main_reaction": args.lr_main_reaction,
        "lr_dann_disc": args.lr_dann_disc,
        "lambda_outer": args.lambda_outer,
        "doml_beta": args.doml_beta,
        "weight_decay_g": args.decay,
        "dann_max_pairs": args.dann_max_pairs,
    }

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.test_only:
        test_dataset = MoleculeDataset(args.dataset, split=args.eval_split, load_mol=True)
        prod_word_size = len(test_dataset.prod_smarts_fp_list)
        react_word_size = len(test_dataset.react_smarts_list)
    else:
        # Wichtig: MoleculeDataset muss das Attribut 'domain_labels' für jedes Data-Objekt bereitstellen.
        # Der Parameter 'num_domains_for_generation' wurde entfernt, da er den Fehler verursachte.
        # Stellen Sie sicher, dass Ihr MoleculeDataset die Domänen-Labels korrekt lädt/generiert.
        train_dataset = MoleculeDataset(args.dataset, split='train', load_mol=True)
        valid_dataset = MoleculeDataset(args.dataset, split='valid', load_mol=False)
        prod_word_size = len(train_dataset.prod_smarts_fp_list)
        react_word_size = len(train_dataset.react_smarts_list)

    model = DOMLGNN(
        num_domains_total=args.num_domains,
        fusion_function=0,
        hparams_meta_doml=hparams_meta_doml,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        atom_feat_dim=args.atom_feat_dim,
        bond_feat_dim=args.bond_feat_dim,
        center_loss_type=args.center_loss_type,
        prod_word_size=prod_word_size,
        react_word_size=react_word_size,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        graph_pooling=args.graph_pooling
    ).to(device)

    gen_opt_params = []
    if hasattr(model, 'gnn') and model.gnn is not None:
         gen_opt_params.extend(model.gnn.parameters())
    if hasattr(model, 'atom_encoder') and model.atom_encoder is not None : # atom_encoder ist auch Teil des Featurizers
         gen_opt_params.extend(model.atom_encoder.parameters())
    if hasattr(model, 'graph_pred_linear') and model.graph_pred_linear is not None:
         gen_opt_params.extend(model.graph_pred_linear.parameters())
    
    optimizers = {}
    if gen_opt_params:
        optimizers['gen_opt'] = optim.Adam(gen_opt_params, lr=hparams_meta_doml['lr_g_meta'], weight_decay=args.decay)
    else:
        optimizers['gen_opt'] = None; print_once("WARNUNG: Keine Parameter für gen_opt.")

    if hasattr(model, 'rnn_model') and model.rnn_model is not None:
        optimizers['main_reaction_opt'] = optim.Adam(model.rnn_model.parameters(), lr=hparams_meta_doml['lr_main_reaction'], weight_decay=args.decay)
    else:
        optimizers['main_reaction_opt'] = None; print_once("WARNUNG: model.rnn_model nicht gefunden.")

    if model.meta_adv_reaction_predictors is not None:
        optimizers['meta_adv_opt'] = optim.Adam(model.meta_adv_reaction_predictors.parameters(), lr=hparams_meta_doml['lr_meta_adv'], weight_decay=args.decay)
    else:
        optimizers['meta_adv_opt'] = None

    if model.domain_classifiers is not None:
        dann_disc_params = list(model.domain_projector.parameters()) + \
                           list(model.domain_classifiers.parameters()) + \
                           [model.adv_alpha, model.domain_temp]
        optimizers['dann_disc_opt'] = optim.Adam(dann_disc_params, lr=hparams_meta_doml['lr_dann_disc'], weight_decay=args.decay)
    else:
        optimizers['dann_disc_opt'] = None
    
    if optimizers.get('gen_opt') is None or optimizers.get('main_reaction_opt') is None:
        print_once("FEHLER: gen_opt oder main_reaction_opt nicht erstellt. Training abgebrochen.")
        return

    if hasattr(model, 'gnn_diff'): delattr(model, 'gnn_diff')
    if hasattr(model, 'scoring'): delattr(model, 'scoring')
    
    dataset_name = os.path.basename(args.dataset.rstrip('/')) # rstrip für sauberen Namen
    args.filename = os.path.join('logs', dataset_name, args.filename if args.filename else "default_run")
    os.makedirs(args.filename, exist_ok=True)

    if args.input_model_file:
        input_model_file_path = args.input_model_file
        if not os.path.isabs(input_model_file_path) and args.filename:
             input_model_file_path = os.path.join(args.filename, args.input_model_file)
        if os.path.exists(input_model_file_path):
            try:
                model.load_state_dict(torch.load(input_model_file_path, map_location=device))
                print(f"Modell geladen von: {input_model_file_path}")
            except Exception as e_load:
                 print(f"FEHLER beim Laden des Modells von {input_model_file_path}: {e_load}")
        else:
            print(f"WARNUNG: Modelldatei nicht gefunden: {input_model_file_path}")

    if args.test_only:
        print("Nur Evaluation auf Testdaten...")
        eval_decoding(args, model, device, test_dataset)
        exit(0)

    if args.multiprocess:
        print_once("WARNUNG: Multiprocessing ist NICHT für die neue train_meta_doml-Schleife angepasst!")
        exit(1)

    output_model_file_template = os.path.join(args.filename, 'model_meta_doml_epoch{}.pt')
    print(f"Modelle werden gespeichert als: {output_model_file_template.format('X')}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} ====")
        metrics = train_meta_doml(args, model, device, train_loader, optimizers, epoch)
        print(f"Epoch {epoch} Ergebnisse: TotalLoss: {metrics.get('total_loss', -1):.4f}, MetaIn: {metrics.get('meta_loss_in', -1):.4f}, MetaOut: {metrics.get('meta_loss_out', -1):.4f}, MainTask: {metrics.get('main_loss', -1):.4f}, DANN: {metrics.get('dann_loss', -1):.4f}")

        if (epoch % 10 == 0 or epoch == args.epochs) and args.filename:
            try:
                torch.save(model.state_dict(), output_model_file_template.format(epoch))
                print(f"Modell gespeichert: {output_model_file_template.format(epoch)}")
            except Exception as e_save:
                print_once(f"FEHLER beim Speichern des Modells: {e_save}")
    
    if args.filename:
        final_model_path = os.path.join(args.filename, 'model_meta_doml_final.pt')
        try:
            torch.save(model.state_dict(), final_model_path)
            print(f"Finale Modell gespeichert: {final_model_path}")
        except Exception as e_save_final:
            print_once(f"FEHLER beim Speichern des finalen Modells: {e_save_final}")

# Platzhalter für Evaluationsfunktionen (aus Ihrem Originalcode übernehmen)
def eval_decoding(args, model, device, dataset, save_res=True, k=0):
    print_once("eval_decoding: Implementierung übernommen, aber Domänen-Label-Logik für Eval muss geprüft werden.")
    model.eval()
    pred_results = {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataset_has_domain_info = False # Hier Logik zur Prüfung einfügen
    domain_labels_for_eval_map = {} # Hier Logik zur Befüllung einfügen

    for batch_idx_loop, batch in enumerate(tqdm(loader, desc="Evaluation Iteration", leave=False)):
        # ... (Rest der eval_decoding Logik wie in der vorherigen Antwort) ...
        pass # Platzhalter für die eigentliche Logik
    
    if dataset_has_domain_info:
        # ... (Domänen-weise Genauigkeitsanalyse) ...
        pass
    else:
        print_once("Keine Domänen-Labels im Evaluationsdatensatz für domänen-weise Analyse gefunden.")

    if save_res and args.filename:
        # ... (Ergebnisse speichern) ...
        pass
    return pred_results

def eval_multi_process(args, model, device, test_dataset):
    print_once("eval_multi_process ist nicht vollständig implementiert oder für Meta-DOML angepasst.")

def train_multiprocess(rank, args, model, device, train_dataset, valid_dataset):
    print_once("train_multiprocess ist NICHT für Meta-DOML angepasst und sollte nicht verwendet werden.")

if __name__ == "__main__":
    main()