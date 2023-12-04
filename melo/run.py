import copy
import random
import importlib
import logging
from time import time
import hydra
from omegaconf import OmegaConf,open_dict
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import models
from trainer import zsre_trainer, hallucination_trainer, scotus_trainer
#TODO
#SUPPORT MELO CONV1D fan_in_fan_out

# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'
OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)
@hydra.main(config_path='config', config_name='config')
def run(config):
    grace_config_keys = ['edit_lr','init_radius','expand_mode','key_id','num_edit_per_block','num_block','num_rank_per_block']
    model_config_keys = ['target_modules','grace_layer']
    GRACE_CONFIG = dict(config.grace)
    MODEL_CONFIG = dict(config.model)

    for k in grace_config_keys:
        LOG.info(f'[-GRACE CONFIG-]  {k}: {GRACE_CONFIG[k]}')
    for k in model_config_keys:
        LOG.info(f'[-MODEL CONFIG-]  {k}: {MODEL_CONFIG[k]}')

    base_dir = hydra.utils.get_original_cwd()
    with open_dict(config):
        config.base_dir = base_dir

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.task == "qa" or config.task == "zsre":
        model = models.get_hf_model(config)
    elif config.task == "hallucination":
        model = models.get_hf_model(config)
    elif config.task == "scotus":
        model = models.get_hf_model(config)
    else:
        print(f"{config.task} task not found")



    model.to(config.device)
    tokenizer = models.get_tokenizer(config)


    '''
    Load Dataset
    '''
    if config.task == "qa" or config.task == "zsre":
        from dataset import NQ, zsRE, zsRE_balanced
        from metrics import F1_ACC, is_qa_error
        upstream = NQ()
        edits = zsRE_balanced(split="edit", n_edits=1000)
        edit_holdouts = zsRE_balanced(split="holdout", n_edits=1000)

        '''Get Loaders
        '''
        batch_size = config.grace.num_edit_per_block
        edit_loader = DataLoader(edits, batch_size=batch_size, shuffle=True)
        edit_holdout_loader = DataLoader(edit_holdouts, batch_size=batch_size, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=batch_size, shuffle=False)
        hold_out = 0
        '''Define Metrics
        '''
        metric = F1_ACC # Measure QA F1
        is_error = is_qa_error
        tokenize = tokenize_qa

    elif config.task == "hallucination":
        from dataset import Hallucination, WebText10k
        from metrics import PPL
        upstream = WebText10k()
        edits = Hallucination(split="edit")
        accurate_dataset = Hallucination(split="accurate")

        '''Get Loaders
        '''
        batch_size = config.grace.num_edit_per_block
        edit_loader = DataLoader(edits, batch_size=batch_size, shuffle=False)
        accurate_loader = DataLoader(accurate_dataset, batch_size=batch_size, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=batch_size, shuffle=False)
        '''Define Metrics
        '''
        metric = PPL # Measure QA F1
        tokenize = tokenize_gpt
    elif config.task == 'scotus':
        from dataset import SCOTUS
        from metrics import Accuracy
        upstream = SCOTUS("train")
        edits = SCOTUS("edit")

        '''Get Loaders
        '''
        batch_size = config.grace.num_edit_per_block
        edit_loader = DataLoader(edits, batch_size=batch_size, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=batch_size, shuffle=False)
        '''Define Metrics
        '''
        metric = Accuracy
        tokenize = tokenize_clf
    else:
        print(f"{config.task} task not found")

    alg_module = importlib.import_module(f'algs.{config.alg}')
    AlgClass = getattr(alg_module,config.alg.upper())
    alg = AlgClass(model,config,tokenizer)
    alg.to(config.device)

    # Trainer
    if config.task == "qa" or config.task == "zsre":
        trainer = zsre_trainer(config,alg,tokenize,metric,edit_loader,upstream_loader,edit_holdout_loader)
    elif config.task == "hallucination":
        trainer = hallucination_trainer(config,alg,tokenize,metric,edit_loader,upstream_loader,accurate_loader)
    elif config.task == "scotus":
        trainer = scotus_trainer(config,alg,tokenize,metric,edit_loader,upstream_loader)

    # trainer.pre_editing_analyse()
    torch.cuda.empty_cache()
    trainer.run_edit()


if __name__ == '__main__':
    run()