<!-- omit in toc -->
# MELO: Enhancing Model Editing with Neuron-Indexd Dynamic LoRA
This repo contains the source code of our proposed MELO, a plug-in model editing method, which routes models' behavoir by dynamically indexing LoRA blocks according to a inner vector databse. Seamlessly integrated in [PEFT](https://github.com/huggingface/peft), MELO supports multiple LLMs such as BERT, T5 and GPT. 

<!-- omit in toc -->
## Updates
- **2024/03/10:** Add some Important Tips for deployment 🪂
- **2023/12/09:** <strong> Our work has been accepted by AAAI 2024</strong> :fire::fire: 
- **2023/7/16:** Experiments with multiple LLMs on different editing tasks. :art:
- **2023/6/24:** Inner vector databse that builds accurate editing scope. :confetti_ball:	
- **2023/6/08:** Support dynamic LoRA block Loding. :star:

<!-- omit in toc -->
## Table of Contents
- [Reference](#reference)
- [Introduction](#introduction)
- [Experiments](#experiments)
- [Prepare Environments](#prepare-environments)
- [Prepare Datasets](#prepare-datasets)
- [Quick Start](#quick-start)
- [Important Tips](#important-tips)
- [Acknowledgments](#acknowledgments)
## Reference
We would appreciate if you could refer to our work as one of your baselines!
```
@article{yu2023melo,
  title={MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA},
  author={Yu, Lang and Chen, Qin and Zhou, Jie and He, Liang},
  journal={arXiv preprint arXiv:2312.11795},
  year={2023}
}
```
## Introduction
Due to the limitation of catastrophic forgetting and the lack of locality, few studies explore recent advanced Low-rank Adapter (LoRA) techniques for continual model editing. To overcome these limitations and take advantage of LoRA's resource efficiency, we propose MELO, a plug-in model editing method implemented with dynamic LoRA, which routes the behavior of language models by dynamically indexing LoRA blocks according to an inner vector database. MELO considers all editing properties and can be easily integrated into multiple LLMs such as BERT, T5 and GPT. Experimental results show that our proposed MELO achieves state-of-the-art editing performance on three sequential editing tasks (document classification, question answering and hallucination correction), while requires the least trainable parameters and computational cost.
![main](./figures/main_00.png)

## Experiments
Comparison of MELO to prior editing methods on sequential editing tasks. Note that MELO edits all language models with a single RTX 3090 GPU.
![table](./figures/table.png)

## Prepare Environments
Required CUDA environment and library dependencies are listed in: 
```
requirements.txt
```
Then you should install our modified PEFT:
<h1 align="center"> <p>🤗 PEFT-MELO</p></h1>

```
cd peft_egg
pip install -e .
```
Detailed implementation of MELO is in `./peft_egg/src/tuners/melo.py`
## Prepare Datasets
The zsRE experiments use data linked by the [MEND](https://github.com/eric-mitchell/mend) repository. Download the data for NQ and zsRE from their Google Drive link and unzip each sub-directory into ./melo/data. SCOTUS and Hallucination data are loaded through huggingface.

## Quick Start
The location of inner vector database and dynamic LoRA target modules can be modified in `./melo/model/config`

### Editing GPT2-XL on Hallucination with MELO
```
cd melo
python run.py +alg=lora +experiment=hallucination +model=gpt2xl
```

### Editing BERT on SCOTUS with MELO
```
cd melo
python run.py +alg=lora +experiment=scotus +model=scotus-bert
```

### Editing T5 on zsRE with MELO
```
cd melo
python run.py +alg=lora +experiment=qa +model=t5small
```
## Important Tips
* [Datasets](https://drive.google.com/file/d/1HDqh4ofYF7B-YkcU3CNlZMYAOJO0XxwX/view?usp=drive_link) for MELO's experiments can be downloaded through GoogleDrive now. Please extract the files and place them under `melo\data`.

* The GPT2-XL model we use is fine-tuned in line with the work [GRACE](https://github.com/Thartvigsen/GRACE/blob/728a52ebcd328ddca0bb1ec975e79625eabfab2a/grace/main.py#L83). Please download the checkpoint with the [Google Drive](https://drive.google.com/drive/folders/1j_DvcUY8goksQVOBt4XqBe7z8fS-0zvI?usp=sharing) link, and place the files under `melo/scr/models--gpt2-xl`
    
    
* Some [logs](https://drive.google.com/drive/folders/1UhlY1W8MUmvsxqIXlRFBfxuTXEQG8FJP?usp=sharing) recording the correct training and inference processes are released for checking hyper-parameters.

* The settings of [torch.optim.lr_scheduler](https://github.com/BruthYU/MELO/blob/51c8322cc06faa2b7665c2d90236f1bd1b8d9575/melo/algs/lora.py#L135) vary on different tasks:
  ```
   # T5-Small and T5-Large
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
   # SCOTUS-BERT and GPT2-XL
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
  ```


## Acknowledgments
We would like to thank the following individuals and organizations for their contributions to this project:
```
Huggingface: for their support of the PEFT community and their development of the PEFT framework (https://github.com/huggingface/peft)

GRACE: for the development of the open-source library GRACE which inspired our work (https://github.com/Thartvigsen/GRACE)
```
