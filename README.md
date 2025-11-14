# Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors
This tutorial based on the server @tau-neutrino.ps.uci.edu  

This repository contains code to train an HPST, and baselines like GAT and RCNN on NoVa Data for Multiple-view particle detector Segmentation


# Setup
We recommend using [conda](https://docs.conda.io/) for environment setup.  

```bash
git clone https://github.com/mrheng9/NOvA-HPST.git
cd HPST-Nova
conda env create -f environment.yml
conda activate hpst
pip install --upgrade pip setuptools wheel
conda install -n base -c conda-forge mamba -y
mamba install -n hpst -c pytorch -c nvidia pytorch=2.5.1 pytorch-cuda=12.1 torchvision torchaudio -y

TORCH_VER=$(python -c "import torch as t; print(t.__version__.split('+')[0])")

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+cu121.html"

echo "PYG URL: $PYG_URL"

pip install -f "$PYG_URL" torch_scatter torch_cluster torch_geometric

pip install -U lightning rich
```

# Logging

We use WandB for logging. Please create a WandB project named "HPST" and use CLI login to use the code as is, or disable logging in the scripts.


# Training
To start the training, run 
```
HPST model:
bash train_hpst.sh

GAT model:
bash train_gat.sh

RCNN model:
bash train_rcnn.sh (not recommanded to run on tau server)
```
For example(commands in train_hpst.py), 
```
CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/train.py --options_file "config/hpst/hpst_tune_nova.json" --name "hpst run" --log_dir "runs" --gpus 2 > hpst.log 2>&1 &
```
The options_file is not a required argument. A description of each option is available in `hpst/utils/options.py`. 

The training log will be stored in the runs/hpst/wandb folder and the checkpoints will be stored under runs/hpst in the folder with the name of the specific moment you run it. 

We mainly foucus on the training and the performance of the HPST model and the results of GAT/RCNN will serve as the baseline to compare with the HPST's. However, the RCNN model is not recommanded to run on tau server for it has largest parameters.


The data we will use as specified in the example option file is in `/mnt/ironwolf_14t/users/ayankelevich/preprocessed_nova_miniprod6_1_cvnlabmaps.h5`

# Testing & Plottings
We provide a unified evaluator for HPST and GAT that reproduces all figures and metrics (confusion matrix, per-class accuracy, efficiency/purity, optional ROC curves).

To test the model, run
```
python scripts/evaluation.py --model gat/hpst --checkpoint_path "your checkpoint path" 
```   
## Notes
- `--model` and `checkpoin_path` are requiured when run the test
- use GPU 
By default the script uses CPU as defult. To force CPU, unset CUDA devices temporarily by adding: `--use_cuda False`

- Speed knobs  
--batch_size, --num_workers, --pin_memory to increase throughput.  
--max_batches N to evaluate on a subset for a quick sanity check.  
--examples_to_save 0 to skip event displays.  
--do_roc to additionally draw per-class ROC curves (AUC scores are always printed).


Outputs (saved under results/model)

- confusion_matrix.png
- confusion_matrix_normalized.png
- class_accuracy.png
- efficiency_purity_distribution.png
- roc_curves_per_class.png (only if -- do_roc is set)
- model_example_event*.png (event displays; count controlled by --examples_to_save)

Printed metrics

- Overall accuracy
- Per-class ROC-AUC scores (+ weighted average)
- Full classification report (precision/recall/F1)
- Per-class accuracy summary aligned with the confusion matrix

You can find more visualization codes in HPST-Nova/hpst/notebooks

# Compiling the network
The [CreateCompiled.ipynb](CreateCompiled.ipynb) jupyter notebook can be used to compile the network into a torchscript file for use in the C++ LArSoft ART framework. Modify the paths in the second cell to point to the directory and checkpoint of the model you want to export.
The final cell will create three torchscript files. All three models take as input a single tensor with the shape [(1+Npng), 3, 400, 280] where the first 3x400x280 image corresponds to the event pixel map followed by the prong pixel maps.   

The file ending in `pid` outputs a tuple of 2 tensors. The first tensor gives the softmax scores corresponding to the event classes. The second tensor is Npng x 8, corresponding to the prong softmax scores in the order that they were input.  

The file ending in `embeddings` also outputs a tuple of 2 tensors. Rather than the final event/prong predictions, this model outputs the intermediate feature representation vectors that are output by the transformer and serve as input into the classification layers. The first tensor has length 128 and corresponds to the event feature representation, and the second tensor is Npng x 128 and corresponds to the prong feature representations.  

The file ending in `combined` gives all of the above outputs as a tuple of 4 tensors: event prediction, prong predictions, event embedding, prong embeddings.  


