# **Structured Composition of Targeted Augmentations for Group Shifts**

## Requirements

We provide two sets of python environment setups for both graph and medical image datasets. 

For experiments on image datasets, install requirements as requirements inside `data-augmentation-images` dataset:

```bash
cd data-augmentaton-images
pip install -r requirements.txt
```

For experiments on graph datasets, install requirements as requirements inside `data-augmentation-graphs` dataset:

```bash
cd data-augmentaton-graphs
pip install -r requirements.txt
```

## Image classification

This subproject includes experiments for the augmentation subset searching and the validation of the selected augmentation subset. For each experiment, we will work on 2 different datasets, which are `fundus` (`Messidor`, `Aptos`, and `Jinchi`) datasets for `self-supervised learning`, and `iWildCam` datasets for `supervised learning`.

### Augmentation subset searching

This part includes 2 key scripts, which are 

- `train_tree_based_augment_simclr.py` 
- `train_tree_based_augment_wilds.py`

**Self-supervised learning on fundus datasets**

The following test script will launch a ugmentation subset searching on fundus dataset with self-supervised learning using our approach

```bash
python train_tree_based_augment_simclr.py \
    --model VisionTransformer\
    --config configs/messidor_simclr.json \
    --epochs 1 \
    --task_set_name tree_messidor --save_name tree_messidor\
    --downsample_tasks 20 \
    --downsample_dataset 500 \
    --device 0
```

key arguments:

- `--model`: the pretrained model for fine-tuning. Here we use `VisionTransformer`.
- `--config`: the config file for this pytorch template project. See `Structure.md` for more details.
- `--task_set_name`: specifying the path for saving the best subset.
- `--save_name`: specifying the path for saving all the searching result that saved as `.cvs` file.
- `--downsample_tasks`: the amount of downsampled augmentation (with probability)
- `--downsample_dataset`: the amount of downsampled data of the dataset. 

The searched dataset is specified in the `.json` config file. You can replace it with `jinchi_simclr.json` or `aptos_simclr.json` to switch the dataset and modify the file for more configuration.

**Supervised learning on iWildCam datasets**

The following test script will launch an augmentation subset searching on iwildcam dataset with supervised learning using our approach

```bash
python train_tree_based_augment_wilds.py \
    --dataset iwildcam --group_id 307 --eval_metric val_accuracy \
    --task_set_name wildcam_307 --save_name wildcam_307 \
    --epochs 10 --batch_size 16 --device 0 
```

Key arguments:

- `--group_id`: The group id of the dataset. It would be used to pick a subset that in the same domain from the whole dataset.
- `--eval_metric` : The value for selecting the best combination.
- `--task_set_name`: specifying the path for saving the best subset.
- `--save_name`: specifying the path for saving all the searching result that saved as `.cvs`

If you need more configuration, please check the `.json` file that is used inside `train_tree_based_augment_wilds.py`

### Evaluating the performance of augmentations

This part includs 2 key scripts, which are

- `train_simclr_multitask.py`
- `train_wilds_multitask.py`

After we finish searching, we need to save the `.csv` file under `/results` folder. The script will automatically load the saved files.

**Contrastive learning on fundus datasets**

The following script will load the searching result of fundus in the `.csv` files and do multitask training on the pretrain model to check the performance of the selected augmentation.

```bash
python train_simclr_combine.py \
    --config configs/jinhong_config/multitask_simclr.json \
    --model VisionTransformer \
    --train_bilevel \
    --weight_lr 0.05  --update_weight_step 30 \
    --n_gpu 1 --device 0 --save_name best_fundus_multi_bi_50ep --epochs 50
```

Key arguments:

 - `--config`: the file of configurations.
 - `--model`: the pretrained model for training.
 - `--train_bilevel`: use `BilevelTrainer` for training, if not specified, use 
   `MultitaskSimCLRTrainer` for training.
 - `--weight_lr`: the learning rate of the weight. It is useful when we train the model with `BilevelTrainer` or `GroupDOR`.
 - `update_weight_step`: the steps between 2 weight updates.

**Supervised learning on the iWildCam dataset**

The following script will load the searching result of IWildCam dataset in the `.csv` files and do multitask training on the pretrain model to check the performance of the selected augmentation.

```bash
python train_wilds_combine.py \
    --config configs/multitask_wild.json \
    --group_ids 97 286 307 316\
    --train_no_transforms \
    --train_bilevel \
    --n_gpu 1 --device 1 --save_name testing --epochs 1 --runs 1\
    --weight_lr 0.01  --update_weight_step 1
```

Key arguments:

 - `--config`: the file of configurations.
 - `--model`: the pretrained model for training.
 - `--group_ids` : the group(domain) id that we used to train together.
 - `--train_no_transforms`: train the model without any augmentations. If not specified, trian with the best augmentation conbination from the searching result.
 - `--train_bilevel`: use `BilevelTrainer` for training, `--train_dro` for `GroupDROTrainer`. If not specified, use `MultitaskSupervisedTrainer`.
 - `save_name`: the name of the save folder
 - `--weight_lr`: the learning rate of the weight. It is useful when we train the model with `BilevelTrainer` or `GroupDOR`.
 - `update_weight_step`: the steps between 2 weight updates.

## Graph classification

Please run the scripts inside the `./data-augmentation-graphs` folder.

```
cd data-augmentaton-graphs
```

Use `train_tree_based_augment.py` to search for the composition of data augmentations: 

- `--max_depth` specifies the maximum depth of the tree composition. 

- `--task_set_name` specifies the filename recording the sampled subsets.
- `--save_name` specifies the filename that saves the evaluation results for the self-supervised model on each subset.

We provide an example to find a tree based composition in one graph dataset:

```bash
python train_tree_based_augment.py --dataset NCI1 --epochs 10\
    --task_set_name tree_NCI1 --save_name tree_NCI1\
    --device 0 --max_depth 4
```

Use `train.py` to train a contrastive model on TUDatasets. Please specify the following main parameters. We provide an example to run a contrastive learning model on a graph dataset:

```bash
python train.py --dataset NCI1 --train_simclr --loss_name nt_xnet_loss --semi_split 1 --epochs 100 --augmentation_names DropNodes PermuteEdges --augmentation_ratios 0.1 0.1 \
--fold_idxes 0 --mnt_metric val_loss --mnt_mode min\
--device 0 --save_name test
```

For protein function prediction, please run under the  `./data-augmentation-graphs/protein_graph_classification` folder. 

Similarly, use `train_tree_based_augment.py` to search for the composition of data augmentations. We provide a script to run the search for a tree composition of data augmentation below. 

```bash
python train_tree_based_augment.py --num_groups 4 --group_ids 0\
    --task_set_name tree_augment_group_0 --save_name tree_augment_group_0\
    --epochs 20 --device 0
```

Use `train.py` to train a model on the protein function prediction dataset. We provide an example below. 

```bash
python train.py --task_idxes -1 --num_groups 4 --group_ids -1 --labeling_rate_threshold 0.005 \
    --device 3 --runs 2 --epochs 50 --save_name results\
    --train_bilevel --weight_lr 0.5 --collect_gradient_step 4
```

## Data Preparation

- For wildlife image classification, please refer to [the WILDS benchmark](https://github.com/p-lambda/wilds) for downloading the iWildCam dataset. 

- For medical image datasets containing eye fundus images for diabetic retinopathy classification. Please refer to `./data-augmentation-images/README.md` for downloading and formatting the images.  Thanks to the authors of [BenchMD](https://github.com/rajpurkarlab/BenchMD#datasets) for providing their data processing code online.

- For the dataset for protein function prediction, we provide a subset of dataset for experimenting with the code in the folder of `./data-augmentation-graphs/protein-function-prediction/dataset`. Please unzip it before use.  

- For datasets from TUDataset, our code automatically downloads the data. Please create a `data` folder under the `./data-augmentation-graphs` folder. 

### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- **[GraphCL](https://github.com/Shen-Lab/GraphCL)**
- **[Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns)**
- [**BenchMD**](https://github.com/rajpurkarlab/BenchMD)
- **[pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)**
