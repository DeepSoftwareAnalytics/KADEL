# KADEL: Knowledge-Aware Denoising Learning for Commit Message Generation

This is the replication package for "[KADEL: Knowledge-Aware Denoising Learning for Commit Message Generation](https://arxiv.org/abs/2401.08376)" accepted to ACM Transactions on Software Engineering and Methodology (TOSEM).

Citation for this work can be found [here](#Citation).


## Environment

### Anaconda

```sh
conda create -n KADEL python==3.8.5 -y
conda activate KADEL
conda install pytorch==1.13 torchvision torchaudio cudatoolkit=10.0 -c pytorch -c nvidia -y
pip install tensorboard==2.4.1 tree-sitter==0.2.2 gsutil tqdm wandb notebook notebook gsutil scikit-learn protobuf protobuf==3.20.*
```

## Data

### Step 1: Download MCMD

```sh
# Downlaod MCMD filtered_data
cd $MCMD_PATH
mkdir filtered_data
wget https://zenodo.org/record/5025758/files/filtered_data.tar.gz
tar -zxvf filtered_data.tar.gz -C filtered_data/
```


### Step 2: Preprocessing (e.g.: Get the indexes of commit which match the template (`<type>(<scope>):<subject>`))

```sh
# Time Cost: about 5 min per programming language.
LANGUAGE=javascript
python preprocess_MCMD.py --MCMD_data_folder_path $MCMD_PATH//filtered_data/$LANGUAGE/sort_random_train80_valid10_test10
```

## Evaluation

Details can be seen at [evaluation_results.ipynb](evaluation_results.ipynb)

> Thanks for the evaluation scripts from [CommitMsgEmpirical](https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical)!

## Experiment

### Our Model(full version)

```bash
cd sh/
python run_KADEL.py
```

### Ablation Study

#### w/o knowledge version

```bash
cd sh/
python run_KADEL_without_knowledge.py
```

#### w/o denoising version

```bash
cd sh/
python run_KADEL_without_denoising.py
```

> Thanks for the code from [CodeT5](https://github.com/salesforce/CodeT5/tree/main/CodeT5)!


## Citation

If you use this code, please consider citing us:)

```bibtex
@article{KADEL_CMG_24,
  author    = {Wei Tao and
               Yucheng Zhou and
               Yanlin Wang and
               Hongyu Zhang and
               Haofen Wang and
               Wenqiang Zhang},
  title     = {KADEL: Knowledge-Aware Denoising Learning for Commit Message Generation},
  journal   = {{ACM} Trans. Softw. Eng. Methodol.},
  year      = {2024}
}
```
