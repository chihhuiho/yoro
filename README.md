# General
This is the official code for YORO - Lightweight End to End Visual Grounding, accepted by European Conference On Computer Vision (ECCV) Workshop on International Challenge on Compositional and Multimodal Perception, Tel-Aviv, Israel, 2022


# Evironment
Use environment/environment.yml or environment/environment_cuda102.yml depending on cuda version for creating the environment.
```
conda env create -f environment/environment.yml 
conda activate yoro
python -m spacy download en_core_web_sm
```

# Datasets
1. Comment out the dataset in download.sh that is not needed. It takes few hours to download all datasets
2. Download the dataset to the "./dataset/raw" folder by 
```
sh download.sh
```


# Preprocess Dataset
Converting the raw data to arrow format.
1. Comment out the dataset in preprocess_dataset.py that is not needed
```
python preprocess_dataset.py
```
2. The preprocessed dataset will be stored in "./dataset/arrow"

# Download Pretrained weight
## Vilt weight for pretraining
```
cd pretrained_weight
sh download_weight.sh
```
## Yoro weight for various VG tasks
1. Download result.zip from google drive [google drive](https://drive.google.com/file/d/1dqwT-YXmVdyUkPPLfm-D3hHfmCFxfk7j/view?usp=share_link)
2. unzip the result.zip


# Evaluation
For each eval.sh file in the script/DATASET, change the flag "debug" to False to run full evaluation. Below, we will describe how to run the eval.sh for different datasets.

## Pretraining tasks
```
sh script/pretrain/eval.sh
```
## Downstream tasks

### RefCoco Dataset
```
sh script/RefCoco/eval.sh
```
### RefCoco+ Dataset
```
sh script/RefCocoP/eval.sh
```
### RefCocog Dataset
```
sh script/RefCocog/eval.sh
```
### CopsRef Dataset
```
sh script/copsref/eval.sh
```
### ReferItGame/RefClef Dataset
```
sh script/ReferItGame/eval.sh
```


# Training
For all run.sh file, please change the "debug" flag to True to run the full training. 

## Pretraining tasks
For Modulated detection pretraining, we start from a mlm-itm pretrained model, such as the vilt pretraining checkpoint. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/pretrain/run.sh 5 40 1
```

## Downstream tasks

### RefCoco Dataset
For RefCoco dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCoco/run.sh 5 10 1
```
### RefCoco+ Dataset
For RefCoco+ dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCocoP/run.sh 5 10 1
```
### RefCocog Dataset
For RefCocog dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 10 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/RefCocog/run.sh 5 10 1
```
### CopsRef Dataset
For copsref dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/copsref/run.sh 5 40 1
```
### ReferItGame/RefClef Dataset
For ReferItGame/RefClef dataset, we load the pretraining checkpoints as initial weight. For example, the below script is for training with 5 det tokens for 40 epochs on 1 gpu. Please refer to the comment in the script for more details.
```
sh script/ReferItGame/run.sh 5 40 1
```

## Citation
If you find this method useful in your research, please cite this article:
```
@inproceedings{ho2022yoro,
  title={YORO-Lightweight End to End Visual Grounding},
  author={Ho, Chih-Hui and Appalaraju, Srikar and Jasani, Bhavan and Manmatha, R and Vasconcelos, Nuno},
  booktitle={ECCV 2022 Workshop on International Challenge on Compositional and Multimodal Perception},
  year={2022}
}
```


# Acknowledgement
Please email to Chih-Hui (John) Ho (chh279@eng.ucsd.edu) if further issues are encountered. We heavily used the code from 
1. https://github.com/dandelin/ViLT
2. https://github.com/ashkamath/mdetr
