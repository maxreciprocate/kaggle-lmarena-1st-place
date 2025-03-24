## WSDM Cup - Multilingual Chatbot Arena

https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena

## Requirements

### Hardware

H100x8 (to speed up teachers training we used H100x32, however it's not required)

### Software

```
python3.12
SLURM
```

### Packages
```
deepspeed==0.15.4
accelerate==1.2.1
transformers==4.46.3
flash-attn==2.7.1.post4
```

## Reproduction 

### Preparation

Download the following datasets and put them into `data/` without renaming the files (the rest of datasets are available on huggingface):
- [dataset for the challenge](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/data) 
- Datasets from @nbroad 🤗 ([v1, v2 and v3](https://www.kaggle.com/datasets/nbroad/wsdm-open-models-nbroad))
- [lmarena-ai/gpt-4o-mini_battles](https://huggingface.co/spaces/lmarena-ai/gpt-4o-mini_battles/)

### Training

- [Stage 1. Pretraining](./stage1)
- [Stage 2. Teachers training](./stage2)
- [Stage 3. Distillation](./stage3)

## Structure

All scripts have to be executed from the root directory

```
├── data
│   ├── train.parquet
│   └── ...
├── ckpts
│   └── ...
├── stage1
│   ├── README.md
│   └── prepare_pretrain_data.py
├── stage2
│   ├── README.md
│   └── prepare_teacher_data.py
├── stage3
│   ├── README.md
│   ├── collect_labels.py
│   ├── merge_students.py
│   ├── pack_student.py
│   ├── prepare_student_data.py
│   └── prepare_synth_data.py
├── packing
│   └── # https://github.com/tascj/kaggle-lmsys-chatbot-arena/tree/main/human_pref
├── format.py
├── label.py
├── label.sh
├── label.slurm
├── launch.slurm
├── models.py
├── readme.md
├── requirements.txt
├── run.sh
└── train.py
```
