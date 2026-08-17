# FlowDec: Temporal Conditional Flow Decorruptor for Robust Continuous Vision Language Navigation [ECCV 2026]

<a href="https://arxiv.org/abs/2606.22424">
  <img src="https://img.shields.io/badge/Paper-Arxiv-red">
</a>
<a href="https://drive.google.com/drive/u/0/folders/1XxXNlipsmabN3C1_xlIWG68a02nLLa4s">
  <img src="https://img.shields.io/badge/Checkpoint-Drive-yellow">
</a>

---
![image](https://github.com/FeeFee-1/FlowDec-Temporal-Conditional-Flow-Decorruptor-for-Robust-Continuous-Vision-Language-Navigation/blob/main/figs/1.png)
FlowDec is a novel image restoration framework tailored for LM-based VLN-CE. FlowDec integrates a hybrid temporal conditioning strategy to align the generative flow path with historical context and employs action-centroid guided filtering to dynamically assess and integrate outputs. 

## Experimental Results
![image](https://github.com/FeeFee-1/FlowDec-Temporal-Conditional-Flow-Decorruptor-for-Robust-Continuous-Vision-Language-Navigation/blob/main/figs/4.png)

## Deployment
### Install Dependencies
```bash
sudo apt update
sudo apt install -y imagemagick libmagickwand-dev
conda env create -f environment.yml
```
### Download Checkpoints
Download the pre-trained checkpoints from [google drive](https://drive.google.com/drive/u/0/folders/1XxXNlipsmabN3C1_xlIWG68a02nLLa4s) and put them under `/checkpoint`.

### Run Evaluation
```python
conda activate flowdec
python test.py --c fog # choose corruption type
```

### Other
To train the model from scratch, please follow the augmentation settings described in [Decorruptor](https://github.com/oyt9306/Decorruptor).

To deploy FlowDec as the visual input stage of [Navid](https://github.com/jzhzhang/NaVid-VLN-CE), please first clone the NaVid repository, then integrate FlowDec at the visual encoding stage.


