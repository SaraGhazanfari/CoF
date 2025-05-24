# Chain-of-Frames: Advancing Video Understanding in Multimodal LLMs via Frame-Aware Reasoning

### [Paper](https://arxiv.org/abs/2412.10594) | [Dataset](#data) | [Checkpoints](#checkpoints) | [Quick Start](#quick) | [Acknowledgement](#ack) 

<p float="left">
  <img src="assets/teaser_acc.png" width="36.5%" />
  <img src="assets/teaser_exp.png" width="60%" /> 
</p>

In this work, we propose to obtain video LLMs whose reasoning steps are grounded in, and explicitly refer to, the relevant video frames.
For this, we first create \cofdata, a large dataset of diverse questions, answers, and corresponding frame-grounded reasoning traces about both natural and synthetic videos, spanning various topics and tasks.
Then, we fine-tune existing video LLMs on this chain-of-frames (CoF) data.
Our approach is simple and self-contained, and, unlike existing approaches for video CoT, does not require auxiliary networks to select or caption relevant frames.
We show that our models based on CoF are able to generate chain-of-thoughts that accurately refer to the key frames to answer the given question.


<a name="data"></a>
### Dataset
<p float="center">
<img src="assets/data_generation.png" width="100%">
</p>

<a name="checkpoints"></a>
### Checkpoints


### Quick Start


- Evaluation scripts for other metrics:
  
```
bash scripts/eval/eval.sh 
```

### Acknowledgement

This work leverages the code and resources from [InternVL](https://github.com/OpenGVLab/InternVL) repository. 

We thank the authors for making their work publicly available and contributing to the research community.

<a name="bibtex"></a>
### Citation
If you use our code or models, please consider citing our work using the following BibTex entry:
```
?
```
