# ddpo-pytorch-eval

This is a customized implementation derived from [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch) and [LLaVA-server](https://github.com/kvablack/LLaVA-server). It is designed to evaluate the effectiveness of the prompt-image alignment reward function in DDPO.

<p align="center">
    <img src="teaser.jpg" width="75%">
</p>

## Installation
Requires Python 3.10 or newer.

```bash
git clone git@github.com:kvablack/ddpo-pytorch.git
cd ddpo-pytorch
pip install -e .
```

## Usage
Before launching the DDPO training process, if you want to run the LLaVA prompt-image alignment experiments, you need to allocate several GPUs to run LLaVA inference using ./LLaVA-server. Detailed instructions can be found in the README file within the ./LLaVA-server directory.

The DDPO training configurations are provided in `config/dgx.py`. To run the prompt-image alignment experiment, execute:
```bash
accelerate launch scripts/train.py --config config/dgx.py:prompt_image_alignment
```

## Prompt-image Alignment Reward Curves in DDPO Official Repository
<p align="center">
    <img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/393a929e-36af-46f2-8022-33384bdae1c8" width="75%">
</p>

## Prompt-image Alignment Reward Curves in Our Evaluation Experiment
<p align="center">
    <img src="assets/reward_curve.png" width="75%">
</p>
