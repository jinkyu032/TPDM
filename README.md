<div align="center">

<h1>Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation</h1>

<p align="center">
<a href="https://arxiv.org/abs/2412.01243"><img src="https://img.shields.io/badge/arXiv-2412.01243-b31b1b.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/MAPLE-WestLake-AIGC/TPDM"><img src="https://img.shields.io/badge/Checkpoint-Huggingface-yellow" alt="Checkpoint"></a>
</p>

[MAPLE Lab, Westlake University](https://maple.lab.westlake.edu.cn/)

</div>

![denosing process](./assets/denosing.png)

We proposes a lightweight diffusion time prediction module that forecasts optimal denoising noise level.
Without modifying original model parameters, we efficiently optimize the entire denoising process using RL, accelerating generation speed and enhancing output quality for diffusion/flow models. The approach demonstrates improvements in human preference alignment
and text-image matching metrics.

## Visualization
![examples](./assets/examples.png)

## Getting start for inference
### Download SD3 Pretrained Model

```shell
mkdir models
pip install -r requirements.txt

# if you are in mainland china, you can use the mirror to accelerate download
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stabilityai/stable-diffusion-3-medium --local-dir models/stabilityai/stable-diffusion-3-medium
```

### Download TPM Checkpoints

```shell
huggingface-cli download MAPLE-WestLake-AIGC/TPDM --local-dir checkpoint
# subdir sd3 is stable diffusion 3 checkpoint
```

### Launch Gradio Web For Inference

```python
python gradio_sd3_inference.py
```

## Getting start for training

If you want to train TPDM, you should prepare prompts and organize it like exmaple/example.jsonl.

Origin datasets we used can be download in [COCO](https://cocodataset.org/#home), [COYO-11M](https://huggingface.co/datasets/CaptionEmporium/coyo-hd-11m-llavanext) and [Laion-Art](https://huggingface.co/datasets/laion/laion-art)

```shell
huggingface-cli download --resume-download THUDM/ImageReward --local-dir models/THUDM/ImageReward
bash scripts/launch_sd3_train.sh
```

## Acknowledgement
Thanks to huggingface team for open-sourcing the [trl](https://github.com/huggingface/trl) and [diffusers](https://github.com/huggingface/diffusers) library, which part of our code is based on.

## Citation
If you find our paper or code useful, wish you can cite our paper.
```
@misc{ye2025scheduleflydiffusiontime,
      title={Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation}, 
      author={Zilyu Ye and Zhiyang Chen and Tiancheng Li and Zemin Huang and Weijian Luo and Guo-Jun Qi},
      year={2025},
      eprint={2412.01243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01243}, 
}
```
