# Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation

![denosing process](./assets/denosing.png)

We proposes a lightweight diffusion time prediction module that forecasts optimal denoising noise level.
Without modifying original model parameters, we efficiently optimize the entire denoising process using RL, accelerating generation speed and enhancing output quality for diffusion/flow models. The approach demonstrates improvements in human preference alignment
and text-image matching metrics.

## Pretrained Model download

```shell
mkdir models
pip install -r requirements.txt

# if you are in mainland china, you can use the mirror to accelerate download
# export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stabilityai/stable-diffusion-3-medium --local-dir models/stabilityai/stable-diffusion-3-medium
```

## Download Checkpoints

```shell
huggingface-cli download MAPLE-WestLake-AIGC/TPDM --local-dir checkpoint
# subdir sd3 is stable diffusion 3 checkpoint
```

## Launch Gradio Web For Inference

```python
python gradio_sd3_inference.py
```

## Training model

If you want to train TPDM, you should prepare prompts and organize it like exmaple/example.jsonl.

Origin datasets we used can be download in [COCO](https://cocodataset.org/#home), [COYO-11M](https://huggingface.co/datasets/CaptionEmporium/coyo-hd-11m-llavanext) and [Laion-Art](https://huggingface.co/datasets/laion/laion-art)

```shell
huggingface-cli download --resume-download THUDM/ImageReward --local-dir models/THUDM/ImageReward
bash scripts/launch_sd3_train.sh
```
## Acknowledgement
Thanks to huggingface team for open-sourcing the [trl](https://github.com/huggingface/trl) library, which part of our code is based on.

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