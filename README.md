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

```shell
huggingface-cli download --resume-download THUDM/ImageReward --local-dir models/THUDM/ImageReward
bash scripts/launch_sd3_train
```

