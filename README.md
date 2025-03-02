# Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation

## Pretrained Model download

```shell
mkdir models
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stabilityai/stable-diffusion-3-medium --local-dir models/stabilityai/stable-diffusion-3-medium
```

## Download Checkpoints

```shell
export HF_ENDPOINT=https://hf-mirror.com
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

