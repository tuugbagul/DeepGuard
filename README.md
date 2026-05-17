# DeepGuard

DeepGuard is a Gradio demo that predicts whether a face image is a deepfake. On the model side, it uses a fusion architecture combining `EfficientNet-B4` with `Xception + FFT + Cross-Attention`.

This repo is now primarily focused on the **demo/inference** experience. Training, data preparation, and evaluation scripts are still present but are intended for advanced/research use.

## Quick Start

```bash
git clone https://github.com/tuugbagul/DeepGuard.git
cd DeepGuard

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python demo.py
```

On Windows, you can also run directly:

```bat
run_demo.bat
```

Once the demo launches, navigate to `http://127.0.0.1:7860` in your browser. Upload a face image to see the model prediction and GradCAM heatmap.

## Weight Resolution

The demo searches for `fusion_finetuned.pth` in the following order:

1. Custom path provided via `--weights`
2. `fusion_finetuned.pth` in the repo root
3. `weights/fusion_finetuned.pth`
4. GitHub Release asset download

Default release URL template:

```text
https://github.com/tuugbagul/DeepGuard/releases/latest/download/fusion_finetuned.pth
```

If the release asset hasn't been published yet, there are two easy options:

```bash
python demo.py --weights path/to/fusion_finetuned.pth
```

or

```bash
set DEEPGUARD_WEIGHTS_URL=https://your-direct-file-url/fusion_finetuned.pth
python demo.py
```

For PowerShell:

```powershell
$env:DEEPGUARD_WEIGHTS_URL="https://your-direct-file-url/fusion_finetuned.pth"
python demo.py
```

## Common Commands

```bash
python demo.py
python demo.py --no-download
python demo.py --weights weights/fusion_finetuned.pth
python demo.py --host 0.0.0.0 --port 7860
```

## Model Results

| Model | Overall Accuracy | Real | Fake |
|-------|-----------------|------|------|
| EfficientNet-B4 (FaceForensics++ pretrain) | 59.60% | 30.00% | 89.20% |
| EfficientNet-B4 (Celeb-DF fine-tune) | 96.80% | 94.40% | 99.20% |
| EfficientNet-B4 (Roop fine-tune) | 89.38% | 74.36% | 94.21% |
| **Fusion Model** | **98.12%** | **94.87%** | **99.17%** |

## Architecture

```text
Input (299x299 RGB)
    |
    |-- EfficientNet-B4 -> 1792-dim features
    |
    `-- Xception + FFT + Cross-Attention
            |-- Xception        -> 2048-dim
            |-- FFT Branch      ->  512-dim
            `-- Cross-Attention ->  256-dim
                                   = 2304-dim features

concat [1792 + 2304 = 4096-dim]
    -> FC(512) -> BN -> ReLU -> Dropout(0.4)
    -> FC(1) -> sigmoid -> fakeness score
```

## Datasets

- **FaceForensics++**: Deepfakes, Face2Face, FaceSwap, NeuralTextures, and real YouTube videos
- **Celeb-DF**: Celebrity deepfake videos
- **Custom Roop dataset**: Fake + real pairs generated with [Roop](https://github.com/s0md3v/roop) on LFW faces

## Repo Structure

```text
DeepGuard/
├── demo.py               # Main Gradio demo entry point
├── weights.py            # Weight resolution / download helpers
├── fusion_model.py       # Fusion model architecture
├── pixelguard_model.py   # PixelGuard branch
├── train/                # Training scripts
├── eval/                 # Evaluation scripts
├── data/                 # Data preparation scripts
├── visualize/            # GradCAM and reporting tools
└── roop_scripts/         # Roop generation / utility scripts
```

## Research Scripts

These folders are not part of the demo flow:

- `train/`
- `eval/`
- `data/`
- `visualize/`
- `roop_scripts/`

They are kept for model development and experimentation. For initial setup, you only need to care about `demo.py`.

## License

This project is for academic purposes.
