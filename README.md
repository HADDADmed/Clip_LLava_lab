# CLIP & LLaVA Laboratory

Vision-language models lab: zero-shot classification and retrieval with **CLIP**, and visual question answering and hallucination analysis with **LLaVA**.

**Main deliverable:** [`lab/lab_clip_llava.ipynb`](lab/lab_clip_llava.ipynb). It contains the code, the outputs and the answers to Q1–Q36 for every exercise.

## Deliverables

| # | Deliverable | Where | Result |
|---|---|---|---|
| 1 | Jupyter notebook | [`lab/lab_clip_llava.ipynb`](lab/lab_clip_llava.ipynb) | all cells executed, outputs saved |
| 2 | Code and outputs for every experiment | notebook, Ex 1–13 | |
| 3 | CLIP classification accuracy | Ex 5 · [`lab/results/clip_eval.csv`](lab/results/clip_eval.csv) | **93.3 %** (14/15) |
| 4 | Top-5 retrieval results | Ex 8 · [`lab/results/top5_retrieval.png`](lab/results/top5_retrieval.png) | top 3 = football, a rugby scene at #4 |
| 5 | LLaVA VQA results | Ex 11 · [`lab/results/vqa_results.csv`](lab/results/vqa_results.csv) | **68 %** (17/25) |
| 6 | Hallucination example | Ex 12 (and Ex 10, 11) | invents a "red Schwinn" bicycle when the prompt asserts one exists |
| 7 | CLIP/LLaVA comparison | Ex 13 | comparison table and Q33–Q36 |
| 8 | Mini-project | [`mini_project/assistant.py`](mini_project/assistant.py) | image + question → answer; bonus: CLIP top-K → LLaVA explanation |
| 9 | Conclusion | end of the notebook | |

![Top-5 retrieval](lab/results/top5_retrieval.png)

## Models

| Model | Use |
|---|---|
| `openai/clip-vit-base-patch32` | Ex 1–8, 13 |
| `llava-hf/llava-v1.6-mistral-7b-hf` (LLaVA-NeXT: CLIP ViT-L/14-336 + MLP projector + Mistral-7B) | Ex 9–13, mini-project |

Both run in bf16 on a single 32 GB NVIDIA GPU. LLaVA-NeXT uses about 16 GB of GPU memory. Generation is greedy (`do_sample=False`), so the answers are reproducible.

## Data

All images are from **COCO val2017**. They were chosen per exercise using COCO's own annotations and captions. Ids and licences are in [`lab/images/SOURCES.md`](lab/images/SOURCES.md).

| Folder | Content |
|---|---|
| `lab/image.jpg` | the statement's image (two cats), used in Ex 2–4, 6, 13 |
| `lab/images/ex05_eval/` | 15 labelled images, 6 classes, including hard cases |
| `lab/images/ex07_retrieval/` | 20 mixed images: 4 football scenes and 1 rugby distractor |
| `lab/images/ex11_vqa/` | 5 scenes with people and objects, including a classroom |
| `lab/images/ex12_hallucination/` | a street scene with no bicycle |

## Repository layout

```
lab/                 notebook, images, results/
mini_project/        assistant.py (final mini-project)
clip/  llava/        standalone scripts: CLIP zero-shot, LLaVA chat
common/              shared helpers
```

## Setup and run

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 build
uv pip install -r requirements.txt
jupyter lab lab/lab_clip_llava.ipynb
```

The mini-project from the command line:

```bash
python mini_project/assistant.py --image lab/images/ex11_vqa/kitchen1.jpg --question "What is the man doing?"
python mini_project/assistant.py --query "a person playing football" --image-dir lab/images/ex07_retrieval --k 3
```

Validated with torch 2.11.0+cu128 and transformers 5.20.1. The full list of pinned versions is in `requirements.lock.txt`.
