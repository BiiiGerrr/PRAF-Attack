# PRAF-Attack

> This repository provides only **100 sample pairs** for demonstration. The actual evaluation uses **1,000 samples**, which can be downloaded from the [FOA-Attack project](https://github.com/jiaxiaojunQAQ/FOA-Attack).

## Setup

Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Steps to Run

### 1. Run PRAF-Attack

```bash
python 1_PRAF-Attack.py
```

### 2. Generate Image Descriptions (Black-box Text Generation)

Generate descriptions for both adversarial examples and target samples. Use `--models` to select the API model (default: `gpt-5.4`). You must also provide `--base_url` and `--api_key`:

```bash
python 2_blackbox_text_generation.py --models gpt-5.4 --base_url YOUR_BASE_URL --api_key YOUR_API_KEY
```

### 3. GPTScore Evaluation

Evaluate the generated description pairs to compute **AvgSim** and **ASR**. You must also provide `--base_url` and `--api_key`:

```bash
python 3_llm_gpt_evaluate.py --base_url YOUR_BASE_URL --api_key YOUR_API_KEY
```

> Replace `adversarial_descriptions.txt`, `target_descriptions.txt`, `YOUR_BASE_URL`, and `YOUR_API_KEY` with your actual file paths and credentials.

## Output Metrics

- **AvgSim**: Average similarity score
- **ASR**: Attack Success Rate

## Notes

- Ensure valid API access for the model used in Step 2
- The description file paths in Step 3 must match your actual generated files
- Do not commit hardcoded `api_key` to version control; consider using environment variables

## Acknowledgements

This project is built on [VILA-Lab/M-Attack](https://github.com/VILA-Lab/M-Attack). We sincerely thank them for their outstanding work.

Our data preparation and evaluation also refer to [jiaxiaojunQAQ/FOA-Attack](https://github.com/jiaxiaojunQAQ/FOA-Attack) and [LiYuanBoJNU/MPCAttack](https://github.com/LiYuanBoJNU/MPCAttack). We sincerely thank the authors for their valuable contributions.

## Contact

If you have any questions, please contact me at [wanghb69@mail2.sysu.edu.cn](mailto:wanghb69@mail2.sysu.edu.cn).
