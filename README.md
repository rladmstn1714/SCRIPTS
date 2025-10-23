<div align="center">
  <h1> Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues </h1>
  <p>
    <a href="https://arxiv.org/pdf/2510.19028">
      <img src="https://img.shields.io/badge/ArXiv-SCRIPTS-red" alt="Paper">
    </a>
    <a href="https://github.com/rladmstn1714/SCRIPTS">
      <img src="https://img.shields.io/badge/GitHub-Code-blue" alt="GitHub">
    </a>
    <a href="https://huggingface.co/datasets/EunsuKim/SCRIPTS">
      <img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow" alt="Hugging Face">
    </a>
    <a>
    </a>
  </p>
</div>


**Official repository for [Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](https://arxiv.org/pdf/2510.19028).**


## Structure

```
SCRIPTS/
├── dataset/           # Dataset files
│   ├── english_combined.csv    # English dialogue dataset
│   └── korean_combined.csv     # Korean dialogue dataset
│
├── experiment/        # Experiment code (see experiment/README.md)
│   ├── core/         # Evaluation, prompts, utilities
│   ├── scoring/      # Scoring and parsing
│   ├── analysis/     # Analysis scripts
│   ├── models/       # LLM model wrappers
│   └── config.py     # Configuration
│
└── results/          # Generated results (auto-created)
```

## Usage

See `experiment/README.md` for detailed documentation, API reference, and usage examples.


## License

CC-BY-NC-ND

## Citation
```
@misc{kim2025loversfriendsevaluatingllms,
      title={Are they lovers or friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues}, 
      author={Eunsu Kim and Junyeong Park and Juhyun Oh and Kiwoong Park and Seyoung Song and A. Seza Dogruoz and Najoung Kim and Alice Oh},
      year={2025},
      eprint={2510.19028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19028}, 
}
```
