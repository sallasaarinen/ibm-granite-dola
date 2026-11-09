# ibm-granite-dola

DoLa decoding evaluation with Granite 4.2 3B

[`DoLa_evaluation.ipynb`](DoLa_evaluation.ipynb) evaluates **DoLa** (Decoding by Contrasting Layers) against standard greedy decoding, using the `ibm-granite/granite-4.2-3b` model on a small set of benchmark questions. It was built for Google Colab with a T4 GPU.

## What the notebook does

1. **Installs and imports requirements** (`transformers`, `torch`, `scipy`, `matplotlib`).
2. **Loads the model** in `float16` with automatic device placement.
3. **Compares three decoding methods** on one prompt: standard, DoLa with dynamic layer selection, and DoLa with a static layer.
4. **Analyses first-token candidates.** A helper function returns the top four candidate tokens and their probabilities under standard or DoLa decoding, and the notebook plots them.
5. **Plots layer divergence.** It calculates the Jensen-Shannon divergence between the final layer and each earlier layer for every generated token, and shows it as a heatmap. It also compares the top candidate tokens at the final layer with those at an earlier layer.
6. **Runs a small benchmark.** A helper function takes a list of questions and, for each one, prints the DoLa answer and the standard answer with the latency per token for each.

## Requirements

- A GPU runtime
- Python packages: `transformers`, `torch`, `numpy`, `scipy`, `matplotlib`
- Internet access to download the model and the DoLa implementation (`transformers-community/dola`) from the Hugging Face Hub
