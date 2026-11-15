# Natural Language Processing (NLP) — Python Notebooks

This repository contains three Jupyter notebooks demonstrating foundational Natural Language Processing (NLP) techniques using **NLTK**, **spaCy**, and **NumPy**. Each notebook focuses on a separate concept, from preparing text to identifying entities and implementing the attention mechanism used in Transformers.

## Projects

| Notebook | Topic | What it demonstrates |
|---|---|---|
| **Notebook 1 — NLP Preprocessing** | Tokenization, stopword removal, stemming | Converts a sentence into tokens, filters common English stopwords and punctuation, and applies Porter stemming. |
| **Notebook 2 — Named Entity Recognition** | spaCy NER | Detects named entities and prints their text, label, and start/end character positions. |
| **Notebook 3 — Scaled Dot-Product Attention** | Transformer attention | Calculates scaled query–key scores, softmax attention weights, and a weighted output using values. |

> **Note:** The notebook titles above describe their contents. Replace them with your actual `.ipynb` filenames if you add links in this README.

## 1. NLP Preprocessing with NLTK

**Input sentence:**

> NLP techniques are used in virtual assistants like Alexa and Siri.

The notebook performs three steps:
1. **Tokenization:** Splits the sentence into words and punctuation.
2. **Stopword removal:** Filters common words such as *are*, *in*, and *and*, as well as punctuation.
3. **Stemming:** Applies the Porter Stemmer to reduce words to stem forms.

It displays **Original Tokens**, **Tokens Without Stopwords**, and **Stemmed Words**. Stems are not always dictionary words (for example, *techniques* becomes *techniqu*).

## 2. Named Entity Recognition with spaCy

**Input sentence:**

> LeBron James as the NBA player of the United States in Laker and won the NBA Final MVP in 2020.

The notebook uses spaCy's `en_core_web_sm` English model to identify entities such as people, organizations, locations, and dates. For each detected entity, it prints:
- **Entity:** The extracted text (`ent.text`)
- **Label:** The entity category (`ent.label_`)
- **Start:** Starting character index (`ent.start_char`)
- **End:** Ending character index (`ent.end_char`, exclusive)

Entity detection and labels may vary slightly by spaCy model version.

## 3. Scaled Dot-Product Attention with NumPy

This notebook implements the attention operation used in Transformer models:

### Attention Formula

Attention(Q, K, V) = softmax((Q × Kᵀ) / √dₖ) × V

Where:
- **Q (Query):** Represents the information being searched for.
- **K (Key):** Represents the information used to calculate attention scores.
- **V (Value):** Contains the information used to generate the output.
- **dₖ:** Dimension of the key vectors.

It:
1. Computes query–key similarity scores (`Q @ K.T`).
2. Scales the scores by the square root of the key dimension.
3. Applies a numerically stable softmax to get attention weights.
4. Multiplies those weights by `V` to obtain the final output.

**Test matrices:**

```python
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
```

**Expected output (rounded to three decimal places):**

```text
Attention Weights Matrix:
[[0.731 0.269]
 [0.269 0.731]]

Final Output Matrix:
[[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]
```

## Setup and Installation

**Requirements:** Python, Jupyter Notebook or JupyterLab, and the libraries listed below.

Create and activate a virtual environment to avoid conflicts with other Python projects:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install notebook numpy nltk spacy ipykernel
python -m spacy download en_core_web_sm
```

The NLTK preprocessing notebook also requires tokenizer and stopword data. Run once in Python or a notebook cell:

```python
import nltk
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Running the Notebooks

1. Open the repository folder in a terminal.
2. Activate the virtual environment created above.
3. Start Jupyter with `jupyter notebook`, or open the `.ipynb` files in PyCharm or JupyterLab.
4. Select the Python interpreter/kernel from this virtual environment.
5. Run each notebook's cells from top to bottom.


## Key Concepts Learned

- **Text preprocessing** prepares raw sentences for downstream NLP tasks.
- **Named Entity Recognition** identifies and categorizes entities in text.
- **Scaled dot-product attention** lets a model assign different levels of importance to input representations.

## Technologies

Python · Jupyter Notebook · NLTK · spaCy · NumPy
