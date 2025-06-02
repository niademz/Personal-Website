# Emotion-Responsive Poetry Page 

This project is an interactive poetry reader built into my personal website. It dynamically styles and reacts to the emotional tone of each poem, using two different neural network models trained to assign intensity scores across 12 core emotions.

>  The rest of the website (`index.html`, etc.) is static, but this `poetry.html` page is a self-contained experiment in affective computing and creative NLP.

---

##  Emotion Models

There are two available models for scoring poem lines:

### 1. Bag-of-Words (BoW)
A basic feedforward neural network using a fixed vocabulary. It scores emotions by counting word frequencies and feeding the resulting vector into a hidden layer.

- Trained on 54 poems (*The Wild Iris* by Louise Glück), manually annotated for 12 emotion intensities.
- Output range: `[0, 10]` per emotion.
- Fast and lightweight, but lacks nuance.

### 2. Semantic Model (TF-IDF + GloVe)
A more advanced model that embeds each word into a 100-dimensional semantic vector (via GloVe) and uses TF-IDF weighting to compute a document vector.

- TF-IDF is approximated using static IDF values.
- The resulting weighted embedding is passed into a separate neural network.
- Captures semantic and emotional nuance much better than BoW.

> Due to GloVe's file size, the actual vectors are not included in this repository. You must download the appropriate file manually (see below).

---

##  Directory Structure

/
├── poetry.html # Main emotion-aware poetry page
├── index.html # Static homepage
├── data/
│ ├── poems.json # my poems with title and text
│ ├── vocab.json # Vocabulary list used by BoW model
│ └── [glove-mini.json] # mini GloVe dictionary of word vectors of all the words in Louise Gluck poems and 
├── models/
│ ├── bow_model.json # Trained BoW model
│ └── w2v_model.json # Trained semantic model
└── scripts/
└── infer-browser.js # In-browser inference logic (shared by both models)


## Running Locally

You don’t need any backend or build tools — this project is designed to run directly in a browser.

### 1. Download GloVe Vectors

Download `glove.6B.100d.txt` from the official GloVe website:
https://nlp.stanford.edu/projects/glove/
add it to an embeddings directory.  

2. Open the Page
Simply open poetry.html in a browser with a live server(e.g. Chrome). That's it!

🧾 Emotion Schema
Each poem is scored using a 12-emotion scale:


[ Happiness, Anger, Joy, Sadness, Fear, Surprise, Nostalgia,
  Longing, Hope, Contempt, Awe, Melancholy ]
Scores range from 0.0 (absent) to 10.0 (intense). These values drive both:

Line-by-line emotional highlighting

Global page theming (color, animation, overlays)

Citing GloVe
This project uses GloVe for semantic embeddings.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
2014. GloVe: Global Vectors for Word Representation.
PDF

Future Plans
Add an info icon (ℹ️) or hover popup on the poetry page to let users explore how emotion detection works

Expand training dataset

Include per-line emotion explanations or overlays

📄 License
This poetry page and its models are part of my personal website and are not licensed for redistribution without permission. For academic or personal experimentation, feel free to explore and adapt.

Let me know if you want:
- A Python script (`glove_to_json.py`) to extract the mini glove file.
- To tweak the tone — more academic, more casual, more poetic?
- To add a section for “Contributors” if you had any collaborators or want to credit your inspirations.

When you’re ready, we can jump to designing that elegant little ℹ️ info icon with a hover/fade popup too.
