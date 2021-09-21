# EMNLP 2019: Dialog and Interactive Systems

https://2019.emnlp.org/program/schedule/


## [Improving Out-of-Scope Detection in Intent Classification by Using Embeddings of the Word Graph Space of the Classes](https://aclanthology.org/2020.emnlp-main.324.pdf)

### Task

Out of Scope Intent Detection

### Data

LARSON dataset: 22.5k in-scope (evenly distributed across 150 classes) and 1200 out-of-scope.
Other 40 dialogue datasets: 20 (en), 20 (pt-br)

### Approach

- A word graph is used to represent words and intent classes (occuring in training) -> class embeddings of these are computed
- and compared with the sentence embeddings of the input sentence (TF-IDF, Glove, LSTM & BERT) : Precisely, the inverse of the class embedding is used to map from an embedding to the class
Note that DeepWalk can convert text to vector and vice versa

### Evaluation

False Acceptance Rate (FAR) = Number of accepted OOS samples/Total of OOS samples | FAR reduced from previous SOTA from ~42% to ~10%
False rejection rate (FRR) = Number of rejected IS samples/Total of IS samples


## [Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems](https://aclanthology.org/2020.emnlp-main.326.pdf)

### Task

Human Evaluation of Dialogue: hard problem since for each context, there can be many answers as well as humans are erroneous in their evaluations
--> Rank a set of bots

### Data

DailyDialog, Empathetic Dialogue, PersonaChat (Blender model)

### Approach
Create bot-to-bot conversations between every pair of bots.
Human evaluators should label if two agents talking to each other are bots or not, at intervals of turns.

### Evaluation

1) Win Function: Meausre how long will a bot survive until recognized as a bot? 
- Compute pairwise table and then compute ranking using Bootstrap sampling
2) Survival rate : How long did the bot survive?


## [Variational Hierarchical Dialog Autoencoder for Dialog State Tracking Data Augmentation](https://aclanthology.org/2020.emnlp-main.274.pdf)

### Task

To Augment data for Dialog State Tracking using a Generative Model: GDA (Generative Data Modelling)

### Data

WoZ2.0, DSTC2, Multi-WoZ , DialEdit

### Approach

Variational Heirarchical Dialog AutoEncoder
- alleviate posterior collapse by modifying VAE objective
- use Neural estimation (as against Monte Carlo estimation)
- Incorporate hierarchial dropout
- Latent Space Interpolation done between 2 global latent variables (i.e. corresponding to 2 conversations)

### Evaluation

1. Performance on downstream task of Dialogue State Tracking: Joint goal accuracy, Turn level inform DA accuracy, request DA accuracy, etc.
2. Diversity Evaluation (no. of Unigrams or no. of dialog acts)

## [Automatically Learning Data Augmentation Policies for Dialogue Tasks](https://aclanthology.org/D19-1132.pdf)

### Task

Data Augmentation for Dialogue Response Generation

### Data

Ubuntu Dialoge Corpus

### Approach

Adapted AutoAugment (used in Computer Vision) for text: 
- Algorithm searches for optimal pertubation policies via a controller trained via RL
- Reward: comes from training the model with the sampled augmentation policy
- Each policy contains 2 sub-policies randomly sampled during training 
   - Each subpolicy contains 2 operations applied in sequence
          - Operations include grammar errors, verb inflexions, word repetitions
### Evaluation

Automatic Evaluation: Entity F1, Activity F1 (overlap of technical nouns & technical verbs between the generated response and the target response)
Human evaluation


## [Title of Paper 5]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


## [Title of Paper 6]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


## [Title of Paper 7]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


## [Title of Paper 8]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


## [Title of Paper 9]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


## [Title of Paper 10]()

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.


