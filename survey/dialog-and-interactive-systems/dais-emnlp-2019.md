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


## [Dialog Intent Induction with Deep Multi-View Clustering](https://aclanthology.org/D19-1413.pdf)

### Task

Discovering user intents from user query utterances in human-human conversations such as dialogs between customer support agents and customers (also taking into account the rest of the dialog)

### Data
1. Human human conversations of Twitter airline customer support (43k dialogs) -> 500 conversations randomly sampled annotated with 14 intents
2. Question Intent Clustering dataset based on AskUbuntu (which was designed for question duplication detection)

### Approach

Alternating-view k-means (proposed in the paper) AV-KMEANS uses different neural encoders to embed the inputs corresponding to two views: the user query utterance and the rest of the conversation.

### Evaluation

Precision, recall, F1 score, and unsupervised clustering accuracy (ACC) after setting the number of clusters manually.

## [A Practical Dialogue-Act-Driven Conversation Model for Multi-Turn Response Selection](https://aclanthology.org/D19-1205.pdf)

### Task

End-to-end multi-task model for conversation modeling, optimized for 1) dialogue act prediction and 2)response selection

### Data

DailyDialog and SwitchBoard Dialogue Act Corpus

### Approach

Use the previous utterances (context) and the predicted dialogue acts of both the context and the response to select a response from a given set of candidate responses.

### Evaluation

1) Dialogue Act Accuracy 2) MRR response selection

## [A Semi-Supervised Stable Variational Network for Promoting Replier-Consistency in Dialogue Generation](https://aclanthology.org/D19-1200.pdf)

### Task

Dialogue Generation

### Data

Cornell Movie Dialogs Corpus,Ubuntu Dialogue Corpus

### Approach

1) a semi-supervised stable variational network -> to promote replier consistency
2) unsupervised personal feature extractor -> to acquire replier specific information

### Evaluation

Automatic: The Distinct n-grams and its ratio over all generated responses.
Manual: How much of the response is not only semantically related and informative, but also consistent with the individual features of the replier.

## [Adaptive Parameterization for Neural Dialogue Generation](https://aclanthology.org/D19-1188.pdf)

### Task

Dialogue Response Generation

### Data

Ubuntu+Reddit+a-chit-chat-dataset -> 87,468 context-response pairs

### Approach

Conversation-specific parameterization: For each conversation, the model generates parameters of the encoder-decoder by referring to the input context:
- context-aware
- topic-aware

### Evaluation

Semantic Relevance between ground truth and generation: BLEU, embedding average, embedding extrema, embedding greedy
Informativeness and Diversity: Distinct-1,2,3 grams
Human Evaluation: measure context relevance, logical consistency, fluency and informativeness

## [Build it Break it Fix it for Dialogue Safety: Robustness from Adversarial Human Attack](https://aclanthology.org/D19-1461.pdf)

### Task

Improve robustness of dialoge models against Offensive Langauage

### Data

Wiki Toxic Comments (WTC) dataset
Workers are shown truncated pieces of a conversation from the ConvAI2 chit-chat task, 
-> asked to continue the conversation with OFFENSIVE responses that our classifier marks as SAFE
-> 3000 examples

### Approach

Train a BERT based offensive message detection, ask crowdsourced-workers to break it, use those examples to retrain the model

### Evaluation

F1, Weighted-F1

## [Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog](https://arxiv.org/pdf/1908.10719.pdf)

### Task

To estimate the reward signal and infer the user goal in the dialog sessions.

### Data

MultiWOZ

### Approach

A novel algorithm based on Adversarial Inverse Reinforcement Learning for joint reward estimation and policy optimization in multi-domain task-oriented dialog.
The reward estimator evaluates state-action pairs so that it can guide the dialog policy at each dialog turn.

### Evaluation

Inform F1 : Evaluates whether all the requested information (e.g. address, phone number of a hotel) has been informed.
Match rate : This evaluates whether the booked entities match all the indicated constraints (e.g.Japanese food in the center of the city) for all domains.


## [DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation](https://aclanthology.org/D19-1015.pdf)

### Task

Emotion recognition in conversation.

### Data

IEMOCAP, AVEC, MELD (multimodal datasets containing textual, visual and acoustic information for every utterance of each conversation)

### Approach

DialogueGCN:
- Sequential Context Encoder (BiGRU)
- Speaker-Level Context Encoder (Graphical Network)
- Emotion Classifier (Concatenaion of the above 2 -> ReLU -> softmax -> argmax)

### Evaluation

F1, Weighted-F1
