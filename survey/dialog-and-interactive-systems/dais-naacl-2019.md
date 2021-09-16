# NAACL 2019: Dialog and Interactive Systems

https://naacl2019.org/schedule/


## [Joint Multiple Intent Detection and Slot Labeling for Goal-Oriented Dialog](https://aclanthology.org/N19-1055.pdf)

### Task

Intent detection and slot filling (in task-oriented dialogue setting).

### Data

Airline Travel Information System (ATIS), Snips, and an unspecified "internal dataset" claimed to have more (52%) multi-intent sentences).

### Approach

Sequence to sequence tagging model using LSTM for encoding and decoding.

### Evaluation

F1 (slot detection), Acc (sentence-level intent classification), F1 (token-level intent classification, relevant for multi-intent sentences).


## [Improving Dialogue State Tracking by Discerning the Relevant Context](https://aclanthology.org/N19-1057.pdf)

### Task

Dialogue State Tracking.

### Data

WoZ 2.0, MultiWoZ 2.0.

### Approach

Binary classification of each token as a value to fill a candidate slot. Each input (sentence) is encoded using LSTM with self-attention, along with the previous sentence in the history for which the candidate slot was updated. Encoded sentence vectors and an attention encoding over previous system actions are used to score the candidate slot-value pair.

### Evaluation

Joint goal accuracy (% of turns where users' informed goals are detected correctly), turn request accuracy (% of turns where user's requests are detected correctly).


## [CLEVR-Dialog](https://aclanthology.org/N19-1058.pdf)

### Task

Data paper for visual dialogue ("dialogue" held about details in an image; system must answer a series of questions about the image).

### Data

CLEVR-Dialog

### Approach

Based on human-annotated scene graph of an image, generate dialogue examples using a grammar (template-based sentence generation on the bottom level of the grammar).

### Evaluation

Multi-class accuracy (system responses are generally expected to be one-word answers to specific questions).


## [Box of Lies: Multimodal Deception Detection in Dialogues](https://aclanthology.org/N19-1175.pdf)

### Task

Detect truthful and deceptive statements given multimodal conversation.

### Data

New data: transcribed and annotated Jimmy Fallons' box of lies videos, including turn segmentation and labeling for various verbal and nonverbal actions.

### Approach

Random Forest classification with various features from labeling.

### Evaluation

Accuracy, F1, precision, recall of both lie and truth detection.


## [Unsupervised Dialog Structure Learning](https://aclanthology.org/N19-1178/)

### Task

Unsupervised induction of a probabilistic latent state transition table modeling dialogue structure from conversation data.

### Data

CamRest676 (task oriented restaurant recommendation dataset). Also this weather conversation dataset https://arxiv.org/abs/1805.04803.

### Approach

VAE setup with LSTM encoder and decoders, learning latent space representing discrete dialogue states. Architecture explicitly models prior distribution of dialogue state given previous dialogue state but without token information to produce transition table.

### Evaluation

Likelihood of test set reconstruction. Also ad hoc human checking of transition structure.


## [Linguistically-Informed Specificity and Semantic Plausibility for Dialogue Generation](https://aclanthology.org/N19-1349.pdf)

### Task

Response generation.

### Data

OpenSubtitles, PersonaChat.

### Approach

Sequence to sequence model with decoder conditioned on the specificity of the target response, as defined by some specificity metrics such as normalized inverse word frequency. During test time, the target specificity is set high to encourage the model to generate more interesting responses. During generative sampling, a reranking model is used that is trained to discriminate against nonsensical language.

### Evaluation

BLEU, ROUGE, distinct-1, distinct-2 (n-gram based diversity metrics), cosine similarity to gold, as well as human scores for informativeness, topicalness, and plausibility.


## [Dialogue Act Classification with Context-Aware Self-Attention](https://aclanthology.org/N19-1373.pdf)

### Task

Dialogue act classification.

### Data

Switchboard Dialogue Act Corpus.

### Approach

BiRNN with "context-aware" self-attention (incorporates encoding of previous utterance) to encode utterances, another BiRNN to contextually encode utterance vector sequence, then CRF to output class per utterance.

### Evaluation

Accuracy.


## [Affect-Driven Dialog Generation](https://aclanthology.org/N19-1374.pdf)

### Task

Response generation.

### Data

Cornell Corpus (movie dialogues), OpenSubtitles 2018.

### Approach

Simple vocabulary-based emotion classifier and sequence-to-sequence response generator used in pipeline. Response generator uses RNN encoder-decorder setup with generative sampling reranking, trained with objective penalizing emotionally neutral and generic responses.

### Evaluation

BLEU, distinct-1, distinct-2. Binary human judgements of grammaticallity and quality.


## [Multi-Level Memory for Task Oriented Dialogs](https://aclanthology.org/N19-1375.pdf)

### Task

Task oriented dialogue.

### Data

CamRest, InCar assistant, Maluuba Frames.

### Approach

Hierarchical BiGRU encoder, copy-generative GRU decoder with copy mechanism using attention over both (1) hierarchical KB storing key-value pairs and (2) context memory of previous turns.

### Evaluation

BLEU, entity F1 (correct usages of entities in response).


## [Topic Spotting using Hierarchical Networks with Self Attention](https://aclanthology.org/N19-1376.pdf)

### Task

Topic classification.

### Data

Switchboard, extended as SWBD2.

### Approach

Hierarchical BiLSTM with self attention.

### Evaluation

Accuracy (both static and "online" (streaming) evaluation).


## [What do Entity-Centric Models Learn? Insights from Entity Linking in Multi-Party Dialogue](https://aclanthology.org/N19-1378.pdf)

### Task

Entity linking in multiparty dialogue.

### Data

Friends character identification.

### Approach

BiLSTM with "entity library" (attention over entity embeddings).

### Evaluation

Accuracy and F1.


## [Continuous Learning for Large-scale Personalized Domain Classification](https://aclanthology.org/N19-1379.pdf)

### Task

Domain classification for intelligent assistants.

### Data

Gave description only, assuming it's a dataset internal to Amazon collected via Alexa.

### Approach

LSTM encoder with dynamically-growing domain embedding table (e.g. new domain can be added such as rideshare requests).

### Evaluation

Accuracy.


## [Cross-Lingual Transfer Learning for Multilingual Task Oriented Dialog](https://aclanthology.org/N19-1380.pdf)

### Task

Intent classification and slot filling.

### Data

Collected new [multilingual dataset](https://fb.me/multilingual_task_oriented_data) in English, Spanish, and Thai for basic intelligent assistand dialogue (set reminder or alarm, check weather).

### Approach

BiLSTM with self-attention and CRF for classification. Language adaptation performed using sequence to sequence machine translation models. 

### Evaluation

Intent accuracy and slot F1, evaluated using various cross-lingual strategies, such as training only on low-resource lanaguage, translate-and-train with machine translation model, using high-resource language embeddings, etc.


## [Evaluating Coherence in Dialogue Systems using Entailment](https://aclanthology.org/N19-1381.pdf)

### Task

To be filled.

### Data

InferConvAI: extension of PersonaChat with entailment labels, MultiNLI sampled for contradictory responses.
Reddit and OpenSubtitles used to train dialogue generation models.

### Approach

Tree-LSTM and BERT each trained to score dialogue responses on entailment status (entailed, neutral, contradictory).

### Evaluation

Pearson's R against human judgements.


## [On Knowledge Distillation from Complex Networks for Response Prediction](https://aclanthology.org/N19-1382.pdf)

### Task

Dialogue response prediction, out of a large number of candidates (all turns in dialogue).

### Data

Holl-E.

### Approach

Student-teacher training of simple attention model with complex selection model like BiDAF as teacher.

### Evaluation

F1 on selecting correct response out of candidates.



