# NAACL 2021: Dialog and Interactive Systems

https://2021.naacl.org/conference-program/main/program.html


## [Human-like informative conversations: Better acknowledgements using conditional mutual information](https://aclanthology.org/2021.naacl-main.61.pdf)

### Task

Dialogue response generation.

### Data

Topical Chat.

### Approach

Improve acknowledgements by selecting candidate response from finetuned GPT3 using PCMI between (1) candidate response and context conditioned on knowledge source, and (2) candidate response and knowledge source conditioned on context.

### Evaluation

Human judgement of response quality and attribution of quality improvement to acknowledgement.


## [A Comparative Study on Schema-Guided Dialogue State Tracking](https://aclanthology.org/2021.naacl-main.62.pdf)

### Task

Dialogue state tracking.

### Data

SG-DST, MultiWoZ 2.2.

### Approach

Bert encoding (1) dialogue token sequence and (2) natural language schemas defining intents and slots.

### Evaluation

Intent accuracy, slot F1. Evaluated different flavors of encoding schema (name only, short description, long description, etc.).


## [Spoken Language Understanding for Task-oriented Dialogue Systems with Augmented Memory Networks](https://aclanthology.org/2021.naacl-main.63.pdf)

### Task

Intent detection and slot filling.

### Data

ATIS and Snips.

### Approach

BiLSTM with self attention encoder, and Key Value Memory Network (KV-MN) to store slot-value information. KV-MN adds, removes, and retrieves information from a value matrix using attention, similar to how LSTM adds and removes information from cell state. 

### Evaluation

Intent accuracy, slot F1, full sentence accuracy.


## [How to Motivate Your Dragon: Teaching Goal-Driven Agents to Speak and Act in Fantasy Worlds](https://aclanthology.org/2021.naacl-main.64.pdf)

### Task

Play the game LIGHT by taking actions given game states and reward signals from the game engine. During play, the agent can communicate via natual language dialogue with a partner.

### Data

LIGHT text-based fantasy game, ATOMIC, Reddit.

### Approach

Value-based deep reinforcement learning (specifically, A3C) training a transfomer encoder. Encoding of the history of actions, environment descriptions, and dialogue turns is encoded by the transformer as a single sequence, then GRU layers are used to decode responses and actions. Rewards are given by a poly-encoder model trained on human demonstrations of playing the game, as well as direct rewards from the game engine. The conversation partner is a poly-encoder model pre trained on Reddit and fine tuned on human demonstrations. The partner and reward models do not update parameters during training.

### Evaluation

Goal completion rate in-game. Additionally, rate of outputting speech in either of (1) the top-k candidates of the reward model or (2) a response contained within the demonstration data.


## [Fine-grained Post-training for Improving Retrieval-based Dialogue Systems]()

### Task

Dialogue response retrieval/selection.

### Data

Ubuntu IRC Corpus V1, E-commerce Corpus, Douban Corpus.

### Approach

BERT encodes context and response candidate, and one final hidden vector used to classify candidate. After fine-tuning BERT in this way, a second post-training session was used that trains model to discriminate between correct and random response candidates using a shorter context window.

### Evaluation

Retrieval @K, mean average precision (MAP), mean reciprocal rank (MRR).


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


