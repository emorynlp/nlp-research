# ACL 2021: Dialog and Interactive Systems

https://acl2021.org/schedule/

## [Conversations Are Not Flat: Modeling the Dynamic Information Flow across Dialogue Utterances](https://aclanthology.org/2021.acl-long.11/)

### Task

Open-domain response generation

### Data

Reddit, Daily Dialog

### Approach

Transformer-based seq2seq model that generates a response based on the previous utterance and the predicted semantic influence, which is based on iterative dialogue context representations.

Present automatic metric Flow score, correlated r=0.90 with human ratings

### Evaluation

#### Automated

NIST-2/4, BLEU-2/4, MEteor, Entropy

#### Human

Static preference evaluation for relevance, informativeness, and human-likeness

## [Transferable Dialogue Systems and User Simulators](https://aclanthology.org/2021.acl-long.13/)

### Task

task-oriented dialogue

### Data

MultiWOZ 2.0

### Approach

Joint-learning of end-to-end neural task-oriented dialogue model (composed of dialogue state tracker, database querying, context encoder, policy learning, NLG model) and user simulator, followed by finetuning with reinforcement learning on their generated dialogues

### Evaluation

#### Automated

Inform rate, success rate, BLEU

## [BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data](https://aclanthology.org/2021.acl-long.14/)

### Task

Persona-grounded response generation

### Data

PersonaChat, PersonalDialog

MNLI, CMNLI, DNLI, KvPI

### Approach

BERT-based encoder-autoregressive decoder model to generate a response given a (persona, query) pair, followed by feeding the generated response representation and the persona into a different bidirectional decoder trained on dialogue natural language inference tasks to generate the final form of the response

### Evaluation

#### Automated

Perplexity, Distinct-1/2, C Score, Delta Perplexity

#### Human

Static numeric evaluation of fluency, informativeness, and relevance, persona consistency

## [TicketTalk: Toward human-level performance with end-to-end, transaction-based dialog systems](https://aclanthology.org/2021.acl-long.55/)

### Task

task-oriented dialogue

### Data

newly created TicketTalk

### Approach

Utilized a text-to-text model (T5) for response generation by designing a text-format for verbal utterances, API calls, and API results that is suitable for the task-oriented dialogue setting

### Evaluation

#### Automated

BLEU

#### Human

Static binary evaluation on sensibility of response, appropriateness of API calls, and completeness of API calls

## [Improving Dialog Systems for Negotiation with Personality Modeling](https://aclanthology.org/2021.acl-long.56/)

### Task

Negotiation task-oriented dialogues

### Data

CraigslistBargain

### Approach

Multi-agent Markov Decision Process with user reaction lookahead and explicit user personality estimation, trained first under supervised learning and then finetuned using reinforcement learning using actor-critic methods

### Evaluation

#### Automated

Agreement rate, Objective utility, Deal fairness, Dialog length

## [Learning from Perturbations: Diverse and Informative Dialogue Generation with Inverse Adversarial Training](https://aclanthology.org/2021.acl-long.57/)

### Task

Open-domain response generation

### Data

OpenSubtitles, Daily Dialog

### Approach

seq2seq models with adversarial training where the model is rewarded for changing the generated response given a dialogue history perturbation and punished otherwise

### Evaluation

#### Automated

Perplexity, Distinct-1/2/3, stopword percentage

#### Human

Interactive numeric evaluation of fluency, consistency, and diversity

## [Increasing Faithfulness in Knowledge-Grounded Dialogue with Controllable Features](https://aclanthology.org/2021.acl-long.58/)

### Task

Knowledge-grounded response generation

### Data

WoW

### Approach

seq2seq response generation model that incorporates control codes to allow for controlled variations in responses along the dimensions of objectivity and truthfulness.

### Evaluation

#### Automated

BLEU, Lexical precision/recall, Objectivity, Entailment ratio 

#### Human

Static numeric evaluation of fluency, relevance, supported/faithfulness, and objectivity

## [Saying No is An Art: Contextualized Fallback Responses for Unanswerable Dialogue Queries](https://aclanthology.org/2021.acl-short.13/)

### Task

Open-domain response generation

### Data

newly created "I Dont Know Dataset"

### Approach

pretrained seq2seq model finetuned to produce less generic fallback responses (e.g. "i dont know") by training on a dataset of (query, don't-know-response) pairs that is generated using handcrafted dependency templates and paraphrasing of the queries

### Evaluation

#### Automated

Coverage, Average Sentene Length, Sentence Length Variance, Average # Novel Words

#### Human

Static numeric evaluation of grammaticality and relevance

## [N-Best ASR Transformer: Enhancing SLU Performance using Multiple ASR Hypotheses](https://aclanthology.org/2021.acl-short.14/)

### Task

Task-oriented NLU (semantic parsing into act-slot-value triplets)

### Data

DSTC2

### Approach

pretrained transformer encoder model with a semantic tuple classifier finetuned on data of the form (concatenated N-best ASR results, act-slot-value triplets) 

### Evaluation

#### Automated

F1-score, Accuracy

## [I like fish, especially dolphins: Addressing Contradictions in Dialogue Modeling](https://aclanthology.org/2021.acl-long.134/)

### Task

Dialogue contradiction detection

### Data

newly created dataset of human-human dialogues and human-bot dialogues containing contradictions

SNLI, MNLI, ANLI-R3, DNLI

### Approach

Transformer-based neural classification model where contradiction is taken as the max over all predictions on pairs of utterances from the same speaker

### Evaluation

#### Automated

Accuracy, F1, Exact Match

## [Discovering Dialog Structure Graph for Coherent Dialog Generation](https://aclanthology.org/2021.acl-long.136/)

### Task

Open-domain response generation

### Data

Weibo, Douban

### Approach

Unsupervised dialog structure graph induction using a discrete variational autoencoder with a graph neural network, which is then used as external knowledge in a reinforcement learning generative dialogue model

### Evaluation

#### Automated

Evaluation of induced graph reconstructions:

NLL, BLEU-1/2

Evaluation of dialogs:

dialog length, Distinct-1/2

#### Human

Evaluation of induced graph:

Ratings of Sess-Utter Appropriateness, Utter-Utter Appropriateness, Session-level Vertex Quality 

Static evaluation of human-bot dialogs:

Multi-turn coherence, Single-turn coherence, engagement

## [Dialogue Response Selection with Hierarchical Curriculum Learning](https://aclanthology.org/2021.acl-long.137/)

### Task

Response matching (e.g. identify correct response from set of candidates)

### Data

Douban, Ubuntu, E-Commerce

### Approach

Hierarchical curriculum learning paradigm where the dual-encoder matching model is trained with positive examples increasing from easiest to hardest and the same for negative candidates 

### Evaluation

#### Automated

Douban: MAP, MRR, P@1, R\_10@1, R\_10@2, R\_10@5

Ubuntu: R\_2@1, R\_10@1, r\_10@2, R\_10@5

E-Commerce: R\_10@1, R\_10@2, R\_10@5

## [PRAL: A Tailored Pre-Training Model for Task-Oriented Dialog Generation](https://aclanthology.org/2021.acl-short.40/)

### Task

Task-oriented dialog

### Data

newly created PretrainDial 

CamRest676, MultiWOZ, PersuasionForGood

### Approach

Improved pretraining strategy for ARDM-based (separate language model for user and for system) dialog generation models by using start position randomization, Teacher model, and history discount

### Evaluation

#### Automated

CamRest676: BLEU-4, Success F1

MultiWOZ: BLEU-4, Inform, Success

PersuasionForGood: Perplexity, BLEU-1/2

#### Human

PersuasionForGood: Interactive numeric evaluation of fluency, logic, coherence, diversity, overall, avg donation

## [Robustness Testing of Language Understanding in Task-Oriented Dialog](https://aclanthology.org/2021.acl-long.192/)

### Task

Analysis of task-oriented dialog models

### Data

Frames, MultiWOZ

### Approach

Model-agnostic toolkit LAUG for testing robustness of task-oriented dialog models across various forms of input perturbation, including word perturbation, text paraphrasing, simulated speech recognition, and inserted speech disfluencies

### Evaluation

#### Automated

Perturbed inputs: Change rate of characters, words, and slot values

Robustness measures: F1

#### Human

Perturbed inputs: Utterance appropriateness, dialog act appropriateness

## [OTTers: One-turn Topic Transitions for Open-Domain Dialogue](https://aclanthology.org/2021.acl-long.194/)

### Task

Topic Transition Generation

### Data

newly created OTTers of one-turn topic transitions with bridging utterance and entity path

### Approach

Finetune transformer-based language generation model with multi-hop reasoning on OTTers

### Evaluation

#### Automated

hits@k, ROUGE-L, METEOR, CIDEr

## [Towards Emotional Support Dialog Systems](https://aclanthology.org/2021.acl-long.269/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [GTM: A Generative Triple-wise Model for Conversational Question Generation](https://aclanthology.org/2021.acl-long.271/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Diversifying Dialog Generation via Adaptive Label Smoothing](https://aclanthology.org/2021.acl-long.272.pdf)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Continual Learning for Task-oriented Dialogue System with Iterative Network Pruning, Expanding and Masking](https://aclanthology.org/2021.acl-short.66/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [HERALD: An Annotation Efficient Method to Detect User Disengagement in Social Conversations](https://aclanthology.org/2021.acl-long.283/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding](https://aclanthology.org/2021.acl-long.285/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [NeuralWOZ: Learning to Collect Task-Oriented Dialogue via Model-Based Simulation](https://aclanthology.org/2021.acl-long.287/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Unsupervised Enrichment of Persona-grounded Dialog with Background Stories](https://aclanthology.org/2021.acl-short.74/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Neural Stylistic Response Generation with Disentangled Latent Variables](https://aclanthology.org/2021.acl-long.339/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [RADDLE: An Evaluation Benchmark and Analysis Platform for Robust Task-oriented Dialog Systems](https://aclanthology.org/2021.acl-long.341/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human


## [Domain-Adaptive Pretraining Methods for Dialogue Understanding](https://aclanthology.org/2021.acl-short.84/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Semantic Representation for Dialogue Modeling](https://aclanthology.org/2021.acl-long.342/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [A Pre-training Strategy for Zero-Resource Response Selection in Knowledge-Grounded Conversations](https://aclanthology.org/2021.acl-long.343/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [SOLOIST: Building Task Bots at Scale with Transfer Learning and Machine Teaching](https://arxiv.org/abs/2005.05298)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Maria: A Visual Experience Powered Conversational Agent](https://aclanthology.org/2021.acl-long.435/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [A Human-machine Collaborative Framework for Evaluating Malevolence in Dialogues](https://aclanthology.org/2021.acl-long.436/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Generating Relevant and Coherent Dialogue Responses using Self-Separated Conditional Variational AutoEncoders](https://aclanthology.org/2021.acl-long.437/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Learning to Ask Conversational Questions by Optimizing Levenshtein Distance](https://aclanthology.org/2021.acl-long.438/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [DVD: A Diagnostic Dataset for Multi-step Reasoning in Video Grounded Dialogue](https://aclanthology.org/2021.acl-long.439/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [On the Generation of Medical Dialogs for COVID-19](https://aclanthology.org/2021.acl-short.112/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Constructing Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant Images](https://aclanthology.org/2021.acl-short.113/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation](https://aclanthology.org/2021.acl-long.440/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [DynaEval: Unifying Turn and Dialogue Level Evaluation](https://aclanthology.org/2021.acl-long.441/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Unsupervised Learning of KB Queries in Task-Oriented Dialogs](https://arxiv.org/abs/2005.00123)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Pretraining the Noisy Channel Model for Task-Oriented Dialogue](https://arxiv.org/abs/2103.10518)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [The R-U-A-Robot Dataset: Helping Avoid Chatbot Deception by Detecting User Questions About Human or Non-Human Identity](https://aclanthology.org/2021.acl-long.544/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Conversation Graph: Data Augmentation, Training, and Evaluation for Non-Deterministic Dialogue Management](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00352/97777/Conversation-Graph-Data-Augmentation-Training-and)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [Space Efficient Context Encoding for Non-Task-Oriented Dialogue Generation with Graph Attention Transformer](https://aclanthology.org/2021.acl-long.546/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human

## [DialogueCRN: Contextual Reasoning Networks for Emotion Recognition in Conversations](https://aclanthology.org/2021.acl-long.547/)

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

#### Automated

#### Human
