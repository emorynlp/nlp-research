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

Emotional Support Conversations

### Data

newly created ESConv

### Approach

Propose a new task of Emotional Support conversations with a framework for dialog systems based on cognitive science theory of conversational support (Hill's Helping Skills theory) that is composed of 3 stages (exploration, comforting, and action) with their respective relevant conversational strategies

Crowdsource dataset for this task with extensive participant training and quality control measures

Finetune pretrained seq2seq models on this dataset with conversational strategy annotation as additional input

### Evaluation

#### Automated

perplexity, BLEU-2, ROUGE-L, BOW Embedding Similarity Extrema

#### Human

Interactive preference evaluation of fluency, identification, comforting, suggestion, overall quality

## [GTM: A Generative Triple-wise Model for Conversational Question Generation](https://aclanthology.org/2021.acl-long.271/)

### Task

Open-domain question generation

### Data

corpus extracted from Reddit

### Approach

Utilize variational latent variables to capture post-query-answer, post-query, and query-answer relationships

### Evaluation

#### Automated

BLEU-1/2, Distinct-1/2, Embedding Similarity-Avg/Extrema/Greedy, RubG, RubA

#### Human

Static numeric evaluation of fluency, coherence, willingness

## [Diversifying Dialog Generation via Adaptive Label Smoothing](https://aclanthology.org/2021.acl-long.272.pdf)

### Task

Open-domain response generation

### Data

Daily Dialog, OpenSubtitles

### Approach

Dynamic adjustments to the supervision signals provided to a seq2seq model at training time to reduce over-fitting/high-frequency word problem by learning to reduce reward of correctly predicted target words and increasing reward of non-target words using an auxiliary decoder

### Evaluation

#### Automated

Distinct-1/2, 1/2-gram Entropy, BLEU-2/3/4, Low-frequency token ratio

#### Human

Static preference evaluations of fluency, coherence, informativeness, and overall quality

## [Continual Learning for Task-oriented Dialogue System with Iterative Network Pruning, Expanding and Masking](https://aclanthology.org/2021.acl-short.66/)

### Task

Task-oriented dialogue

### Data

In-Car Assistant, Multi-WOZ 2.1, CamRest

### Approach

Framework for learning new tasks, given a model trained on previous tasks, that utilizes iterative pruning of old task weights, network expansion to create new weights, and task-specific weight masks

### Evaluation

#### Automated

BLEU, Entity F1

## [HERALD: An Annotation Efficient Method to Detect User Disengagement in Social Conversations](https://aclanthology.org/2021.acl-long.283/)

### Task

Open-domain dialogue user engagement detection

### Data

Gunrock Movie Dataset, ConvAI2, Google Meena, Facebook Blender

### Approach

Finetune a pretrained BERT-based classifier on data that is automatically labelled with heuristics and denoised using Shapely

### Evaluation

#### Automated

balanced accuracy, F_2 Score

## [MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding](https://aclanthology.org/2021.acl-long.285/)

### Task

Multi-party conversation (MPC)

### Data

2 Ubuntu IRC datasets (Hu et al. 2019; Ouchi and Tsuboi 2016)

### Approach

New pre-training paradigm for language models of multi-party conversations by joint training on 5 self-supervised tasks in a multi-task learning setup: reply-to utterance recognition, identical speaker searching, pointer consistency distinction, masked shared utterance restoration, and shared node detection

### Evaluation

#### Automated

Use pretraining approach on BERT-based language model and then evaluate how well it performs after finetuning on each task below:

Addressee Recognition: P@1, Accuracy

Speaker Identification: P@1

Response Selection: R_2@1, R_10@1


## [NeuralWOZ: Learning to Collect Task-Oriented Dialogue via Model-Based Simulation](https://aclanthology.org/2021.acl-long.287/)

### Task

Task-oriented dialogue collection

### Data

MultiWOZ 2.1

### Approach

Transformer-based seq2seq model to generate a synthetic dialogue between the user and the system based on provided goal instructions and API call results where each utterance is labelled with its realized slot-value pairs from a transformer-based multiple-choice selection model

### Evaluation

#### Automated

Zero-shot learning: Joint goal accuracy, slot accuracy

Intrinsic evaluation on model: Perplexity, Joint goal accuracy

#### Human

## [Unsupervised Enrichment of Persona-grounded Dialog with Background Stories](https://aclanthology.org/2021.acl-short.74/)

### Task

Persona-grounded response generation

### Data

Persona-Chat, ROCStories

### Approach

Utilize story excerpts to generate more interesting dialogue responses by performing gradient-based decoding (modify the underlying hidden state representation of each output timestep by backpropogation) such that the generated response is fluent with the context, minimally different from the retrieved story excerpt, and is maximally consistent with the persona

### Evaluation

#### Automated

Distint-1/2, Entropy

#### Human

Static preference evaluation of sensibility and engagingness

## [Neural Stylistic Response Generation with Disentangled Latent Variables](https://aclanthology.org/2021.acl-long.339/)

### Task

Stylistic open-domain response generation

### Data

Daily Dialog, Holmes monolingual stylistic dataset

### Approach

Generative model that uses a structured latent space with a seq2seq model for dialog history, an autoencoder model for response and stylistic sentence, and a shared decoder. The model is trained such that it learns to distinguish latent content and style information by averaging the style features of the response at decoding

### Evaluation

#### Automated

Style Intensity, BLEU-3/4, Distinct-1/2

#### Human

Static preference evaluation of content relevance and style intensity

## [RADDLE: An Evaluation Benchmark and Analysis Platform for Robust Task-oriented Dialog Systems](https://aclanthology.org/2021.acl-long.341/)

### Task

Task-oriented dialog benchmarking

### Data

Built from MultiWOZ 2.0

### Approach

Presents a benchmark for evaluating task-oriented dialog models along the dimensions of generalizability to low-resource new tasks, dialog state tracking, end-to-end modeling, and robustness to varied users (language variation, speech errors, unseen entities, out-of-domain utterances) 

### Evaluation

#### Automated

Dialog State Tacking: joint goal accuracy

End-to-end modeling: Inform, Success, BLEU

#### Human

Top-ranked submissions to the benchmark have access to human evaluations (unspecified in this paper what those human evaluations consist of)

## [Domain-Adaptive Pretraining Methods for Dialogue Understanding](https://aclanthology.org/2021.acl-short.84/)

### Task

Conversational Semantic Role Labeling and Spoken Language Understanding

### Data

DuConv, NewsDialog, CrossWOZ

### Approach

Analyze the impacts of pretraining objective (MLM, Span Boundary Objective, proposed Perturbation Masking Objective) on downstream task performance

Perturbation Masking Objective maximizes the correlation between predicates and arguments by rewarding large contextualized vector distances of arguments between when the predicate is masked versus not masked in the sequence

### Evaluation

#### Automated

DuConv, NewsDialog: F1_all/cross/intra

CrossWOZ: F1_intent/slot/all

## [Semantic Representation for Dialogue Modeling](https://aclanthology.org/2021.acl-long.342/)

### Task

Open-domain response generation and dialogue relation extraction

### Data

DialogRE, Daily Dialog

### Approach

Incorporate dialogue-level AMR representations into Transformer-based models for the tasks by: 

(1) encoding the AMR-relations between Transformer hidden-state token representations using a Graph-Transformer 

or 

(2) separately encoding AMR graph and sentence, then combine their embeddings using either feature fusion (for dialogue relation extraction) or dual attention (for dialogue response generation)

### Evaluation

#### Automated

relation extraction: F1

response generation: BLEU-1/2/3/4, Distinct-1/2

#### Human

response generation: Static numeric evaluation of coherence, fluency, informativeness, and overall quality

## [A Pre-training Strategy for Zero-Resource Response Selection in Knowledge-Grounded Conversations](https://aclanthology.org/2021.acl-long.343/)

### Task

Knowledge-grounded response matching

### Data

Train: MS MARCO, Reddit 

Test: WoW, CMU_DoG

### Approach

Pretraining paradigm of joint-learning on query-passage matching, query-dialogue history matching, and multi-turn response matching tasks in order to leverage ad-hoc retrieval datasets in addition to knowledge-grounded dialogue datasets

### Evaluation

#### Automated

R_n@k

## [SOLOIST: Building Task Bots at Scale with Transfer Learning and Machine Teaching](https://arxiv.org/abs/2005.05298)

### Task

Task-oriented dialog

### Data

CamRest676, MultiWOZ

### Approach

seq2seq model that takes as input a concatenated text sequence of dialog history and database query results and learns to output the response by jointly learning belief  state prediction and response generation

### Evaluation

#### Automated

Inform, Success, BLEU

#### Human

Interactive numeric evaluation of success, understanding, and appropriateness

## [Maria: A Visual Experience Powered Conversational Agent](https://aclanthology.org/2021.acl-long.435/)

### Task

Open-domain response generation

### Data

Reddit, Open Images, MS-COCO

### Approach

Ground response generation on visual images by retrieving a relevant image using a cross-modal matching model trained on image captioning data, extracting concept labels and features from the image using an object detection model pretrained on Visual Genome, and producing a response conditioned on the context and extracted visual information using a transformer encoder-decoder architecture.

### Evaluation

#### Automated

perplexity, BLEU-1, ROUGE-L, Embedding Similarity-Avg/Extrema/Greedy, Distinct-1/2

#### Human

Static numeric evaluation of fluency, relevance, and richness

## [A Human-machine Collaborative Framework for Evaluating Malevolence in Dialogues](https://aclanthology.org/2021.acl-long.436/)

### Task

Dialogue evaluation - improve efficiency without sacrificing reliability

### Data

MDRDC

### Approach

Designed a sample assignment execution module that assigns a dialogue to either the automatic or human evaluation sets using integer linear programming over the model confidence estimation (MCE) module output and the human effort estimation (HEE) module output. The model confidence estimation module utilizes a BERT-based maximum class probability, trust score, and true class probability. The human effort estimation module estimates annotation time using random forest regression based on dialogue and worker features.

### Evaluation

#### Automated

Overall: precision, recall, F1-score, accuracy, human-ratio, time cost, 

MCE: AUC, top-k accuracy

HEE: MSE, RMSE, MAE, R^2

## [Generating Relevant and Coherent Dialogue Responses using Self-Separated Conditional Variational AutoEncoders](https://aclanthology.org/2021.acl-long.437/)

### Task

Open-domain response generation

### Data

Daily Dialog, OpenSubtitles

### Approach

Conditional variational auto-encoder model for response generation where dialogue contexts are partitioned into groups and inter-group similarity is decreased while intra-group similarity is increased.

### Evaluation

#### Automated

Perplexity, response length, distinct-1/2, BLEU-1, Embedding Similarity Avg, coherence

#### Human

Static numeric evaluation of diversity, relevance, fluency

## [Learning to Ask Conversational Questions by Optimizing Levenshtein Distance](https://aclanthology.org/2021.acl-long.438/)

### Task

Conversational Question Simplification

### Data

CANARD, CAsT

### Approach

Predict edits and phrasings needed to transform input question into conversation question using a Hierarchical Combinatorial Markov Decision Process using an Iterative Reinforce Training algorithm that is rewarded according to Levenshtein distance

### Evaluation

#### Automated

BLEU-1/2/3/4, ROUGE-L, CIDEr

## [DVD: A Diagnostic Dataset for Multi-step Reasoning in Video Grounded Dialogue](https://aclanthology.org/2021.acl-long.439/)

### Task

Video-grounded conversational question dataset creation

### Data

Built from CATER

### Approach

(Functional program, question template) pairs for video intervals are created and used to generate relevant questions and answers in sequential order for a video. 3 semantic relation transformations are defined and applied to the generations in order to capture conversational linguistic patterns and conversation-relevant reasoning, including temporal relations, object reference (pronouns, etc.), and topic transfer.

### Evaluation

Report results of baseline models (e.g. popularity, tf-idf retrieval, RNN, RNN+CNN, transformer-based) on a retrieval multiple-choice task

#### Automated

Accuracy

## [On the Generation of Medical Dialogs for COVID-19](https://aclanthology.org/2021.acl-short.112/)

### Task

COVID-19 doctor dialogue system

### Data

newly created CovidDialog (English and Chinese versions)

### Approach

encoder-decoder model trained with multi-task learning of response generation and masked-token prediction to regularize the model

### Evaluation

#### Automated

perplexity, NIST-4, BLEU-2/4, METEOR, 4-gram Entropy, and Distinct-1/2

#### Human

Static numeric evaluation of correctness, relevance, informativeness, and doctor-likeness

## [Constructing Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant Images](https://aclanthology.org/2021.acl-short.113/)

### Task

Text-and-image dialogue dataset creation

### Data

newly created dataset based on 3 dialogue datasets (Daily Dialog, EmpatheticDialogues, and Persona-Chat) and 2 image-captioning datasets (MS-COCO and Flicker 30k)

### Approach

Apply text-to-image replacement using pre-trained Visual Semantic Reasoning Network on dialogue utterances to substitute in a semantically-coherent image and filter out these modified dialogues if their similarity score is below a specified empirically-derived threshold from human-evaluation studies

### Evaluation

#### Automated

Evaluate utility of dataset in dialogue sentence prediction tasks: R@1, R@5, Mean Rank

#### Human

Dataset quality: numeric ratings of the swapped images along 4 dimensions (key object portrayal, meaning, context consistency)

Evaluate utility of dataset in dialogue sentence prediction tasks: numeric rating of relevance


## [MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation](https://aclanthology.org/2021.acl-long.440/)

### Task

Emotion classification in dialog

### Data

IEMOCAP, MELD

### Approach

For each modality (textual, acoustic, visual), encode the utterance into a context-aware representation using a BLSTM, FC, and FC network, respectively. Then, construct a graph where each node is an encoded utterance representation per modality and weighted edges exist between all nodes of the same modality and all nodes of the same utterance. A spectral domain deep GCN operates over the graph to produce a graph embedding for each node, which are then used as input to a MLP emotion classifier. 

### Evaluation

#### Automated

F1-score

## [DynaEval: Unifying Turn and Dialogue Level Evaluation](https://aclanthology.org/2021.acl-long.441/)

### Task

Automated dialogue evaluation

### Data

Empathetic Dialogue, ConvAI2 PersonaChat, Daily Dialog, FED (for dialogue evaluation task)

### Approach

attention-based two-stage GCN over a graph constructed with nodes as utterances and weighted edges between window-adjacent utterances paired with a fully-connected layer to output a final dialogue coherence score, trained using preference learning across positive and negative samples

### Evaluation

#### Automated

Dialogue-level discrimination task: Accuracy

Dialogue evaluation task: spearman correlation between predicted score and human score

## [Unsupervised Learning of KB Queries in Task-Oriented Dialogs](https://arxiv.org/abs/2005.00123)

### Task

Task-oriented dialogue

### Data

CamRest, DSTC2

### Approach

Remove the need for KB query annotations in the training data by designing a model that predicts when to generate a KB query (binary classifier position predictor) and what KB query to generate (encoder-decoder query predictor) via reinforcement learning (specifically, a modified version of MAPO)

### Evaluation

#### Automated

KB query predictor: accuracy, total reward, PIQ ratio

KB query position predictor: accuracy, turn difference

Overall system: BLEU, entity F1, entity F1 KB

#### Human

Static numeric evaluation of informativeness, grammar

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
