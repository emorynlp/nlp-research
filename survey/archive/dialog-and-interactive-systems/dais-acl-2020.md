# ACL 2020: Dialog and Interactive Systems

https://acl2020.org/schedule/


## [PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable](https://aclanthology.org/2020.acl-main.9.pdf)

### Task

Open-domain response generation 

### Data

Persona-Chat, Daily Dialog, DSTC7-AVSD

### Approach

Joint-learning of transformer-based response generation and latent speech act recognition with self-attention

### Evaluation

#### Automated
*Persona-Chat and Daily Dialog:* BLEU-1/2, Distinct-1/2, Knowledge Recall/Precision/F1

*DSTC7-AVSD:* BLEU-1/2/3/4, METEOR, ROUGH-L, CIDEr

#### Human

*Persona-Chat and Daily Dialog:* Static numeric evaluation on fluency, coherence, informativeness, and overall (quality)

## [Large Scale Multi-Actor Generative Dialog Modeling](https://aclanthology.org/2020.acl-main.8.pdf)

### Task

Persona-grounded response generation

### Data

Newly created Reddit dataset, with conversations attributed to the involved users

### Approach

GPT-2-based models conditioned on reference utterances for an individual persona

### Evaluation

#### Automated

Perplexity

#### Human

Static preference evaluation on realisticness, reference coherence, quality, and coherence

## [Data Manipulation: Towards Effective Instance Learning for Neural Dialogue Generation via Learning to Augment and Reweight](https://aclanthology.org/2020.acl-main.564.pdf)

### Task

Open-domain response generation

### Data

Daily Dialog and OpenSubtitles

### Approach

Automatic iterative data manipulation (diversity augmentation and reweighting) to increase training reliability for response generation models

Joint-learning for data manipulation network and dialogue model

### Evaluation

#### Automated

BLEU, Distinct-1/2/3, Embedding Similarity-Avg/Extrema/Greedy, 1/2/3-gram Entropy, Intra(distinct)-1/2/3

#### Human

Static preference evaluation on dialogue responses

## [Negative Training for Neural Dialogue Response Generation](https://aclanthology.org/2020.acl-main.185.pdf)

### Task

Open-domain response generation

### Data

Ubuntu, Switchboard, and OpenSubtitles

### Approach

seq2seq model trained with negative examples which are automatically identified as undesirable generated responses

### Evaluation

#### Automated

Perplexity, 2/3-gram Entropy, Malicious hit rate, Max ratio


## [Don’t Say That! Making Inconsistent Dialogue Unlikely with Unlikelihood Training](https://aclanthology.org/2020.acl-main.428.pdf)

### Task

Open-domain response generation

### Data

Reddit, ConvAI2 persona-based dialogues, WoW, ELI5

### Approach

Utilize unlikelihood loss to reduce known biases (repetition, context copying, skewed vocabulary, contradiction) in transformer-based response generation models

### Evaluation

#### Automated

Perplexity, F1, context/label repetition, token frequency distributions, non-contradiction selection accuracy

#### Human

Static preference evaluations

## [Generating Informative Conversational Response using Recurrent Knowledge-Interaction and Knowledge-Copy](https://aclanthology.org/2020.acl-main.6/)

### Task

Knowledge-grounded response generation

### Data

WoW and DuConv

### Approach

seq2seq model using dynamic knowledge attention and a knowledge-aware & utterance-aware pointer network 

### Evaluation

#### Automated

BLEU-1/2/3, F1, Distinct-1/2

#### Human

Static numeric evaluation on fluency, informativeness, and coherence

## [You Impress Me: Dialogue Generation via Mutual Persona Perception](https://aclanthology.org/2020.acl-main.131/)

### Task

Persona-grounded response generation

### Data

Persona-Chat

### Approach

Transmitter-receiver architecture to model mutual persona perception (receiver) from conversation generations (transmitter), where self-play reinforcement-learning encourages transmitter to maximize both language modelling and mutual persona perception objectives

### Evaluation

#### Automated

Hits@1, Perplexit, F1, BLEU

#### Human

Static rating evaluation on responses


## [CDL: Curriculum Dual Learning for Emotion-Controllable Response Generation](https://aclanthology.org/2020.acl-main.52.pdf)

### Task

Emotion-controllable response generation

### Data

NLPCC2017 (NLPCC 2017 Emotional Conversation Generation Challenge)

### Approach

Curriculum dual learning of response and query generation via reinforcement learning with rewards that encourage emotion expression and content consistency

### Evaluation

#### Automated

Embedding Similarity-Avg/Extrema/Greedy/Coherence, BLEU, Distinct-1/2, Emotion-acc, Emotion-word

#### Human

Static numeric evaluation on content and emotion


## [Can You Put it All Together: Evaluating Conversational Agents’ Ability to Blend Skills](https://aclanthology.org/2020.acl-main.183.pdf)

### Task

Open-domain response retrieval

### Data

new BlendedSkillTalk

ConvAI2 (extension of Persona-Chat), Wow, Empathetic Dialogues

### Approach

transformer-based poly-encoder

various approaches to combining conversational skills:

* train on dataset that contains all skills
* multi-task training on datasets for individual skills
* individual models per skill moderated by a skill-selection classifier

### Evaluation

#### Automated

Hits@1

#### Human

Interactive numeric evaluation for knowledgeable, empathetic, personal, and overall (quality)


## [Generate, Delete and Rewrite: A Three-Stage Framework for Improving Persona Consistency of Dialogue Generation](https://aclanthology.org/2020.acl-main.516.pdf)

### Task

Persona-grounded response generation

### Data

Persona-Chat

### Approach

3-stage pipeline that produces a response, identifies inconsistent words in response vs the persona via a trained Dialogue Natural Language Inference model, and replaces those inconsistent words; all stages employ transformer-based models with a combination of persona-/query-/response-attention depending on the stage

### Evaluation

#### Automated

Perplexity, Distinct-1/2, Entailment ratios

#### Human

Static numeric evaluation for fluency, relevance, informativeness, and persona consistency

Static preference evaluation for overall response quality

## [Learning to Customize Model Structures for Few-shot Dialogue Generation Tasks](https://aclanthology.org/2020.acl-main.517.pdf)

### Task

Few-shot open-domain response generation

### Data

Persona-Chat and MojiTalk

### Approach

Model-agnostic framework for dialogue model decomposition into shared, private, and gate modules, where the private modules learn distinct parameters and structures through finetuning and pruning for different dialogue tasks and the gate dictates what information to fuse between the shared and private module outputs 

### Evaluation

#### Automated

BLEU, Perplexity, Distinct-1, C-score (for Persona-Chat), E-acc (for MojiTalk)

#### Human

Static numeric evaluation for quality and task consistency

## [Diversifying Dialogue Generation with Non-Conversational Text](https://aclanthology.org/2020.acl-main.634/)

### Task

Open-domain response generation

### Data

newly created non-conversational utterance dataset (sources: zhihu, idiom/quote websites, wechat read)

Weibo, Douban

### Approach

seq2seq back translation on gold (context,response) parallel data and silver (context, nonconversational utterance) parallel data, where the silver data is generated from the backward seq2seq model after gold training

### Evaluation

#### Automated

BLEU-2, Distinct-1/2, 4-gram Entropy, Adversarial discriminator (Adver)

#### Human

Static numeric evaluations for relevance, interestingness, and fluency

## [Guiding Variational Response Generator to Exploit Persona](https://aclanthology.org/2020.acl-main.7/)

### Task

Persona-grounded response generation

### Data

Douban, Cornell Movie Dialogues

### Approach

Variational latent space encoder-decoder using the query, response, and user embeddings, with two new regularization terms that encourage incorporation of the user embedding into the latent variable distributions 

### Evaluation

#### Automated

BLEU-1, Embedding Similarity-Svg/Extrema/Greedy, uRank, uPPL, uDistinct

#### Human

Static numeric evaluation for relevance and persona portrayal

## [Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness](https://aclanthology.org/2020.acl-main.515/)

### Task

Commonsense-grounded response generation

### Data

Weibo, Reddit

### Approach

seq2seq with weighted fact fusion and a 3-source decoder (vocabulary words, fact words, and context words)

### Evaluation

#### Automated

E_match, E_use, E_recall, Embedding Similarity-Avg/Extrema, BLEU-2/3, Distinct-1/2, character-level entropy

#### Human

Static preference evaluation of appropriateness and informativeness

## [Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation](https://aclanthology.org/2020.acl-main.166/)

### Task

Open-domain dialogue policy learning 

### Data

Weibo, Persona-Chat

### Approach

Given a Conversational Graph built from a large dialogue corpus where vertices are keywords and edges are dialog transitions, a dialog policy (user message -> (keyword, responding mechanism)) is learned via reinforcement learning

### Evaluation

#### Automated

Simulated user conversations used for evaluation

Distinct-2, dialog-target success rate

#### Human

Human-machine dialogues that are statically evaluated

Global coherence, appropriateness, informativeness

## [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs](https://aclanthology.org/2020.acl-main.184/)

### Task

Commonsense-grounded response generation

### Data

Reddit, ConceptNet

### Approach

Encoder-decoder response generation using graph-embeddings and graph-attention on ConceptNet subgraphs related to entites in utterance

### Evaluation

#### Automated

Perplexity, BLEU, Nist, ROUGE, MEteor, Distinct-1/2, 4-gram Entropy, Entity precision/recall/F1

#### Human

Static numeric evaluation for appropriateness and informativeness

