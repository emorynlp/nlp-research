# ACL 2019: Dialog and Interactive Systems

https://acl2019.org/EN/program.xhtml.html


## [Learning from Dialogue after Deployment: Feed Yourself, Chatbot!]()

### Task

In this work, we propose the self-feeding chatbot, a dialogue agent with the ability to extract new training examples from the conversations it participates in. As our agent engages in con- versation, it also estimates user satisfaction in its responses. When the conversation appears to be going well, the user’s responses become new training examples to imitate. When the agent believes it has made a mistake, it asks for feedback; learning to predict the feedback that will be given improves the chatbot’s dialogue abilities further.

### Data

PERSONACHAT chit-chat dataset with over 131k training examples. https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat

### Approach

1. Two components: interface and model
  a. interface: The interface includes "input/output processing, conversation history storage, candida preparation, and control flow"
  b. model: There is neural network for each task, share parameters are for the FEEDBACK and DIALOGUE tasks, but separate parameters for SATISFACTION task.
  
2. Training: In the initial training phase, the dialogue agent is trained on two tasks—DIALOGUE (next utterance prediction, or what should I say next?) and SATISFACTION (how satisfied is my speaking partner with my responses?)—using whatever supervised training data is available. They use Human-Human (HH) examples as initial DIALOGUE examples. The examples were generated in conversations between two humans.

3. Deployment: The agent engages in multi-turn conversations with users, extracting new deployment examples of two types. Each turn, the agent observes the context x (i.e., the conver- sation history) and uses it to predict its next utterance y^ and its partner’s satisfaction s^. If the satisfaction score is above a specified threshold t, the agent extracts a new Human-Bot (HB) DIALOGUE example using the previous context x and the human’s response y and continues the conversation. If, however, the user seems unsatisfied with its pre- vious response (sˆ < t), the agent requests feedback with a question q, and the resulting feedback response f is used to create a new example for the FEEDBACK task (what feedback am I about to receive?).

![Screen Shot 2021-09-16 at 9 28 27 AM](https://user-images.githubusercontent.com/15247433/133620817-59cc9a2f-f2ce-48d4-902e-1d1d71f3673c.png)


### Evaluation

Metric: ranking metric hits@X/Y, the fraction of time that the correct candidate was ranked in the top X out of Y available candidates; accuracy is another name for hits@1/Y.

![Screen Shot 2021-09-16 at 9 35 07 AM](https://user-images.githubusercontent.com/15247433/133621779-129d4de3-17da-4f21-b5c1-3430bd169fe7.png)

## [Generating Responses with a Specific Emotion in Dialog]()

### Task

The paper proposed an emotional dialogue system (EmoDS) that can generate the meaningful responses with a coherent structure for a post, and meanwhile express the desired emotion explicitly or implicitly within a unified framework.

### Data

Train a classifier on NLPCC and used trained classifier to label STC dataset

NLPCC: http://tcci.ccf.org.cn/conference/2019/taskdata.php

STC: http://ntcir12.noahlab.com.hk/stc.htm (not found!)

stats:

STC:

![Screen Shot 2021-09-16 at 9 58 07 AM](https://user-images.githubusercontent.com/15247433/133625509-4431418f-9714-44d1-83b3-ef35dc5dcc1d.png)

NLPCC:

![Screen Shot 2021-09-16 at 9 58 30 AM](https://user-images.githubusercontent.com/15247433/133625558-a3b1b8ca-58c5-4e56-8eec-a3286186e431.png)

### Approach

The system is based on Bi-LSTM. 

The decoder contains a lexicon-based attention mechanism that takes an emotion vector to update hidden states.

The emotion classification module is a sequence-level emotion classifier to guide the generation process, which helps to recognize the responses expressing a certain emotion but not containing any emotional word.

![Screen Shot 2021-09-16 at 9 59 56 AM](https://user-images.githubusercontent.com/15247433/133625806-6bdac013-0a26-4e05-a13f-0b2818b072a6.png)


### Evaluation

Metrics: 

1. Embedding Score: averages, greedy and extreme

2. BLEU score

3. Distinct

4. Emotion Evaluation (designed by the authors): emotion- a and emotion-w. Emotion-a is the agreement between the predicted labels through the Bi-LSTM classifier in Data Preparation and the ground truth labels. Emotion-w is the percentage of the generated responses that contain the corresponding emotional words.

[Screen Shot 2021-09-16 at 9 50 58 AM](https://user-images.githubusercontent.com/15247433/133624333-9c2b8fe9-56c9-4f64-95e3-0bdf2acf88bb.png)


## [Semantically Conditioned Dialog Response Generation via Hierarchical Disentangled Self-Attention]()

### Task

Build semantically controlled neural response generation model for multi-domain scenarios. The authors exploit the structure of dialog acts to build a multi-layer hierarchical graph, where each act is represented as a root-to-leaf route on the graph. Then, they incorporate such graph structure prior as an inductive bias to build a hierarchical disentangled self-attention network, where they disentangle attention heads to model designated nodes on the dialog act graph. By activating different (disentangled) heads at each layer, combinatorially many dialog act semantics can be modeled to control the neural response generation.

### Data

MultiWOZ dataset: https://github.com/budzianowski/multiwoz

### Approach

1. Dialogue Act Representation: Graph Structure to capture cross-branch relationshi and lower sample complexity.

![Screen Shot 2021-09-16 at 10 19 53 AM](https://user-images.githubusercontent.com/15247433/133628974-c9512fd1-3ab1-4c11-88b8-85b30e856d04.png)

2. Model.

  a. Dialog Act Predictor: Neural network to take in dialog history and predict dialogue act.
  
  b. Disentangled Self-Attention: The authors propose to use a switch to activate certain heads and only pass through their information to the next level, hence they are able to disentangle the H attention heads to model H different semantic functionalities, and they refer to such a module as the disentangled self-attention (DSA).
  
  c. Hierarchical DSA: They further propose to stack multiple DSA layers to better model the huge semantic space with strong compositionality.
  
  ![Screen Shot 2021-09-16 at 10 28 22 AM](https://user-images.githubusercontent.com/15247433/133630430-7b8a5167-bb4b-4089-8758-c9a73115d132.png)

### Evaluation

Metric: BLEU, Entiry F1(entit coverage accuracy)

![Screen Shot 2021-09-16 at 10 30 18 AM](https://user-images.githubusercontent.com/15247433/133630764-4899028b-ec02-4c1a-950c-cfa740891cb2.png)

## [OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs]()

### Task

The authors study a conversational reasoning model that strategically traverses through a large- scale common fact knowledge graph (KG) to introduce engaging and contextually diverse entities and attributes. And then propose the DialKG Walker model that learns the symbolic transitions of dialog contexts as structured traversals over KG, and predicts natural entities to introduce given previous dialog contexts via a novel domain-agnostic, attention-based graph path decoder.

### Data

OpenDialKG: https://github.com/facebookresearch/opendialkg

### Approach

The model retrieves a set of entities from a provided KG given multiple modalities of dialog contexts. Specifically, for each turn the model takes as input a set of KG entities mentioned at its current turn, a full sentence at the current turn, and all sentences from previous turns of dialog, which are encoded using Bi-LSTMs with self-attention modules. The auto-regressive graph decoder takes attention-based encoder output at each decoding step to generate a walk path for each starting KG entity, which is combined with zeroshot KG embeddings prediction results to rank candidate entities.

![Screen Shot 2021-09-16 at 10 35 57 AM](https://user-images.githubusercontent.com/15247433/133631736-d7c6371b-8383-48ac-9248-6f39b21208b1.png)

### Evaluation

Metric: recall@N

![Screen Shot 2021-09-16 at 10 36 50 AM](https://user-images.githubusercontent.com/15247433/133631907-db954e0d-0730-42fc-9d45-f91cf7d6b88d.png)

## [Incremental Learning from Scratch for Task-Oriented Dialogue Systems]()

### Task

Clarifying user needs is essential for existing task-oriented dialogue systems. However, in real-world applications, developers can never guarantee that all possible user demands are taken into account in the design phase. Consequently, existing systems will break down when encountering unconsidered user needs. To address this problem, the authors propose a novel incremental learning framework to design task-oriented dialogue systems, or for short Incremental Dialogue System (IDS), without pre-defining the exhaustive list of user needs. Specifically, they introduce an uncertainty estimation module to evaluate the confidence of giving correct responses. If there is high confidence, IDS will provide responses to users. Otherwise, humans will be involved in the dialogue process, and IDS can learn from human intervention through an online learning module.

### Data

Self-constructed. No link.

### Approach

IDS consists of three main components: dialogue embedding module, uncertainty estimation module and online learning module.

Dialogue Embedding: Bi-RNNs

Uncertainty Estimation: If the confidence is high enough, IDS will give the response with the maximum score in Pavg to the user. Otherwise, the hired customer service staffs will be asked to select an appropriate response from the top T response candidates of P(avg) or propose a new response if there is no appropriate candidate.

![Screen Shot 2021-09-16 at 10 54 44 AM](https://user-images.githubusercontent.com/15247433/133635009-ef268f43-b5a3-4b06-98b7-f7653157d2d4.png)

### Evaluation

Metric: Average turn accuracy

![Screen Shot 2021-09-16 at 10 57 15 AM](https://user-images.githubusercontent.com/15247433/133635433-b1bbba63-b350-4e3c-ae5e-e6f70962d95a.png)

## [ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation]()

### Task

propose a new model, named ReCoSa, to tackle the problem which detects relevant contexts discriminately and produce a suitable response accordingly.

### Data

JDC: https://www.jddc.jd.com (cannot open)

Ubuntu: https://github.com/rkadlec/ubuntu-ranking-dataset-creator

### Approach

ReCoSa consists of a context representation encoder, a response representation encoder and a context-response attention decoder. For each part, the multi-head self-attention module obtains the context representation, response representation and the context-response attention weights. Firstly, the word-level encoder encodes each context as a low-dimension representation. And then, a multi-head self-attention component transforms these representations and position embeddings to the context attention representation. Secondly, another multi-head self-attention component transforms the masked response’s word embedding and position embedding to the response attention representation. Thirdly, the third multi-head attention component feeds the context representation as key and value, and the response representation as query in the context-response attention module. Finally, a softmax layer uses the output of the third multi-head attention component to obtain the word probability for the generation process.

![Screen Shot 2021-09-16 at 11 04 47 AM](https://user-images.githubusercontent.com/15247433/133636778-7f64b760-286d-4c8b-b3f1-2e2ac82c27f7.png)

### Evaluation

Quantitative: PPL and BLEU didn't specifically give tables, just saying their model is the best with one BLUE score.

Human judgement: win, loss, tie

![Screen Shot 2021-09-16 at 11 09 23 AM](https://user-images.githubusercontent.com/15247433/133637591-065197c2-d088-4358-a870-e7dc6b508086.png)

## [Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems]()

### Task

In this paper, the authors propose a TRAnsferable Dialogue statE generator (TRADE) that generates dialogue states from utterances using a copy mechanism, facilitating knowledge transfer when predicting (domain, slot, value) triplets not encountered during training. The model is composed of an utterance encoder, a slot gate, and a state generator, which are shared across domains.

### Data

MultiWOZ: https://github.com/budzianowski/multiwoz

### Approach

The proposed model comprises three components: an utterance encoder, a slot gate, and a state generator. Instead of predicting the probability of every predefined ontology term, our model directly generates slot values. All the model parameters are shared, and the state generator starts with a different start-of-sentence token for each (domain, slot) pair. The utterance encoder encodes dialogue utterances into a sequence of fixed-length vectors. To determine whether any of the (domain, slot) pairs are mentioned, the context-enhanced slot gate is used with the state generator. The state generator decodes multiple output tokens for all (domain, slot) pairs independently to predict their corresponding values. The context-enhanced slot gate predicts whether each of the pairs is actually triggered by the dialogue via a three-way classifier.

![Screen Shot 2021-09-16 at 11 15 35 AM](https://user-images.githubusercontent.com/15247433/133638634-6765edfc-7b59-4bfb-b468-664958074aa6.png)

### Evaluation

Metric: joint goal accuracy, slot accuracy

![Screen Shot 2021-09-16 at 11 17 15 AM](https://user-images.githubusercontent.com/15247433/133638903-b5f060f9-8adc-4e2e-95cd-54ba0fdf8529.png)

## [Learning a Matching Model with Co-teaching for Multi-turn Response Selection in Retrieval-based Dialogue Systems]()

### Task

Study learning of a matching model for response selection in retrieval-based dialogue systems. The authors propose a general co-teaching framework with three specific teaching strategies that cover both teaching with loss functions and teaching with data curriculum. Under the framework, we simultaneously learn two matching models with independent training sets. In each iteration, one model transfers the knowledge learned from its training set to the other model, and at the same time receives the guide from the other model on how to overcome noise in training. Through being both a teacher and a student, the two models learn from each other and get improved together.

### Data

Douban Conversation Corpus (Douban): https://github.com/MarkWuNLP/MultiTurnResponseSelection

E-commerce Dialogue Corpus: https://github.com/cooelf/DeepUtteranceAggregation

E-commerce Dialogue Corpus Labeled Test Data: https://drive.google.com/open?id=1HMDHRU8kbbWTsPVr6lKU_-Z2Jt-n-dys.

### Approach

Co-teaching Framework
  1. Dynamic Margins
  2. Dynamic Instance Weighting
  3. Dynamic Data Curriculum
 
![Screen Shot 2021-09-16 at 11 41 32 AM](https://user-images.githubusercontent.com/15247433/133642805-02e0b60e-7482-465d-911e-55f660861066.png)

### Evaluation

Metrics: MAP, MRR, P@1, R@N

![Screen Shot 2021-09-16 at 11 44 03 AM](https://user-images.githubusercontent.com/15247433/133643247-46251ec3-58e7-4866-b624-0210e5ebb223.png)

## [The PhotoBook Dataset:Building Common Ground through Visually-Grounded Dialogue]()

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


