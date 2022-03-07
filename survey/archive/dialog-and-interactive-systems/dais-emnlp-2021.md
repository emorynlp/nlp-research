# EMNLP 2021: Dialog and Interactive Systems (Main)



## 1. [Automatically Exposing Problems with Neural Dialog Models](https://aclanthology.org/2021.emnlp-main.37.pdf)
Dian Yu, Kenji Sagae
<details>
	<summary>
	Abstract
	</summary>
	Neural dialog models are known to suffer from problems such as generating unsafe and inconsistent responses. Even though these problems are crucial and prevalent, they are mostly manually identified by model designers through interactions. Recently, some research instructs crowdworkers to goad the bots into triggering such problems. However, humans leverage superficial clues such as hate speech, while leaving systematic problems undercover. In this paper, we propose two methods including reinforcement learning to automatically trigger a dialog model into generating problematic responses. We show the effect of our methods in exposing safety and contradiction issues with state-of-the-art dialog models.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 2. [Detecting Speaker Personas from Conversational Texts](https://aclanthology.org/2021.emnlp-main.86.pdf)
Jia-Chen Gu, Zhenhua Ling, Yu Wu, Quan Liu, Zhigang Chen, Xiaodan Zhu
<details>
	<summary>
	Abstract
	</summary>
	Personas are useful for dialogue response prediction. However, the personas used in current studies are pre-defined and hard to obtain before a conversation. To tackle this issue, we study a new task, named Speaker Persona Detection (SPD), which aims to detect speaker personas based on the plain conversational text. In this task, a best-matched persona is searched out from candidates given the conversational text. This is a many-to-many semantic matching task because both contexts and personas in SPD are composed of multiple sentences. The long-term dependency and the dynamic redundancy among these sentences increase the difficulty of this task. We build a dataset for SPD, dubbed as Persona Match on Persona-Chat (PMPC). Furthermore, we evaluate several baseline models and propose utterance-to-profile (U2P) matching networks for this task. The U2P models operate at a fine granularity which treat both contexts and personas as sets of multiple sequences. Then, each sequence pair is scored and an interpretable overall score is obtained for a context-persona pair through aggregation. Evaluation results show that the U2P models outperform their baseline counterparts significantly.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 3. [<fixed-case>ConvFiT:</fixed-case> <fixed-case>C</fixed-case>onversational Fine-Tuning of Pretrained Language Models](https://aclanthology.org/2021.emnlp-main.88.pdf)
Ivan Vulić, Pei-Hao Su, Samuel Coope, Daniela Gerz, Paweł Budzianowski, Iñigo Casanueva, Nikola Mrkšić, Tsung-Hsien Wen
<details>
	<summary>
	Abstract
	</summary>
	Transformer-based language models (LMs) pretrained on large text collections are proven to store a wealth of semantic knowledge. However, 1) they are not effective as sentence encoders when used off-the-shelf, and 2) thus typically lag behind conversationally pretrained (e.g., via response selection) encoders on conversational tasks such as intent detection (ID). In this work, we propose ConvFiT, a simple and efficient two-stage procedure which turns any pretrained LM into a universal conversational encoder (after Stage 1 ConvFiT-ing) and task-specialised sentence encoder (after Stage 2). We demonstrate that 1) full-blown conversational pretraining is not required, and that LMs can be quickly transformed into effective conversational encoders with much smaller amounts of unannotated data; 2) pretrained LMs can be fine-tuned into task-specialised sentence encoders, optimised for the fine-grained semantics of a particular task. Consequently, such specialised sentence encoders allow for treating ID as a simple semantic similarity task based on interpretable nearest neighbours retrieval. We validate the robustness and versatility of the ConvFiT framework with such similarity-based inference on the standard ID evaluation sets: ConvFiT-ed LMs achieve state-of-the-art ID performance across the board, with particular gains in the most challenging, few-shot setups.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 4. [<fixed-case>EARL</fixed-case>: Informative Knowledge-Grounded Conversation Generation with Entity-Agnostic Representation Learning](https://aclanthology.org/2021.emnlp-main.184.pdf)
Hao Zhou, Minlie Huang, Yong Liu, Wei Chen, Xiaoyan Zhu
<details>
	<summary>
	Abstract
	</summary>
	Generating informative and appropriate responses is challenging but important for building human-like dialogue systems. Although various knowledge-grounded conversation models have been proposed, these models have limitations in utilizing knowledge that infrequently occurs in the training data, not to mention integrating unseen knowledge into conversation generation. In this paper, we propose an Entity-Agnostic Representation Learning (EARL) method to introduce knowledge graphs to informative conversation generation. Unlike traditional approaches that parameterize the specific representation for each entity, EARL utilizes the context of conversations and the relational structure of knowledge graphs to learn the category representation for entities, which is generalized to incorporating unseen entities in knowledge graphs into conversation generation. Automatic and manual evaluations demonstrate that our model can generate more informative, coherent, and natural responses than baseline models.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 5. [Towards Automatic Evaluation of Dialog Systems: A Model-Free Off-Policy Evaluation Approach](https://aclanthology.org/2021.emnlp-main.589.pdf)
Haoming Jiang, Bo Dai, Mengjiao Yang, Tuo Zhao, Wei Wei
<details>
	<summary>
	Abstract
	</summary>
	Reliable automatic evaluation of dialogue systems under an interactive environment has long been overdue. An ideal environment for evaluating dialog systems, also known as the Turing test, needs to involve human interaction, which is usually not affordable for large-scale experiments. Though researchers have attempted to use metrics for language generation tasks (e.g., perplexity, BLEU) or some model-based reinforcement learning methods (e.g., self-play evaluation) for automatic evaluation, these methods only show very weak correlation with the actual human evaluation in practice. To bridge such a gap, we propose a new framework named ENIGMA for estimating human evaluation scores based on recent advances of off-policy evaluation in reinforcement learning. ENIGMA only requires a handful of pre-collected experience data, and therefore does not involve human interaction with the target policy during the evaluation, making automatic evaluations feasible. More importantly, ENIGMA is model-free and agnostic to the behavior policies for collecting the experience data, which significantly alleviates the technical difficulties of modeling complex dialogue environments and human behaviors. Our experiments show that ENIGMA significantly outperforms existing methods in terms of correlation with human evaluation scores.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 6. [Perspective-taking and Pragmatics for Generating Empathetic Responses Focused on Emotion Causes](https://aclanthology.org/2021.emnlp-main.170.pdf)
Hyunwoo Kim, Byeongchang Kim, Gunhee Kim
<details>
	<summary>
	Abstract
	</summary>
	Empathy is a complex cognitive ability based on the reasoning of others’ affective states. In order to better understand others and express stronger empathy in dialogues, we argue that two issues must be tackled at the same time: (i) identifying which word is the cause for the other’s emotion from his or her utterance and (ii) reflecting those specific words in the response generation. However, previous approaches for recognizing emotion cause words in text require sub-utterance level annotations, which can be demanding. Taking inspiration from social cognition, we leverage a generative estimator to infer emotion cause words from utterances with no word-level label. Also, we introduce a novel method based on pragmatics to make dialogue models focus on targeted words in the input during generation. Our method is applicable to any dialogue models with no additional training on the fly. We show our approach improves multiple best-performing dialogue agents on generating more focused empathetic responses in terms of both automatic and human evaluation.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 7. [<fixed-case>C</fixed-case>onv<fixed-case>A</fixed-case>buse: Data, Analysis, and Benchmarks for Nuanced Detection in Conversational <fixed-case>AI</fixed-case>](https://aclanthology.org/2021.emnlp-main.587.pdf)
Amanda Cercas Curry, Gavin Abercrombie, Verena Rieser
<details>
	<summary>
	Abstract
	</summary>
	We present the first English corpus study on abusive language towards three conversational AI systems gathered ‘in the wild’: an open-domain social bot, a rule-based chatbot, and a task-based system. To account for the complexity of the task, we take a more ‘nuanced’ approach where our ConvAI dataset reflects fine-grained notions of abuse, as well as views from multiple expert annotators. We find that the distribution of abuse is vastly different compared to other commonly used datasets, with more sexually tinted aggression towards the virtual persona of these systems. Finally, we report results from bench-marking existing models against this data. Unsurprisingly, we find that there is substantial room for improvement with F1 scores below 90%.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 8. [Transferable Persona-Grounded Dialogues via Grounded Minimal Edits](https://aclanthology.org/2021.emnlp-main.183.pdf)
Chen Henry Wu, Yinhe Zheng, Xiaoxi Mao, Minlie Huang
<details>
	<summary>
	Abstract
	</summary>
	Grounded dialogue models generate responses that are grounded on certain concepts. Limited by the distribution of grounded dialogue data, models trained on such data face the <i>transferability</i> challenges in terms of the data distribution and the type of grounded concepts. To address the challenges, we propose the <i>grounded minimal editing</i> framework, which minimally edits existing responses to be grounded on the given concept. Focusing on personas, we propose Grounded Minimal Editor (GME), which learns to edit by disentangling and recombining persona-related and persona-agnostic parts of the response. To evaluate persona-grounded minimal editing, we present the PersonaMi-nEdit dataset, and experimental results show that GME outperforms competitive baselines by a large margin. To evaluate the transferability, we experiment on the test set of BlendedSkillTalk and show that GME can edit dialogue models’ responses to largely improve their persona consistency while preserving the use of knowledge and empathy.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 9. [<tex-math>Q^{2}</tex-math>: <fixed-case>E</fixed-case>valuating Factual Consistency in Knowledge-Grounded Dialogues via Question Generation and Question Answering](https://aclanthology.org/2021.emnlp-main.619.pdf)
Or Honovich, Leshem Choshen, Roee Aharoni, Ella Neeman, Idan Szpektor, Omri Abend
<details>
	<summary>
	Abstract
	</summary>
	Neural knowledge-grounded generative models for dialogue often produce content that is factually inconsistent with the knowledge they rely on, making them unreliable and limiting their applicability. Inspired by recent work on evaluating factual consistency in abstractive summarization, we propose an automatic evaluation metric for factual consistency in knowledge-grounded dialogue using automatic question generation and question answering. Our metric, denoted <tex-math>Q^2</tex-math>, compares answer spans using natural language inference (NLI), instead of token-based matching as done in previous work. To foster proper evaluation, we curate a novel dataset of dialogue system outputs for the Wizard-of-Wikipedia dataset, manually annotated for factual consistency. We perform a thorough meta-evaluation of <tex-math>Q^2</tex-math> against other metrics using this dataset and two others, where it consistently shows higher correlation with human judgements.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 10. [<fixed-case>CSAGN</fixed-case>: Conversational Structure Aware Graph Network for Conversational Semantic Role Labeling](https://aclanthology.org/2021.emnlp-main.177.pdf)
Han Wu, Kun Xu, Linqi Song
<details>
	<summary>
	Abstract
	</summary>
	Conversational semantic role labeling (CSRL) is believed to be a crucial step towards dialogue understanding. However, it remains a major challenge for existing CSRL parser to handle conversational structural information. In this paper, we present a simple and effective architecture for CSRL which aims to address this problem. Our model is based on a conversational structure aware graph network which explicitly encodes the speaker dependent information. We also propose a multi-task learning method to further improve the model. Experimental results on benchmark datasets show that our model with our proposed training objectives significantly outperforms previous baselines.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 11. [Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems](https://aclanthology.org/2021.emnlp-main.169.pdf)
Yicheng Zou, Zhihua Liu, Xingwu Hu, Qi Zhang
<details>
	<summary>
	Abstract
	</summary>
	Human dialogue contains evolving concepts, and speakers naturally associate multiple concepts to compose a response. However, current dialogue models with the seq2seq framework lack the ability to effectively manage concept transitions and can hardly introduce multiple concepts to responses in a sequential decoding manner. To facilitate a controllable and coherent dialogue, in this work, we devise a concept-guided non-autoregressive model (CG-nAR) for open-domain dialogue generation. The proposed model comprises a multi-concept planning module that learns to identify multiple associated concepts from a concept graph and a customized Insertion Transformer that performs concept-guided non-autoregressive generation to complete a response. The experimental results on two public datasets show that CG-nAR can produce diverse and coherent responses, outperforming state-of-the-art baselines in both automatic and human evaluations with substantially faster inference speed.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 12. [Don’t be Contradicted with Anything! <fixed-case>CI</fixed-case>-<fixed-case>T</fixed-case>o<fixed-case>D</fixed-case>: Towards Benchmarking Consistency for Task-oriented Dialogue System](https://aclanthology.org/2021.emnlp-main.182.pdf)
Libo Qin, Tianbao Xie, Shijue Huang, Qiguang Chen, Xiao Xu, Wanxiang Che
<details>
	<summary>
	Abstract
	</summary>
	Consistency Identification has obtained remarkable success on open-domain dialogue, which can be used for preventing inconsistent response generation. However, in contrast to the rapid development in open-domain dialogue, few efforts have been made to the task-oriented dialogue direction. In this paper, we argue that <i>consistency problem</i> is more urgent in task-oriented domain. To facilitate the research, we introduce CI-ToD, a novel dataset for <b>C</b>onsistency <b>I</b>dentification in <b>T</b>ask-<b>o</b>riented <b>D</b>ialog system. In addition, we not only annotate the single label to enable the model to judge whether the system response is contradictory, but also provide more fine-grained labels (i.e., Dialogue History Inconsistency, User Query Inconsistency and Knowledge Base Inconsistency) to encourage model to know what inconsistent sources lead to it. Empirical results show that state-of-the-art methods only achieve 51.3%, which is far behind the human performance of 93.2%, indicating that there is ample room for improving consistency identification ability. Finally, we conduct exhaustive experiments and qualitative analysis to comprehend key challenges and provide guidance for future directions. All datasets and models are publicly available at <url>https://github.com/yizhen20133868/CI-ToD</url>.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 13. [<fixed-case>C</fixed-case>o<fixed-case>LV</fixed-case>: A Collaborative Latent Variable Model for Knowledge-Grounded Dialogue Generation](https://aclanthology.org/2021.emnlp-main.172.pdf)
Haolan Zhan, Lei Shen, Hongshen Chen, Hainan Zhang
<details>
	<summary>
	Abstract
	</summary>
	Knowledge-grounded dialogue generation has achieved promising performance with the engagement of external knowledge sources. Typical approaches towards this task usually perform relatively independent two sub-tasks, i.e., knowledge selection and knowledge-aware response generation. In this paper, in order to improve the diversity of both knowledge selection and knowledge-aware response generation, we propose a collaborative latent variable (CoLV) model to integrate these two aspects simultaneously in separate yet collaborative latent spaces, so as to capture the inherent correlation between knowledge selection and response generation. During generation, our proposed model firstly draws knowledge candidate from the latent space conditioned on the dialogue context, and then samples a response from another collaborative latent space conditioned on both the context and the selected knowledge. Experimental results on two widely-used knowledge-grounded dialogue datasets show that our model outperforms previous methods on both knowledge selection and response generation.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 14. [More is Better: Enhancing Open-Domain Dialogue Generation via Multi-Source Heterogeneous Knowledge](https://aclanthology.org/2021.emnlp-main.175.pdf)
Sixing Wu, Ying Li, Minghui Wang, Dawei Zhang, Yang Zhou, Zhonghai Wu
<details>
	<summary>
	Abstract
	</summary>
	Despite achieving remarkable performance, previous knowledge-enhanced works usually only use a single-source homogeneous knowledge base of limited knowledge coverage. Thus, they often degenerate into traditional methods because not all dialogues can be linked with knowledge entries. This paper proposes a novel dialogue generation model, MSKE-Dialog, to solve this issue with three unique advantages: (1) Rather than only one, MSKE-Dialog can simultaneously leverage multiple heterogeneous knowledge sources (it includes but is not limited to commonsense knowledge facts, text knowledge, infobox knowledge) to improve the knowledge coverage; (2) To avoid the topic conflict among the context and different knowledge sources, we propose a Multi-Reference Selection to better select context/knowledge; (3) We propose a Multi-Reference Generation to generate informative responses by referring to multiple generation references at the same time. Extensive evaluations on a Chinese dataset show the superior performance of this work against various state-of-the-art approaches. To our best knowledge, this work is the first to use the multi-source heterogeneous knowledge in the open-domain knowledge-enhanced dialogue generation.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 15. [Graph Based Network with Contextualized Representations of Turns in Dialogue](https://aclanthology.org/2021.emnlp-main.36.pdf)
Bongseok Lee, Yong Suk Choi
<details>
	<summary>
	Abstract
	</summary>
	Dialogue-based relation extraction (RE) aims to extract relation(s) between two arguments that appear in a dialogue. Because dialogues have the characteristics of high personal pronoun occurrences and low information density, and since most relational facts in dialogues are not supported by any single sentence, dialogue-based relation extraction requires a comprehensive understanding of dialogue. In this paper, we propose the TUrn COntext awaRE Graph Convolutional Network (TUCORE-GCN) modeled by paying attention to the way people understand dialogues. In addition, we propose a novel approach which treats the task of emotion recognition in conversations (ERC) as a dialogue-based RE. Experiments on a dialogue-based RE dataset and three ERC datasets demonstrate that our model is very effective in various dialogue-based natural language understanding tasks. In these experiments, TUCORE-GCN outperforms the state-of-the-art models on most of the benchmark datasets. Our code is available at https://github.com/BlackNoodle/TUCORE-GCN.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 16. [Neural Path Hunter: Reducing Hallucination in Dialogue Systems via Path Grounding](https://aclanthology.org/2021.emnlp-main.168.pdf)
Nouha Dziri, Andrea Madotto, Osmar Zaïane, Avishek Joey Bose
<details>
	<summary>
	Abstract
	</summary>
	Dialogue systems powered by large pre-trained language models exhibit an innate ability to deliver fluent and natural-sounding responses. Despite their impressive performance, these models are fitful and can often generate factually incorrect statements impeding their widespread adoption. In this paper, we focus on the task of improving faithfulness and reducing hallucination of neural dialogue systems to known facts supplied by a Knowledge Graph (KG). We propose Neural Path Hunter which follows a generate-then-refine strategy whereby a generated response is amended using the KG. Neural Path Hunter leverages a separate token-level fact critic to identify plausible sources of hallucination followed by a refinement stage that retrieves correct entities by crafting a query signal that is propagated over a k-hop subgraph. We empirically validate our proposed approach on the OpenDialKG dataset (Moon et al., 2019) against a suite of metrics and report a relative improvement of faithfulness over dialogue responses by 20.35% based on FeQA (Durmus et al., 2020). The code is available at https://github.com/nouhadziri/Neural-Path-Hunter.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 17. [Investigating Robustness of Dialog Models to Popular Figurative Language Constructs](https://aclanthology.org/2021.emnlp-main.592.pdf)
Harsh Jhamtani, Varun Gangal, Eduard Hovy, Taylor Berg-Kirkpatrick
<details>
	<summary>
	Abstract
	</summary>
	Humans often employ figurative language use in communication, including during interactions with dialog systems. Thus, it is important for real-world dialog systems to be able to handle popular figurative language constructs like metaphor and simile. In this work, we analyze the performance of existing dialog models in situations where the input dialog context exhibits use of figurative language. We observe large gaps in handling of figurative language when evaluating the models on two open domain dialog datasets. When faced with dialog contexts consisting of figurative language, some models show very large drops in performance compared to contexts without figurative language. We encourage future research in dialog modeling to separately analyze and report results on figurative language in order to better test model capabilities relevant to real-world use. Finally, we propose lightweight solutions to help existing models become more robust to figurative language by simply using an external resource to translate figurative language to literal (non-figurative) forms while preserving the meaning to the best extent possible.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 18. [<fixed-case>DIALKI</fixed-case>: Knowledge Identification in Conversational Systems through Dialogue-Document Contextualization](https://aclanthology.org/2021.emnlp-main.140.pdf)
Zeqiu Wu, Bo-Ru Lu, Hannaneh Hajishirzi, Mari Ostendorf
<details>
	<summary>
	Abstract
	</summary>
	Identifying relevant knowledge to be used in conversational systems that are grounded in long documents is critical to effective response generation. We introduce a knowledge identification model that leverages the document structure to provide dialogue-contextualized passage encodings and better locate knowledge relevant to the conversation. An auxiliary loss captures the history of dialogue-document connections. We demonstrate the effectiveness of our model on two document-grounded conversational datasets and provide analyses showing generalization to unseen documents and long dialogue contexts.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 19. [Zero-Shot Dialogue Disentanglement by Self-Supervised Entangled Response Selection](https://aclanthology.org/2021.emnlp-main.400.pdf)
Ta-Chung Chi, Alexander Rudnicky
<details>
	<summary>
	Abstract
	</summary>
	Dialogue disentanglement aims to group utterances in a long and multi-participant dialogue into threads. This is useful for discourse analysis and downstream applications such as dialogue response selection, where it can be the first step to construct a clean context/response set. Unfortunately, labeling all <i>reply-to</i> links takes quadratic effort w.r.t the number of utterances: an annotator must check all preceding utterances to identify the one to which the current utterance is a reply. In this paper, we are the first to propose a <b>zero-shot</b> dialogue disentanglement solution. Firstly, we train a model on a multi-participant response selection dataset harvested from the web which is not annotated; we then apply the trained model to perform zero-shot dialogue disentanglement. Without any labeled data, our model can achieve a cluster F1 score of 25. We also fine-tune the model using various amounts of labeled data. Experiments show that with only 10% of the data, we achieve nearly the same performance of using the full dataset.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 20. [Proxy Indicators for the Quality of Open-domain Dialogues](https://aclanthology.org/2021.emnlp-main.618.pdf)
Rostislav Nedelchev, Jens Lehmann, Ricardo Usbeck
<details>
	<summary>
	Abstract
	</summary>
	The automatic evaluation of open-domain dialogues remains a largely unsolved challenge. Despite the abundance of work done in the field, human judges have to evaluate dialogues’ quality. As a consequence, performing such evaluations at scale is usually expensive. This work investigates using a deep-learning model trained on the General Language Understanding Evaluation (GLUE) benchmark to serve as a quality indication of open-domain dialogues. The aim is to use the various GLUE tasks as different perspectives on judging the quality of conversation, thus reducing the need for additional training data or responses that serve as quality references. Due to this nature, the method can infer various quality metrics and can derive a component-based overall score. We achieve statistically significant correlation coefficients of up to 0.7.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 21. [<fixed-case>MRF</fixed-case>-Chat: Improving Dialogue with <fixed-case>M</fixed-case>arkov Random Fields](https://aclanthology.org/2021.emnlp-main.403.pdf)
Ishaan Grover, Matthew Huggins, Cynthia Breazeal, Hae Won Park
<details>
	<summary>
	Abstract
	</summary>
	Recent state-of-the-art approaches in open-domain dialogue include training end-to-end deep-learning models to learn various conversational features like emotional content of response, symbolic transitions of dialogue contexts in a knowledge graph and persona of the agent and the user, among others. While neural models have shown reasonable results, modelling the cognitive processes that humans use when conversing with each other may improve the agent’s quality of responses. A key element of natural conversation is to tailor one’s response such that it accounts for concepts that the speaker and listener may or may not know and the contextual relevance of all prior concepts used in conversation. We show that a rich representation and explicit modeling of these psychological processes can improve predictions made by existing neural network models. In this work, we propose a novel probabilistic approach using Markov Random Fields (MRF) to augment existing deep-learning methods for improved next utterance prediction. Using human and automatic evaluations, we show that our augmentation approach significantly improves the performance of existing state-of-the-art retrieval models for open-domain conversational agents.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 22. [Conversational Multi-Hop Reasoning with Neural Commonsense Knowledge and Symbolic Logic Rules](https://aclanthology.org/2021.emnlp-main.588.pdf)
Forough Arabshahi, Jennifer Lee, Antoine Bosselut, Yejin Choi, Tom Mitchell
<details>
	<summary>
	Abstract
	</summary>
	One of the challenges faced by conversational agents is their inability to identify unstated presumptions of their users’ commands, a task trivial for humans due to their common sense. In this paper, we propose a zero-shot commonsense reasoning system for conversational agents in an attempt to achieve this. Our reasoner uncovers unstated presumptions from user commands satisfying a general template of if-(state), then-(action), because-(goal). Our reasoner uses a state-of-the-art transformer-based generative commonsense knowledge base (KB) as its source of background knowledge for reasoning. We propose a novel and iterative knowledge query mechanism to extract multi-hop reasoning chains from the neural KB which uses symbolic logic rules to significantly reduce the search space. Similar to any KBs gathered to date, our commonsense KB is prone to missing knowledge. Therefore, we propose to conversationally elicit the missing knowledge from human users with our novel dynamic question generation strategy, which generates and presents contextualized queries to human users. We evaluate the model with a user study with human users that achieves a 35% higher success rate compared to SOTA.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.

***

#Findings of EMNLP 2021



## 23. [Refine and Imitate: Reducing Repetition and Inconsistency in Persuasion Dialogues via Reinforcement Learning and Human Demonstration](https://aclanthology.org/2021.findings-emnlp.295.pdf)
Weiyan Shi, Yu Li, Saurav Sahay, Zhou Yu
<details>
	<summary>
	Abstract
	</summary>
	Persuasion dialogue system reflects the machine’s ability to make strategic moves beyond verbal communication, and therefore differentiates itself from task-oriented or open-domain dialogues and has its own unique values. However, the repetition and inconsistency problems still persist in dialogue response generation and could substantially impact user experience and impede the persuasion outcome. Besides, although reinforcement learning (RL) approaches have achieved big success in strategic tasks such as games, it requires a sophisticated user simulator to provide real-time feedback to the dialogue system, which limits the application of RL on persuasion dialogues. To address these issues towards a better persuasion dialogue system, we apply RL to refine a language model baseline without user simulators, and distill sentence-level information about repetition, inconsistency, and task relevance through rewards. Moreover, to better accomplish the persuasion task, the model learns from human demonstration to imitate human persuasion behavior and selects the most persuasive responses. Experiments show that our model outperforms previous state-of-the-art dialogue models on both automatic metrics and human evaluation results on a donation persuasion task, and generates more diverse, consistent and persuasive conversations according to the user feedback. We will make the code and model publicly available.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 24. [<fixed-case>FCM</fixed-case>: A Fine-grained Comparison Model for Multi-turn Dialogue Reasoning](https://aclanthology.org/2021.findings-emnlp.362.pdf)
Xu Wang, Hainan Zhang, Shuai Zhao, Yanyan Zou, Hongshen Chen, Zhuoye Ding, Bo Cheng, Yanyan Lan
<details>
	<summary>
	Abstract
	</summary>
	Despite the success of neural dialogue systems in achieving high performance on the leader-board, they cannot meet users’ requirements in practice, due to their poor reasoning skills. The underlying reason is that most neural dialogue models only capture the syntactic and semantic information, but fail to model the logical consistency between the dialogue history and the generated response. Recently, a new multi-turn dialogue reasoning task has been proposed, to facilitate dialogue reasoning research. However, this task is challenging, because there are only slight differences between the illogical response and the dialogue history. How to effectively solve this challenge is still worth exploring. This paper proposes a Fine-grained Comparison Model (FCM) to tackle this problem. Inspired by human’s behavior in reading comprehension, a comparison mechanism is proposed to focus on the fine-grained differences in the representation of each response candidate. Specifically, each candidate representation is compared with the whole history to obtain a history consistency representation. Furthermore, the consistency signals between each candidate and the speaker’s own history are considered to drive a model prefer a candidate that is logically consistent with the speaker’s history logic. Finally, the above consistency representations are employed to output a ranking list of the candidate responses for multi-turn dialogue reasoning. Experimental results on two public dialogue datasets show that our method obtains higher ranking scores than the baseline models.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 25. [Combining Curriculum Learning and Knowledge Distillation for Dialogue Generation](https://aclanthology.org/2021.findings-emnlp.111.pdf)
Qingqing Zhu, Xiuying Chen, Pengfei Wu, JunFei Liu, Dongyan Zhao
<details>
	<summary>
	Abstract
	</summary>
	Curriculum learning, a machine training strategy that feeds training instances to the model from easy to hard, has been proven to facilitate the dialogue generation task. Meanwhile, knowledge distillation, a knowledge transformation methodology among teachers and students networks can yield significant performance boost for student models. Hence, in this paper, we introduce a combination of curriculum learning and knowledge distillation for efficient dialogue generation models, where curriculum learning can help knowledge distillation from data and model aspects. To start with, from the data aspect, we cluster the training cases according to their complexity, which is calculated by various types of features such as sentence length and coherence between dialog pairs. Furthermore, we employ an adversarial training strategy to identify the complexity of cases from model level. The intuition is that, if a discriminator can tell the generated response is from the teacher or the student, then the case is difficult that the student model has not adapted to yet. Finally, we use self-paced learning, which is an extension to curriculum learning to assign weights for distillation. In conclusion, we arrange a hierarchical curriculum based on the above two aspects for the student model under the guidance from the teacher model. Experimental results demonstrate that our methods achieve improvements compared with competitive baselines.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 26. [Improving Empathetic Response Generation by Recognizing Emotion Cause in Conversations](https://aclanthology.org/2021.findings-emnlp.70.pdf)
Jun Gao, Yuhan Liu, Haolin Deng, Wei Wang, Yu Cao, Jiachen Du, Ruifeng Xu
<details>
	<summary>
	Abstract
	</summary>
	Current approaches to empathetic response generation focus on learning a model to predict an emotion label and generate a response based on this label and have achieved promising results. However, the emotion cause, an essential factor for empathetic responding, is ignored. The emotion cause is a stimulus for human emotions. Recognizing the emotion cause is helpful to better understand human emotions so as to generate more empathetic responses. To this end, we propose a novel framework that improves empathetic response generation by recognizing emotion cause in conversations. Specifically, an emotion reasoner is designed to predict a context emotion label and a sequence of emotion cause-oriented labels, which indicate whether the word is related to the emotion cause. Then we devise both hard and soft gated attention mechanisms to incorporate the emotion cause into response generation. Experiments show that incorporating emotion cause information improves the performance of the model on both emotion recognition and response generation.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 27. [Constructing Emotional Consensus and Utilizing Unpaired Data for Empathetic Dialogue Generation](https://aclanthology.org/2021.findings-emnlp.268.pdf)
Lei Shen, Jinchao Zhang, Jiao Ou, Xiaofang Zhao, Jie Zhou
<details>
	<summary>
	Abstract
	</summary>
	Researches on dialogue empathy aim to endow an agent with the capacity of accurate understanding and proper responding for emotions. Existing models for empathetic dialogue generation focus on the emotion flow in one direction, that is, from the context to response. We argue that conducting an empathetic conversation is a bidirectional process, where empathy occurs when the emotions of two interlocutors could converge on the same point, i.e., reaching an emotional consensus. Besides, we also find that the empathetic dialogue corpus is extremely limited, which further restricts the model performance. To address the above issues, we propose a dual-generative model, Dual-Emp, to simultaneously construct the emotional consensus and utilize some external unpaired data. Specifically, our model integrates a forward dialogue model, a backward dialogue model, and a discrete latent variable representing the emotional consensus into a unified architecture. Then, to alleviate the constraint of paired data, we extract unpaired emotional data from open-domain conversations and employ Dual-Emp to produce pseudo paired empathetic samples, which is more efficient and low-cost than the human annotation. Automatic and human evaluations demonstrate that our method outperforms competitive baselines in producing coherent and empathetic responses.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 28. [Distilling the Knowledge of Large-scale Generative Models into Retrieval Models for Efficient Open-domain Conversation](https://aclanthology.org/2021.findings-emnlp.286.pdf)
Beomsu Kim, Seokjun Seo, Seungju Han, Enkhbayar Erdenee, Buru Chang
<details>
	<summary>
	Abstract
	</summary>
	Despite the remarkable performance of large-scale generative models in open-domain conversation, they are known to be less practical for building real-time conversation systems due to high latency. On the other hand, retrieval models could return responses with much lower latency but show inferior performance to the large-scale generative models since the conversation quality is bounded by the pre-defined response set. To take advantage of both approaches, we propose a new training method called G2R (Generative-to-Retrieval distillation) that preserves the efficiency of a retrieval model while leveraging the conversational ability of a large-scale generative model by infusing the knowledge of the generative model into the retrieval model. G2R consists of two distinct techniques of distillation: the data-level G2R augments the dialogue dataset with additional responses generated by the large-scale generative model, and the model-level G2R transfers the response quality score assessed by the generative model to the score of the retrieval model by the knowledge distillation loss. Through extensive experiments including human evaluation, we demonstrate that our retrieval-based conversation system trained with G2R shows a substantially improved performance compared to the baseline retrieval model while showing significantly lower inference latency than the large-scale generative models.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 29. [Retrieval Augmentation Reduces Hallucination in Conversation](https://aclanthology.org/2021.findings-emnlp.320.pdf)
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, Jason Weston
<details>
	<summary>
	Abstract
	</summary>
	Despite showing increasingly human-like conversational abilities, state-of-the-art dialogue models often suffer from factual incorrectness and hallucination of knowledge (Roller et al., 2020). In this work we explore the use of neural-retrieval-in-the-loop architectures - recently shown to be effective in open-domain QA (Lewis et al., 2020b; Izacard and Grave, 2020) - for knowledge-grounded dialogue, a task that is arguably more challenging as it requires querying based on complex multi-turn dialogue context and generating conversationally coherent responses. We study various types of architectures with multiple components - retrievers, rankers, and encoder-decoders - with the goal of maximizing knowledgeability while retaining conversational ability. We demonstrate that our best models obtain state-of-the-art performance on two knowledge-grounded conversational tasks. The models exhibit open-domain conversational capabilities, generalize effectively to scenarios not within the training data, and, as verified by human evaluations, substantially reduce the well-known problem of knowledge hallucination in state-of-the-art chatbots.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.



## 30. [<fixed-case>TIAGE</fixed-case>: A Benchmark for Topic-Shift Aware Dialog Modeling](https://aclanthology.org/2021.findings-emnlp.145.pdf)
Huiyuan Xie, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu, Ann Copestake
<details>
	<summary>
	Abstract
	</summary>
	Human conversations naturally evolve around different topics and fluently move between them. In research on dialog systems, the ability to actively and smoothly transition to new topics is often ignored. In this paper we introduce TIAGE, a new topic-shift aware dialog benchmark constructed utilizing human annotations on topic shifts. Based on TIAGE, we introduce three tasks to investigate different scenarios of topic-shift modeling in dialog settings: topic-shift detection, topic-shift triggered response generation and topic-aware dialog generation. Experiments on these tasks show that the topic-shift signals in TIAGE are useful for topic-shift response generation. On the other hand, dialog systems still struggle to decide when to change topic. This indicates further research is needed in topic-shift aware dialog modeling.
</details>

### Task

To be filled.

### Data

To be filled.

### Approach

To be filled.

### Evaluation

To be filled.

