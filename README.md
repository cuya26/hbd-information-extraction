# hbd-information-extraction

This repository contains scripts for the Information Extraction (IE) task of the WG1 for the project Health Big Data (HBD). This scripts use intensively libraries provided by Hugging Face (HF).

## Question Answering Bots

The first IE tool is Question Answering Bot (QABot). Here is a description of what Question Answering is: https://huggingface.co/tasks/question-answering.
In a nutshell, QABots are fed a text, called context, and are asked questions related to it. "Advanced" QABots can also understand when an answer is impossible (it is not possible to give an answer from the context).
QABot are usually extractive (for example based on BERT or RoBERTa models), but this task can be fulfilled also by generative models (for example based on GPT-2 or T5 models). The main difference is that an extractive QABot identifies answers as spans of the context.

### Extractive QABots:
One of the main tools used in these script is the HF pipeline. With that, if a context is longer than the maximum sequence length, it is split in sub-contexts; the QABot is applied to every one of them, and then the answer with the highest score is selected as the correct one. While this is very useful, it does not give full control on the process, so we will be implementing 3 working modalities:
- MANUAL (#TODO): the pipeline will be fed not the whole text but only the paragraph containing the correct answer, extracted manually. This script should provide the best possible answers and thus the best possible scores;
- TOPIC: the pipeline will be fed not the whole text but only the paragraph containing the correct answer, extracted by a topic detection algorithm. Since automatic topic detection is expected to behave worse than manual one, this script should provide worse results than MANUAL mode;
- PIPELINE: the pipeline will be fed the whole text.

### Metrics:
The metrics used to evaluate QABOt are the following:
- Exact Match score (EM): compares the predicted answer with reference. If there's a perfect match, the answer is given a score of 1, 0 in every other case. The total EM is the sum of scores divided by the number of answers;
- F1 score (F1): calculated as the harmonic mean of Precision and Recall, obtained in the classical way starting from True Positives (number of tokens shared between predicted answer and reference), False Positives (number of tokens in predicted answer but not in reference), and False Negatives (number of tokens in the reference answer but not in predicted). For more info, see: https://en.wikipedia.org/wiki/F-score
- BLEU score (Bi-Lingual Evaluation Understudy): For more info, see: https://en.wikipedia.org/wiki/BLEU and https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86. 

Metrics are calculated globally, for single questions, and for single documents.

## How to

### Main:
The most important script is Main.ipynb, which contains the implemented pipeline. Note that every results related to a code run will be saved in the folder 'Results'; here, a subfolder will be created at every run, named as the date of the run in format YYYYmmDD_HHMMSS, so that different runs with same parameters will not overwrite previous results.
First of all, user will have to specify "experiment_name", that will be written in the session_info file (it will contain results).
Then, user will have to specify the following parameters:
- load_pregenerated_answers: since extracting answers could take some time, it is possible to store and load them in a second moment to analyze performance.
		- If this parameters is set to 'False', answers will be extracted and then stored in a file called 'pred_answers.json';
		- If this parameters is set to 'True', user will have to specify the path of file with answers, and answers will be loaded from it.
- my_path: the folder that contains files annotated with Haystack annotations tool for QA;
- model_checkpoint: the name (if loaded from HF remote repository) or path (if loaded from local file) of the model that will be used to get answers. If pre-generated answers are loaded, the tokenizer of the model will still be used to calculate statistical parameters;
- topic_model_checkpoint: the name (if loaded from HF remote repository) or path (if loaded from local file) of the model that will be used to assign topic labels to document splits;
- candidate_labels: candidate labels for topic detection.

The outputs of Main will be:
- session_info.json: this fill will contain general information about the run and results from the QABot pipeline, divided by general (calculated on all document and all questions), by question, and by document;
- QUESTIONS_heat.png: heatmap with results for single questions;
- DOCUMENTS_heat.png: heatmap with results for single documents.

### Custom libraries scripts:
- my_utility.py: custom library, contains several functions used by Main;
- my_utility.ipynb: custom library in jupyter notebook, used to edit in an easy way;

### Additional scripts (located in folder "Utility_notebooks"):
- Convert_notebook_to_py.ipynb: script used to convert 'my_utility' from ntoebook to python file. Edit 'my_utility' as you like, then run this script;
- Split_Annotations.ipynb: split a single file with all annotations (as it comes downloaded from Haystack) to several files;
- SQuAD_to_BioASQ.ipynb: used to convert from SQuAD to BioASQ format;

### "Spikes" folder - contains code used to test:
- Split_Paragraph.ipynb: Spike to test paragraph splitting