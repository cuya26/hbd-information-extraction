# This module contains utility stuff, like functions used to calculate scores and so on.

#################################
####### Import libraries ########
#################################

# Utility libraries
import json
import collections

# NLP libraries
from transformers import AutoTokenizer

#################################
########### Functions ###########
#################################

# Extract data from json and organized them in a more readable way
def get_annotations_data(file):
	'''
	This functions gets the path of a .json file, organized in Haystack annotation fashion, and prints it in a conveninent way.
	INPUT:
	- file = path of the annotation file, format expected .json
	OUTPUT:
	- #TODO
	'''
	counter = 0
	triplets, paragraph_sort = [], []
	try:
		with open(file) as f:
			whole = json.load(f)
		for i, doc in enumerate(whole['data']):
			to_append = {}
			for j, par in enumerate(doc['paragraphs']):
				# Extract paragraph
				paragraph = par['qas']
				# Sort paragraph by question index
				paragraph_sort = sorted(paragraph, key=lambda d: int(d['question'][:3])) 
				to_append['context'] = par['context']
				for k, qas in enumerate(paragraph_sort):
					to_append[f'question_{counter}'] = qas['question'][4:]
					to_append[f'question_{counter}_ID'] = qas['id']
					try: # Answer exists
						to_append[f'answer_{counter}'] = qas['answers'][0]['text']
					except: # Answer does not exist
						to_append[f'answer_{counter}'] = ''
					counter+=1
			triplets.append(to_append)
	except:
		print('Error while loading data from file!')
	return triplets, paragraph_sort

####################################################################################################################################

def get_EM_score(ref_list, pred_list, debugmode=False):
	'''
	This function gets two list of answers, and calculate the EM score between them.
	INPUT:
	- ref_list = list of reference answers
	- pred_list = list of predicted answers. The number of elements must be the same.
	OUTPUT:
	- EM_score = EM score of all answers
	- EM_score_not_empty = EM score only of not-empty answers
	- EM_score_empty = EM score only of empty answers
	'''
	
	# ALL Answers
	tot = len(ref_list)
	part = 0
	for i, ref in enumerate(ref_list):
		pred = pred_list[i]
		if ref==pred:
			part+=1
	EM_score = part/tot
	
	# ONLY not empty Answers
	ref_list_not_empty = list(filter(lambda ref_list: ref_list != '', ref_list))
	pred_list_not_empty = []
	for i, el in enumerate(ref_list):
		if el!='':
			pred_list_not_empty.append(pred_list[i])
	tot_not_empty = len(ref_list_not_empty)
	part_not_empty = 0
	for i, ref in enumerate(ref_list_not_empty):
		pred = pred_list_not_empty[i]
		if ref==pred:
			part_not_empty+=1
	EM_score_not_empty = part_not_empty/tot_not_empty
	
	#TODO test and correct following code
	# # ONLY empty Answers
	# ref_list_empty = list(filter(lambda ref_list: ref_list == '', ref_list))
	# # print(ref_list_not_empty)
	# pred_list_empty = []
	# for i, el in enumerate(ref_list):
	#	 # print(i, el)
	#	 if el=='':
	#		 pred_list_empty.append(pred_list[i])
	# tot_empty = len(ref_list_empty)
	# part_empty = 0
	# for i, ref in enumerate(ref_list_empty):
	#	 pred = pred_list_empty[i]
	#	 if ref==pred:
	#		 part_empty+=1
	# EM_score_empty = part_empty/tot_empty
	
	return EM_score, EM_score_not_empty#, EM_score_empty

####################################################################################################################################

def get_EM_score_single_questions(ref_list, pred_list, debugmode=False):
	'''
	This function gets two list of answers, and calculate the EM score on single answers.
	INPUT:
	- ref_list = list of reference answers
	- pred_list = list of predicted answers. The number of elements must be the same.
	OUTPUT:
	- EM_score_single = EM scores on single answers
	'''
	EM_score_single = []
	tot = len(ref_list)
	part = 0
	for ref, pred in zip(ref_list, pred_list):
		if ref==pred:
			EM_score_single.append(1)
		else:
			EM_score_single.append(0)

	return EM_score_single

####################################################################################################################################

def calculate_stat_param(TP, FP, FN):
	'''
	This function gets the number of True Positives, False Positives, and False Negatives, and calculates Precision, Recall, and F1-Score
	INPUT:
	- TP = number of True Positives
	- FP = number of False Positives
	- FN = number of False Negatives
	OUTPUT:
	- precision
	- recall
	- f1_score
	'''
	if (TP+FP)==0:
		precision = 'N/A'
	else:
		precision = TP/(TP+FP)
	if (TP+FN)==0:
		recall = 'N/A'
	else:
		recall = TP/(TP+FN)
	if ((TP+FP)==0) or ((TP+FN)==0) or ((precision+recall)==0):
		# f1_score = 'N/A'
		f1_score = 0
	else:
		f1_score = 2*precision*recall/(precision+recall)
	return precision, recall, f1_score

####################################################################################################################################

def get_all_quest_param(ref_list, pred_list, model_checkpoint):
	'''
	This function gets two list of answers, and calculate the number of True Positives, False Positives, and False Negatives.
	INPUT:
	- ref_list = list of reference answers
	- pred_list = list of predicted answers. The number of elements must be the same
	- model_checkpoint = NLP model, used to tokenize texts
	OUTPUT:
	- all_TP = list that contains True Positives for every single answer
	- all_FP = list that contains False Positives for every single answer
	- all_FN = list that contains False Negatives for every single answer
	'''
	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
	all_TP, all_FP, all_FN = [], [], []
	TP, FP, FN = 0, 0, 0
	for ref, pred in zip(ref_list, pred_list):
		ref_toks = tokenizer.encode(ref)[1:-1]
		pred_toks = tokenizer.encode(pred)[1:-1]
		common = collections.Counter(ref_toks) & collections.Counter(pred_toks)
		TP = sum(common.values()) #  TP for specific question is number of shared tokens
		all_TP.append(TP)
		FP =  sum((collections.Counter(pred_toks) - collections.Counter(ref_toks)).values()) #  FPs are tokens in pred but not in ref
		all_FP.append(FP)
		FN =  sum((collections.Counter(ref_toks) - collections.Counter(pred_toks)).values()) #  FNs are tokens in ref but not in pred= len(ref_toks) - len(pred_toks) #  FNs are tokens in ref but not in pred
		all_FN.append(FN)
	return all_TP, all_FP, all_FN

####################################################################################################################################

def nth_largest(position, my_list):
	'''
	This function gets the n-th maximum element from a list and the list with item above max removed
	# TODO
	'''
	for i in range(position-1):
		to_remove = max(my_list)
		my_list.remove(to_remove)
	return max(my_list), my_list

####################################################################################################################################

def get_answers(contexts, questions, num_answers, model, tokenizer, debug=False, advancedDebug=False):
	'''This function gets answers for questions.
	Inputs:
	- contexts = list of contexts
	- questions = list of question, expressed as sentences in natural language
	- num_answers = number of possible answers to show
	- model = model to perform NLP
	- tokenizer = tokenizer to perform NLP
	Outputs:
	- Answer for every question
	- Probability for every answer
	
	NOTE:
	Model is case-sensitive: "Come" is a different token from "come"
	Input of tokenizer is question + context
	[CLS] = Token that indicates the beginning of input
	[SEP] = Token that marks separation between question and context
	'''
	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
	for context in contexts:
		print("-"*90)
		print(f'Context: "{context}"')
	for i, question in enumerate(questions):
		print("-"*90)
		print("-"*90)
		inputs = my_tokenizer(question, context, padding=True, truncation=True, return_tensors="pt") # return_tensors="pt" -> the returned tensor is in PyTorch format
		# inputs["input_ids"] is a tensor containing ids of question + context
		# input_ids converts the tensor to list and extract the list itself
		input_ids = inputs["input_ids"].tolist()[0] 
		text_tokens = tokenizer.convert_ids_to_tokens(input_ids) # Very useful for debug!!!

		if debug:
			print('--Token numbers and respective word--')
			for j, id in enumerate(input_ids):
				print(f'Token #{j} num = {id}, Token #{j} text = {text_tokens[j]}')

		# Extract outputs from model
		outputs = my_model(**inputs)
		# answer_start_scores is a tensor that contains a logit for every token
		# Such logits represent the probability that the token is the START of answer
		answer_start_scores = outputs.start_logits
		# answer_end_scores is a tensor that contains a logit for every token
		# Such logits represent the probability that the token is the END of answer
		answer_end_scores = outputs.end_logits

		if advancedDebug:
			print(torch.softmax(answer_start_scores, dim=1))
			print(torch.softmax(answer_end_scores, dim=1))

		# Get probabilities of START and END
		# answer_start_list = % for every token to be START of answer
		answer_start_list = torch.softmax(answer_start_scores, dim=1).tolist()[0]
		# answer_end_list = % for every token to be END of answer
		answer_end_list = torch.softmax(answer_end_scores, dim=1).tolist()[0]

		print(f'Question: "{question}"')
		# Print the most probable n answers, where n = num_answers
		for k in range(num_answers):
			# Answer START
			(start_prob, start_list) = nth_largest(k+1, answer_start_list) 
			start_index = start_list.index(start_prob)
			# Answer END
			(end_prob, end_list) = nth_largest(k+1, answer_end_list) 
			end_index = end_list.index(end_prob)+1
			answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index]))
			print("-"*30)
			print(f'Answer #{k+1}: "{answer}"')
			print(f'Start token probability #{k+1}: {start_prob*100:.2f}')
			print(f'End token probability #{k+1}: {end_prob*100:.2f}')