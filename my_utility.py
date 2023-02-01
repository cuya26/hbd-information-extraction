def nth_largest(position, my_list):
    '''
    This function gets the n-th maximum element from a list and the list with item above max removed
    # TODO
    '''
    for i in range(position-1):
        to_remove = max(my_list)
        my_list.remove(to_remove)
    return max(my_list), my_list

def remove_elements(my_list, indices):
    '''
    This function removes the elements in indices from my_list
    INPUTS:
    - my_list: list with elements to be removed
    - indices: indeices of elements to remove
    OUTPUTS:
    - my_list: list cleaned
    '''
    for i in sorted(indices, reverse=True):
        del my_list[i]
    return my_list

# Load annotation files
def get_annotations_file(basepath, file_format='json', sort=True):
    from os import listdir
    from os.path import join
    '''
    INPUTS:
    - basepath: path containing the annotation files
    - fmt: extention of files (default value: json)
    - sort: boolean, specify if sorting files (default value: True)
    OUTPUTS:
    - all_annotations: list of path of annotation files
    '''
    files = []
    for file in listdir(basepath):
        if file.endswith(file_format):
            files.append(join(basepath, file))
            
    if sort:
        all_annotations = sorted(files)
    else:
        all_annotations = files
    return all_annotations

# Extract data from json and organized them in a more readable way
def get_annotations_data(all_files):
    import json
    '''
    INPUTS:
    - file = path of the annotation file, format expected .json
    OUTPUTS:
    - triplets: list of dicts, every dict is the content of a file, it contains:
        - 'context': text of document
        - 'question_N': text of question index N
        - 'question_N_ID': ID of question N
        - 'answer_N': text of answer to question N
    - paragraph_sort: 
    - all_questions: list of questions
    '''
    counter = 0
    try:
        triplets = []
        for file in all_files:
            
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
        # Get questions
        all_questions = []
        counter = 0
        for i, (k, v) in enumerate(triplets[0].items()):
            if (k.startswith('question') and not k.endswith('ID')):
                all_questions.append((counter, v))
                counter+=1
        return triplets, paragraph_sort, all_questions
    except:
        print('Error while loading data from file!')

# Load annotation files
def setup_pipeline(model_checkpoint, max_ans_len, handle_impossible_ans):
    from transformers import pipeline, AutoTokenizer
    '''
    This function gets the model checkpoint and setup the pipeline
    INPUTS:
    - model_checkpoint: path to the model to use
    - max_ans_len: max length in token of the answer
    - handle_impossible_ans: True for SQuAD 2.0+ models
    OUTPUTS:
    - nlp_pipeline: NLP pipeline for the specified downstream task
    '''
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # Setup QA pipeline
    nlp_pipeline = pipeline(
        'question-answering',
        model=model_checkpoint,
        tokenizer=model_checkpoint,
        max_answer_len = max_ans_len,
        handle_impossible_answer = handle_impossible_ans,
    )
    return nlp_pipeline

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
    #     # print(i, el)
    #     if el=='':
    #         pred_list_empty.append(pred_list[i])
    # tot_empty = len(ref_list_empty)
    # part_empty = 0
    # for i, ref in enumerate(ref_list_empty):
    #     pred = pred_list_empty[i]
    #     if ref==pred:
    #         part_empty+=1
    # EM_score_empty = part_empty/tot_empty
    
    return EM_score, EM_score_not_empty#, EM_score_empty

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
#         precision = 'N/A'
        precision = 0
    else:
        precision = TP/(TP+FP)
    if (TP+FN)==0:
#         recall = 'N/A'
        recall = 0
    else:
        recall = TP/(TP+FN)
    if ((TP+FP)==0) or ((TP+FN)==0) or ((precision+recall)==0):
        # f1_score = 'N/A'
        f1_score = 0
    else:
        f1_score = 2*precision*recall/(precision+recall)
    return precision, recall, f1_score

def get_all_quest_param(ref_list, pred_list, model_checkpoint):
    import collections
    from transformers import AutoTokenizer
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
        ref_toks = tokenizer.encode(ref.strip('\n'))[1:-1]
        pred_toks = tokenizer.encode(pred.strip('\n'))[1:-1]
        common = collections.Counter(ref_toks) & collections.Counter(pred_toks)
        TP = sum(common.values()) #  TP for specific question is number of shared tokens
        all_TP.append(TP)
        FP =  sum((collections.Counter(pred_toks) - collections.Counter(ref_toks)).values()) #  FPs are tokens in pred but not in ref
        all_FP.append(FP)
        FN =  sum((collections.Counter(ref_toks) - collections.Counter(pred_toks)).values()) #  FNs are tokens in ref but not in pred= len(ref_toks) - len(pred_toks) #  FNs are tokens in ref but not in pred
        all_FN.append(FN)
    return all_TP, all_FP, all_FN

# Elaborate results by single question
def elaborate_results_global(pred_ans_for_stats, correct_ans_for_stats, all_annotations, my_questions, pred_answers, model_checkpoint, show_answers=True):
    from itertools import repeat
    import statistics
    import string
    '''
    This function get as input annotation files, questions, predicted answers and the model checkpoint, and
    calculates the global statistical results.
    INPUTS:
    - pred_ans_for_stats: list of predicted answers
    - correct_ans_for_stats: list of reference answers
    - all_annotations: paths of the files with data
    - my_questions: list of questions the model has answered
    - model_checkpoint: model used to get answers, it is used for tokeninzing answers an thus calculate TPs, FP, and FNs
    - show_answers: boolean, if True prints question VS reference answer VS predicted answer (default value: True)
    OUTPUTS:
    - EM_TOTAL: EM score calculated on every question
    - EM_NOT_EMPTY_TOTAL: EM score calculated only on not empty questions
    - P_TOTAL: Precision calculated on every question
    - R_TOTAL: Recall calculated on every question
    - F1_MACRO_TOTAL: F1 score calculated as the average of every single F1 scores
    - F1_WEIGHTED_TOTAL: Precision calculated starting from TPs, FPs, and FNs of every single question
    '''
    # Create variables
    EM_scores = [[] for i in repeat(None, len(all_annotations))] # List of EM scores
    EM_score_not_emptys = [[] for i in repeat(None, len(all_annotations))] # List of EM scores only for NOT empty answers
    EM_score_emptys = [[] for i in repeat(None, len(all_annotations))] # List of EM scores only for empty answers
    all_TP_global = [[] for i in repeat(None, len(all_annotations))] # List of all TPs for every document
    all_FP_global = [[] for i in repeat(None, len(all_annotations))] # List of all FPs for every document
    all_FN_global = [[] for i in repeat(None, len(all_annotations))] # List of all FNs for every document
                
    # Compare correct answer with predicted one
    if show_answers:
        for i, (ref_list, pred_list) in enumerate(zip(correct_ans_for_stats, pred_ans_for_stats)):
            print('#'*100)
            print(f'Doc {all_annotations[i].split("/")[-1]}')
            for j, (ref, pred) in enumerate(zip(ref_list, pred_list)):
                print('-'*50)
                print(f'Question {my_questions[j][0]}: {my_questions[j][1]}')
                print(f'Reference answer = "{ref}"')
                print(f'Predicted answer = "{pred}"')
                  
    # Calculate EM Scores
    for i, (correct_answers, pred_answers) in enumerate(zip(correct_ans_for_stats, pred_ans_for_stats)):
        EM_scores[i], EM_score_not_emptys[i] = get_EM_score(correct_answers, pred_answers)
    
    # Calculate FPs, FNs, TPs
    for i, (correct_answers, pred_answers) in enumerate(zip(correct_ans_for_stats, pred_ans_for_stats)):
        all_TP_global[i], all_FP_global[i], all_FN_global[i] = get_all_quest_param(correct_answers, pred_answers, model_checkpoint)
    F1_global = [[] for i in repeat(None, len(all_annotations))]
    for i, (doc_TP, doc_FP, doc_FN) in enumerate(zip(all_TP_global, all_FP_global, all_FN_global)):
        temp_FP = []
        for j, (TP, FP, FN) in enumerate(zip(doc_TP, doc_FP, doc_FN)):
            P, R, F1 = calculate_stat_param(TP, FP, FN)
            temp_FP.append(F1)
            F1_global[i] = temp_FP

    # Aggregated results
    EM_TOTAL = statistics.mean(EM_scores)
    EM_NOT_EMPTY_TOTAL = statistics.mean(EM_score_not_emptys)
    full_TP = sum([sum(i) for i in zip(*all_TP_global)])
    full_FP = sum([sum(i) for i in zip(*all_FP_global)])
    full_FN = sum([sum(i) for i in zip(*all_FN_global)])
    F1_MACRO_TOTAL = statistics.mean([item for sublist in F1_global for item in sublist])
    P_TOTAL, R_TOTAL, F1_WEIGHTED_TOTAL = calculate_stat_param(full_TP, full_FP, full_FN)
               
    results = {
        'EM_TOTAL': EM_TOTAL,
        'EM_NOT_EMPTY_TOTAL': EM_NOT_EMPTY_TOTAL,
        'P_TOTAL': P_TOTAL,
        'R_TOTAL': R_TOTAL,
        'F1_MACRO_TOTAL': F1_MACRO_TOTAL,
        'F1_WEIGHTED_TOTAL': F1_WEIGHTED_TOTAL,
    }
                  
    return results, all_TP_global, all_FP_global, all_FN_global

# Elaborate results by single question
def elaborate_results_by_question(all_TP_global, all_FP_global, all_FN_global):
    import pandas as pd
    '''
    This function ...
    INPUTS:
    - 
    OUTPUTS:
    - 
    '''
    quest_num = len(all_TP_global[0])
    TPs = [0] * quest_num
    FPs = [0] * quest_num
    FNs = [0] * quest_num
    EMs = [0] * quest_num
    EMs_not_empty = [0] * quest_num
    not_empty = [0] * quest_num
    
    for i, stats in enumerate(zip(all_TP_global, all_FP_global, all_FN_global)):
        tp, fp, fn = stats[0], stats[1], stats[2]
        for j in range(quest_num):
            TPs[j]+=tp[j]
            FPs[j]+=fp[j]
            FNs[j]+=fn[j]
            if ((fp[j]==0) & (fn[j]==0)):
                EMs[j]+=1 
            # EM for not empty questions only
            if ((tp[j]+fp[j]+fn[j])!=0):
                not_empty[j]+=1
                if ((fp[j]==0) & (fn[j]==0)):
                    EMs_not_empty[j]+=1 
            
    index = []
    for i in range(quest_num):
        index.append(f'Q_{i}')        
    
    df_questions = pd.DataFrame(
        index=index,
        columns=['Precision', 'Recall', 'F1-score', 'EM', 'EM_not_empty']
    )
    counter = 0
    for i in range(quest_num):
        P, R, F1 = calculate_stat_param(TPs[counter], FPs[counter], FNs[counter])
        try:
            df_questions.loc[f'Q_{i}', 'Precision']=f'{P*100:.2f}'
        except:
            df_questions.loc[f'Q_{i}', 'Precision']='0'
        try:
            df_questions.loc[f'Q_{i}', 'Recall']=f'{R*100:.2f}'
        except:
            df_questions.loc[f'Q_{i}', 'Recall']='0'
        try:
            df_questions.loc[f'Q_{i}', 'F1-score']=f'{F1*100:.2f}'
        except:
            df_questions.loc[f'Q_{i}', 'F1-score']='0'
        try:
            em = EMs[counter]/len(all_TP_global)
            df_questions.loc[f'Q_{i}', 'EM']=f'{em*100:.2f}'
        except:
            pass
        try:
            em_ne = EMs_not_empty[counter]/not_empty[counter]
            df_questions.loc[f'Q_{i}', 'EM_not_empty']=f'{em_ne*100:.2f}'
        except:
            pass
        counter+=1
    df_questions = df_questions.apply(pd.to_numeric)
    
    return df_questions

# Elaborate results by single document
def elaborate_results_by_document(all_TP_global, all_FP_global, all_FN_global):
    import pandas as pd
    import statistics
    '''
    This function ...
    INPUTS:
    - 
    OUTPUTS:
    - 
    '''
    index = []
    for i in range(len(all_TP_global)):
        index.append(f'Doc_{i}')        

    df_documents = pd.DataFrame(
        index=index,
        columns=['Precision', 'Recall', 'F1-score', 'EM', 'EM_not_empty']
    )
                                  
    for i, stats in enumerate(zip(all_TP_global, all_FP_global, all_FN_global)):
        TPs, FPs, FNs = stats[0], stats[1], stats[2]
        P_list, R_list, F1_list, EM_list, EM_NE_list, not_empty = [], [], [], [], [], []
        # Get P, R, F1, EM for every question
        for tp, fp, fn in zip(TPs, FPs, FNs):
            P, R, F1 = calculate_stat_param(tp, fp, fn)
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)
            if ((fp==0) & (fn==0)):
                EM_list.append(1)
            else:
                EM_list.append(0)
            # EM for not empty questions only
            if ((tp+fp+fn)!=0):
                if ((fp==0) & (fn==0)):
                    EM_NE_list.append(1)
                else:
                    EM_NE_list.append(0)
        # Get P, R, F1 for every document by averaging single question ones
        P_avg = statistics.mean(P_list)
        R_avg = statistics.mean(R_list)
        F1_avg = statistics.mean(F1_list)
        EM_avg = statistics.mean(EM_list)
        EM_NE_avg = statistics.mean(EM_NE_list)
        df_documents.loc[f'Doc_{i}', 'Precision'] = f'{P_avg*100:.2f}'
        df_documents.loc[f'Doc_{i}', 'Recall'] = f'{R_avg*100:.2f}'
        df_documents.loc[f'Doc_{i}', 'F1-score'] = f'{F1_avg*100:.2f}'
        df_documents.loc[f'Doc_{i}', 'EM'] = f'{EM_avg*100:.2f}'
        df_documents.loc[f'Doc_{i}', 'EM_not_empty'] = f'{EM_NE_avg*100:.2f}'

    df_documents = df_documents.apply(pd.to_numeric)
                                  
    return df_documents

# Split text in topics

# # Define pipeline to check topic
# classifier = pipeline(
#     "zero-shot-classification",
#     model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# )
# candidate_labels = [
#     'dati sociodemografici',
#     'storia clinica',
#     'fattori di rischio',
#     'comorbidità somatica',
#     'esame obiettivo',
#     'farmaci',
# ]

# # Split in topics before answering

# for j, file in enumerate(all_annotations):
#     data, paragraph = utility.get_annotations_data(file)
#     for doc in data:
#         # # Split text in chunks
#         # text = doc['context']
#         # split_list = text.split(sep='\n \n')
#         # split_list_clean = []
#         # for split in split_list:
#         #     if len(split.strip())==0:
#         #         pass
#         #     else:
#         #         split_list_clean.append(split)
#         # # Check topics
#         # split_topics = []
#         # for split in split_list_clean:
#         #     result = classifier(split, candidate_labels, multi_label=True)
#         #     split_topics.append(result['labels'][0])
#         # # Put together all split with same topic
#         # context_by_topic = {}
#         # for label in candidate_labels:
#         #     to_keep = [i for i, x in enumerate(split_topics) if x==label]
#         #     # Merge split with same topic
#         #     text = ''
#         #     for el in to_keep:
#         #         text+=split_list_clean[el]
#         #     context_by_topic[label] = text
#         for i, q in my_questions:
#             # print(q)
#             # print(doc['context'][:20])
#             # continue
#             QA_input = {
#                 'question': q,
#                 'context': doc['context'] # Process complete text, TEST03
#             }
#             pred_answers[j].append((i, nlp(QA_input)))
# end = time.time()

# def get_answers(contexts, questions, num_answers, model, tokenizer, debug=False, advancedDebug=False):
#     '''
#     This function gets answers for questions.
#     Inputs:
#     - contexts = list of contexts
#     - questions = list of question, expressed as sentences in natural language
#     - num_answers = number of possible answers to show
#     - model = model to perform NLP
#     - tokenizer = tokenizer to perform NLP
#     Outputs:
#     - Answer for every question
#     - Probability for every answer
    
#     NOTE:
#     Model is case-sensitive: "Come" is a different token from "come"
#     Input of tokenizer is question + context
#     [CLS] = Token that indicates the beginning of input
#     [SEP] = Token that marks separation between question and context
#     '''
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#     for context in contexts:
#         print("-"*90)
#         print(f'Context: "{context}"')
#     for i, question in enumerate(questions):
#         print("-"*90)
#         print("-"*90)
#         inputs = my_tokenizer(question, context, padding=True, truncation=True, return_tensors="pt") # return_tensors="pt" -> the returned tensor is in PyTorch format
#         # inputs["input_ids"] is a tensor containing ids of question + context
#         # input_ids converts the tensor to list and extract the list itself
#         input_ids = inputs["input_ids"].tolist()[0] 
#         text_tokens = tokenizer.convert_ids_to_tokens(input_ids) # Very useful for debug!!!

#         if debug:
#             print('--Token numbers and respective word--')
#             for j, id in enumerate(input_ids):
#                 print(f'Token #{j} num = {id}, Token #{j} text = {text_tokens[j]}')

#         # Extract outputs from model
#         outputs = my_model(**inputs)
#         # answer_start_scores is a tensor that contains a logit for every token
#         # Such logits represent the probability that the token is the START of answer
#         answer_start_scores = outputs.start_logits
#         # answer_end_scores is a tensor that contains a logit for every token
#         # Such logits represent the probability that the token is the END of answer
#         answer_end_scores = outputs.end_logits

#         if advancedDebug:
#             print(torch.softmax(answer_start_scores, dim=1))
#             print(torch.softmax(answer_end_scores, dim=1))

#         # Get probabilities of START and END
#         # answer_start_list = % for every token to be START of answer
#         answer_start_list = torch.softmax(answer_start_scores, dim=1).tolist()[0]
#         # answer_end_list = % for every token to be END of answer
#         answer_end_list = torch.softmax(answer_end_scores, dim=1).tolist()[0]

#         print(f'Question: "{question}"')
#         # Print the most probable n answers, where n = num_answers
#         for k in range(num_answers):
#             # Answer START
#             (start_prob, start_list) = nth_largest(k+1, answer_start_list) 
#             start_index = start_list.index(start_prob)
#             # Answer END
#             (end_prob, end_list) = nth_largest(k+1, answer_end_list) 
#             end_index = end_list.index(end_prob)+1
#             answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index]))
#             print("-"*30)
#             print(f'Answer #{k+1}: "{answer}"')
#             print(f'Start token probability #{k+1}: {start_prob*100:.2f}')
#             print(f'End token probability #{k+1}: {end_prob*100:.2f}')

