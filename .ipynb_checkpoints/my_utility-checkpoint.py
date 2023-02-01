{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This module contains utility stuff, like functions used to calculate scores and so on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all necessary packages\n",
    "def import_compact():\n",
    "    '''\n",
    "    This function has no INPUTS nor OUTPUTS. It just import packages for running the notebook.\n",
    "    '''\n",
    "    # Utility libraries\n",
    "    import collections\n",
    "    from itertools import repeat\n",
    "    import json\n",
    "    import numpy as np\n",
    "    from os import listdir\n",
    "    from os.path import isfile, join\n",
    "    import pandas as pd\n",
    "    import pdfplumber\n",
    "    import statistics\n",
    "    import string\n",
    "    import time\n",
    "    import tqdm\n",
    "    \n",
    "    # Graphic libraries\n",
    "    import seaborn as sns\n",
    "    import matplotlib.pyplot as plt\n",
    "    %matplotlib inline\n",
    "    \n",
    "    # NLP libraries\n",
    "    import torch\n",
    "    import transformers\n",
    "    from transformers import pipeline, AutoTokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract data from json and organized them in a more readable way\n",
    "def get_annotations_data(file):\n",
    "    '''\n",
    "    INPUTS:\n",
    "    - file = path of the annotation file, format expected .json\n",
    "    OUTPUTS:\n",
    "    - triplets: list of dicts, every dict is the content of a file, it contains:\n",
    "        - 'context': text of document\n",
    "        - 'question_N': text of question index N\n",
    "        - 'question_N_ID': ID of question N\n",
    "        - 'answer_N': text of answer to question N\n",
    "    paragtaph_sort: \n",
    "    '''\n",
    "    counter = 0\n",
    "    try:\n",
    "        with open(file) as f:\n",
    "            whole = json.load(f)\n",
    "        triplets = []\n",
    "        for i, doc in enumerate(whole['data']):\n",
    "            to_append = {}\n",
    "            for j, par in enumerate(doc['paragraphs']):\n",
    "                # Extract paragraph\n",
    "                paragraph = par['qas']\n",
    "                # Sort paragraph by question index\n",
    "                paragraph_sort = sorted(paragraph, key=lambda d: int(d['question'][:3])) \n",
    "                to_append['context'] = par['context']\n",
    "                for k, qas in enumerate(paragraph_sort):\n",
    "                    to_append[f'question_{counter}'] = qas['question'][4:]\n",
    "                    to_append[f'question_{counter}_ID'] = qas['id']\n",
    "                    try: # Answer exists\n",
    "                        to_append[f'answer_{counter}'] = qas['answers'][0]['text']\n",
    "                    except: # Answer does not exist\n",
    "                        to_append[f'answer_{counter}'] = ''\n",
    "                    counter+=1\n",
    "            triplets.append(to_append)\n",
    "        return triplets, paragraph_sort\n",
    "    except:\n",
    "        print('Error while loading data from file!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
