# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:07:15 2018

@author: niittunen
"""

import os
import sys
import time
import string
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.utils import shuffle

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

os.chdir("D:\projects\dippatyo\scripts")
dir_path = os.getcwd()
print(dir_path)
sys.path.append('./icd9/')
from icd9 import ICD9

tree = ICD9('icd9/codes.json')
my_best = []

train = []
target = []
admissions = []
admission_dict = {}
target_dict = {}

note_book = []

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 3 and not t[0].isdigit() and "name" not in t]
 
    if stem:
        stemmer = SnowballStemmer("english")
        tokens.extend([stemmer.stem(t) for t in tokens])
 
    return tokens

def unify_adm_notes(notes, targets):
    final = {}
    for indx, note in enumerate(note_book):
        final.setdefault('data', {}).setdefault(note['adm'], []).append(notes[indx])
        final.setdefault('targets', {}).setdefault(note['adm'], []).append(targets[indx])
    return final

def fill_note_rows(adm_notes):
    max_row_len = 0
    for nts in admission_dict.values():
        if max_row_len < len(nts):
            max_row_len = len(nts)
    #for notes in adm_notes:
        #for x in range(len(notes), max_row_len):
            #notes.append()
            
def parse_pre_admsission_data(text):
    list1 = text.lower().split("history of present illness:")
    result = False
    if len(list1) > 1:
        list2 = list1[1].split("hospital course:")
        if len(list2) == 2:
            result = list2[0]
        #else:
        #    list2 = list1[1].split("hospital course:")
        #    if len(list2) == 2:
        #        result = list2[0]
    return result

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def format_icd9_code(raw_icd9):
    result = None
    if '.' not in raw_icd9:
        if raw_icd9.startswith('E') and len(raw_icd9) > 4:
            result = raw_icd9[:4] #+ '.' + raw_icd9[4:]
        elif len(raw_icd9) > 3:
            result = raw_icd9[:3] #+ '.' + raw_icd9[3:]
        
    return result

def parent_icd9_code(raw_icd9s):
    result = []
    for icd9 in raw_icd9s:
        res = format_icd9_code(icd9)
        if res != None:
            result.append(res)
    return result

def del_indices(array, del_indices):
    for index in sorted(del_indices, reverse=True):
        del array[index]
        

print("Phase 1 complete")

#%%
ts0 = time.time()

nrows = 1000000
chunksize = 10000
counter = 1
end_found = False
result = pd.DataFrame()
adm_result = {}
for df in pd.read_csv('../mimic_data/NOTEEVENTS.csv', chunksize=chunksize, nrows=nrows, header=None):
    print("Handled {0} of {1} note chunks".format(counter, int(nrows/chunksize)))
    
    for index, row in df.iterrows():
        if row[0] == "ROW_ID":
            continue
        if row[6] != "Discharge summary" or row[7] != "Report":
            end_found = True
            break
        adm_result[int(row[2])] = (row[10], row[1])
    if end_found:
        #selected = df.loc[(df[6] == "Discharge summary") & (df[7] == "Report")]
        #result = result.append(selected)
        break
    #result = result.append(df)
    counter += 1
    
print("Phase 2 complete")
print("Time elapsed: {0}s".format(time.time() - ts0))

#%%
ts0 = time.time()

main_data = {'train': {}, 'target': {}, 'group': {}}
nrows = 660000
chunksize = 60000
counter = 1
bad_rows = 0
handled_count = 0
for df1 in pd.read_csv('../mimic_data/DIAGNOSES_ICD.csv', chunksize=chunksize, nrows=nrows, header=None):
    print("Handled {0} of {1} diagnosis chunks".format(counter, int(nrows/chunksize)))
    for index, row in df1.iterrows():
        if row[0] == "ROW_ID":
            continue
        handled_count += 1
        adm_key = int(row[2])
        if adm_key in adm_result:
            parsed_row = parse_pre_admsission_data(adm_result[adm_key][0])
            if not parsed_row:
                bad_rows += 1
                continue
            main_data['train'][adm_key] = parsed_row
            main_data['target'].setdefault(adm_key, []).append(str(row[4]))
            main_data['group'][adm_key] = adm_result[adm_key][1]
    counter += 1
        
print("{0}/{1} notes were bad".format(bad_rows, handled_count))
print("Phase 3 complete")
print("Time elapsed: {0}s".format(time.time() - ts0))

#%%
ts0 = time.time()

train = []
target_p = []
group = []
label_counts = {}
label_set = set()

for adm in main_data['train'].keys():
    train.append(main_data['train'][adm])
    labels = parent_icd9_code(main_data['target'][adm])
    target_p.append(labels)
    for label in labels:
        label_counts[label] = label_counts.setdefault(label, 0) + 1
    group.append(main_data['group'][adm])

for label, counts in label_counts.items():
    if(counts > 500):
        label_set.add(label)

empties = []
        
for index, sample in enumerate(target_p):
    seen_labels = set()
    new_sample = []
    for s_label in sample:
        if "V" in s_label:
            continue
        if s_label not in seen_labels:
            seen_labels.add(s_label)
            if s_label in label_set:
                new_sample.append(s_label)
    if len(new_sample) == 0:
        empties.append(index)
    else:
        target_p[index] = new_sample

del_indices(target_p, empties)
del_indices(train, empties)
del_indices(group, empties)

train_f = np.array(train)
target_f = np.array(target_p)
group_f = np.array(group)

 
mlb = MultiLabelBinarizer()
target = mlb.fit_transform(target_f)
selected_labels = mlb.inverse_transform(np.array([np.ones(150)]))[0]

print("Phase 4 complete")
print("Time elapsed: {0}s".format(time.time() - ts0))


#%%
ts0 = time.time()

tv = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 ngram_range=(1,4),
                                 max_df=0.90,
                                 min_df=0.001,
                                 lowercase=True)
train1 = tv.fit_transform(train_f)
train1, target, group_f = shuffle(train1, target, group_f)

print("Phase 6 complete")
print("Time elapsed: {0}s".format(time.time() - ts0))

#%%
#ts0 = time.time()
#
#fs = SelectKBest(chi2, k=6000)
#train2 = fs.fit_transform(train1, target)
#
#print("Phase 7 complete")
#print("Time elapsed: {0}s".format(time.time() - ts0))


#%%
ts0 = time.time()
nbr_entries = 20000

text_clf1 = Pipeline([
#        ('tfidf_vect', TfidfVectorizer(tokenizer=process_text,
#                                 stop_words=stopwords.words('english'),
#                                 max_df=0.9,
#                                 min_df=0.05,
#                                 lowercase=True)),
        ('feature_selection', SelectKBest(chi2, k=6000)),
        ('clf', OneVsRestClassifier(SGDClassifier()))])

gkf = list(GroupKFold(n_splits=10).split(train1[:nbr_entries], target[:nbr_entries], group_f[:nbr_entries]))
params = [
            {
                  'clf': [OneVsRestClassifier(SGDClassifier(penalty='elasticnet', max_iter=1000, class_weight='balanced'))],
                  #'clf__estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 
                  'clf__estimator__alpha': [0.000085],
                  'clf__estimator__tol': [0.00001],
                  'feature_selection__k': [11000]
            },
#            {
#                   'clf': [RandomForestClassifier(class_weight='balanced_subsample')],
#                   'clf__max_depth': [8],
#                   'clf__max_features': [0.8],
#                   'clf__n_estimators' : [100],
#                   #"clf__min_samples_leaf": [1,2,3,4,5],
#                   #"clf__max_features": (4,5,6,"sqrt"),
#                   #"clf__criterion": ('gini','entropy'),
#            },
#            {
#                   'clf': [ExtraTreesClassifier(class_weight='balanced')],
#                   'clf__max_depth': [5],
#                   'clf__max_features': [0.0016],
#                   'clf__n_estimators' : [1000],
#                   #'feature_selection__k': [100, 300, 600]
#            },
#            {
#                   'clf': [AdaBoostClassifier()],
#                   #'clf__learning_rate ': [1.0],
#                   'clf__n_estimators' : [20, 50, 100],
#                   'feature_selection__k': [50, 500, 5000]
#            },
#            {
#                  'clf': [OneVsRestClassifier(Perceptron())],
#                  'clf__estimator__max_iter': [85],
#                  'clf__estimator__alpha': [0.000025]
#            },
#            {
#                  'clf': [OneVsRestClassifier(PassiveAggressiveClassifier(fit_intercept=True, max_iter=25, class_weight='balanced', C=0.056))],
##                  'clf__estimator__max_iter': [25, 100, 300],
##                  'clf__estimator__C': [0.56, 0.056, 0.0056],
#                  'feature_selection__k': [7500]
#            },
#            {
#                  'clf': [OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10, weights='distance'))],
#                  'clf__estimator__n_neighbors': [3],
#                  'clf__estimator__leaf_size': [5]
#            },
#            {
#                  'clf': [OneVsRestClassifier(MultinomialNB(alpha=0.01))],
#                  'clf__estimator__alpha': [0.0046]
#            },
#            {
#                  'clf': [OneVsRestClassifier(BernoulliNB())],
#                  'clf__estimator__alpha': [0.0171],
#                  'clf__estimator__binarize': [0.0198],
#                  'feature_selection__k': [10000]
#            },
#            {
#                  'clf': [OneVsRestClassifier(LinearSVC())],
#                  'clf__estimator__C': [25],
#                  'clf__estimator__tol': [0.0005]
#            }
]

random_search = GridSearchCV(text_clf1, param_grid=params,
                             scoring='f1_weighted', n_jobs=2, cv=gkf, verbose=10)
random_search.fit(train1[:nbr_entries], target[:nbr_entries])
report(random_search.cv_results_, 10)

print("Phase 8 complete")
print("Time elapsed: {0}s".format(time.time() - ts0))

#%%
nbr_entries = 20000
text_clf2 = Pipeline([('chi2', SelectKBest(chi2, k=50)),
                ('clf', SGDClassifier())])
gkf = list(GroupKFold(n_splits=3).split(train1[:nbr_entries], target[:nbr_entries], group_f[:nbr_entries]))
params = [
            {
                  'clf': [SGDClassifier(penalty='elasticnet', loss='modified_huber', max_iter=1000, class_weight='balanced', alpha=0.000038, tol=0.0000055)],
#                  'clf__estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 
#                  'clf__alpha': [0.00005, 0.000038, 0.000023],
#                  'clf__tol': [0.0000078, 0.0000055, 0.0000032],
            },
           {
                   'clf': [RandomForestClassifier(class_weight='balanced')],
                   'clf__max_depth': [13],
                   'clf__max_features': [0.0077],
                   'clf__n_estimators' : [1500],
                   'chi2__k': [430]
            },
            {
                   'clf': [ExtraTreesClassifier(class_weight='balanced')],
                   'clf__max_depth': [15],
                   'clf__max_features': [0.016],
                   'clf__n_estimators' : [1700],
                   'chi2__k': [500]
            },
#            {
#                  'clf': [Perceptron()],
#                  'clf__max_iter': [5, 30, 85],
#                  'clf__alpha': [0.1, 0.001, 0.000025],
#            },
#            {
#                  'clf': [PassiveAggressiveClassifier(fit_intercept=True, max_iter=50, class_weight='balanced')],
#                  'clf__max_iter': [450],
#                  'clf__C': [2.1],
#                  'clf__tol': [0.0000001],
#                  'chi2__k': [180]
#            },
#            {
#                  'clf': [KNeighborsClassifier(n_neighbors=10)],
#                  'clf__n_neighbors': [2, 3, 4],
#                  'clf__leaf_size': [5],
#            },
            {
                  'clf': [MultinomialNB(alpha=0.01)],
                  'clf__alpha': [0.1],
            },
            {
                  'clf': [BernoulliNB(alpha=0.01)],
                  'clf__alpha': [0.1],
                  'clf__binarize': [0.00168],
            },
#            {
#                  'clf': [LinearSVC(class_weight='balanced')],
#                  'clf__C': [5],
#                  'clf__tol': [0.6],
#            }
]


multitarget = np.transpose(target)
best_scores = []
score_mean = 0
for index, label in enumerate(multitarget):
    try:
        print("Building a classifier for: {0}".format(selected_labels[index]))
        random_search1 = GridSearchCV(text_clf2, param_grid=params, scoring='f1', n_jobs=4, cv=gkf, verbose=10)
        random_search1.fit(train1[:nbr_entries], label[:nbr_entries])
        report(random_search1.cv_results_, 3)
        score_mean = ((score_mean * index) + random_search1.best_score_) / (index + 1)
        print("Score mean is now: {0}".format(score_mean))
        best_scores.append((random_search1.best_score_, random_search1.best_estimator_, random_search1.best_params_))
    except:
        break
#%%
Y_pred = []
#Y_prob = []
Y_true = target[nbr_entries:nbr_entries + 5000]
for index, score in enumerate(best_scores):
    predicted = score[1].predict(train1[nbr_entries:nbr_entries + 5000])
#    probabilities = score[1].predict_proba(train1[nbr_entries:nbr_entries + 5000])
    Y_pred.append(predicted)
#    Y_prob.extend(probabilities)
    
Y_predicted = np.transpose(Y_pred)
#Y_true_b = [1 if x else 0 for x in Y_true]
#Y_pred_b = [1 if x else 0 for x in Y_pred]
print(classification_report(target[nbr_entries:nbr_entries + 5000], Y_predicted, target_names=selected_labels))

#%%

C_predicted = random_search.best_estimator_.predict(train1[nbr_entries:nbr_entries + 5000])
print(classification_report(target[nbr_entries:nbr_entries + 5000], C_predicted, target_names=selected_labels))
c_wrong = np.zeros(150)
s_wrong = np.zeros(150)
b_wrong = np.zeros(150)
for s_index, sample in enumerate(C_predicted):
    for l_index, label in enumerate(sample):
        if (label != Y_predicted[s_index][l_index]):
            if(label != Y_true[s_index][l_index]):
                c_wrong[l_index] += 1
            else:
                s_wrong[l_index] += 1
        elif(label != Y_true[s_index][l_index]):
            b_wrong[l_index] += 1
            c_wrong[l_index] += 1
            s_wrong[l_index] += 1
print("Combined got {0} wrong, and singulars got {1} wrong. Both were wrong {2} times".format(np.sum(c_wrong), np.sum(s_wrong), np.sum(b_wrong)))

#%%
def get_prob(index):
    offset = 5000 * index
    mp_count = 0
    fp_count = 0
    rp_count = 0
    for x in range(offset, offset + 5000):
        if(Y_true_b[x] or Y_pred_b[x]):
            if (Y_true_b[x] and Y_pred_b[x]):
                rp_count += 1
            if (Y_true_b[x] and not Y_pred_b[x]):
                mp_count += 1
            if (not Y_true_b[x] and Y_pred_b[x]):
                fp_count += 1
            print("True value: {0}, prediction: {1}, prediction proba: {2}. Result: {3}".format(Y_true_b[x], Y_pred_b[x], Y_prob[x], Y_true_b[x] and Y_pred_b[x]))
    print("True predictions: {0}, with {1} missed positives and {2} false positives".format(rp_count, mp_count, fp_count))

#%%

def print_curve(index):
    offset = 5000 * index
    Y_score = best_scores[index][1].decision_function(train1[nbr_entries:nbr_entries + 5000])
    print(Y_score)
    precision, recall, _ = precision_recall_curve(Y_true_b[offset:offset + 5000], Y_score)
    average_precision = average_precision_score(Y_true_b[offset:offset + 5000], Y_score)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

#%%
for tru, neg in zip(Y_true, Y_pred):
    if tru or neg:
        print((tru, neg))

#%%
for index, score in enumerate(best_scores):
    print("Score for {0}: {1}".format(selected_labels[index], score[0]))

#%%
print(np.mean([score[0] for score in best_scores]))
    
#%%
for index, score in enumerate(best_scores):
    print("Score for {0}: {1}, with {2}".format(selected_labels[index], score[0], score[2]))

#%%
import matplotlib.pyplot as plt

t = np.arange(0, 150)
true_positives = np.zeros(150)
false_positives = np.zeros(150)
false_negatives = np.zeros(150)
for si, sample in enumerate(C_predicted):
    for vi, val in enumerate(sample):
        true_positives[vi] += 1 if (val == 1 and target[nbr_entries + si][vi] == 1) else 0
        false_positives[vi] += 1 if (val == 1 and target[nbr_entries + si][vi] == 0) else 0
        false_negatives[vi] += 1 if (val == 0 and target[nbr_entries + si][vi] == 1) else 0
        
true_positives1 = np.zeros(150)
false_positives1 = np.zeros(150)
false_negatives1 = np.zeros(150)
for si, sample in enumerate(Y_predicted):
    for vi, val in enumerate(sample):
        true_positives1[vi] += 1 if (val == 1 and target[nbr_entries + si][vi] == 1) else 0
        false_positives1[vi] += 1 if (val == 1 and target[nbr_entries + si][vi] == 0) else 0
        false_negatives1[vi] += 1 if (val == 0 and target[nbr_entries + si][vi] == 1) else 0
        
precision = []
precision1 = []
c_score = []
recall = []
recall1 = []
c_score1 = []
choose_sing = []
for vi, val in enumerate(true_positives):
    precision.append(true_positives[vi] / (true_positives[vi] + false_negatives[vi]))
    precision1.append(true_positives1[vi] / (true_positives1[vi] + false_negatives1[vi]))
    recall.append(true_positives[vi] / (true_positives[vi] + false_positives[vi]))
    recall1.append(true_positives1[vi] / (true_positives1[vi] + false_positives1[vi]))
    c_score.append(precision[vi] * recall[vi])
    c_score1.append(precision1[vi] * recall1[vi])
    choose_sing.append(1 if c_score1[vi] > c_score[vi] else 0)

plt.figure(1)
#plt.subplot(211)
#plt.plot(t, c_score, 'g-', t, c_score1, 'r-')
#plt.xticks(t, selected_labels, rotation=80)
plt.subplot(311)
plt.title("OneVsRest-SGD-luokittelija", fontsize=20)
plt.plot(t, true_positives, 'g-', t, false_positives, 'r--', t, false_negatives, 'b:')
plt.xlabel('ICD-9 diagnoosikoodi')
plt.ylabel('lkm')
plt.xticks(t, selected_labels, rotation=80)

plt.subplot(312)
plt.title("Luokkakohtaiset luokittelijat", fontsize=20)
plt.plot(t, true_positives1, 'g-', label="true positives")
plt.plot(t, false_positives1, 'r--', label="false positives")
plt.plot(t, false_negatives1, 'b:', label="false negatives")
plt.xlabel('ICD-9 diagnoosikoodi')
plt.ylabel('lkm')
plt.xticks(t, selected_labels, rotation=80)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(313)
plt.title("Ratkaisumallien vertalu", fontsize=20)
predicted_t = np.transpose(C_predicted)
predicted_yt = np.transpose(Y_predicted)
target_t = np.transpose(target[nbr_entries:nbr_entries + 5000])
f1_scores = [f1_score(target_t[index], predicted_t[index], average="weighted") for index, label in enumerate(predicted_t)]
f1_scores1 = [f1_score(target_t[index], label, average="weighted") for index, label in enumerate(predicted_yt)]
choose_sing1 = [1 if score < f1_scores1[index] else 0 for index, score in enumerate(f1_scores)]
plt.plot(t, f1_scores, 'g-', label="OneVsRest-SGD-luokittelija")
plt.plot(t, f1_scores1, 'b-', label="Luokkakohtaiset luokittelijat")
plt.xlabel('ICD-9 diagnoosikoodi')
plt.ylabel('f1-arvo')
plt.xticks(t, selected_labels, rotation=260)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
#plt.tight_layout()

plt.show()

#%%
#C_predicted1 = random_search.best_estimator_.predict(train1[nbr_entries + 5000:nbr_entries + 10000])
#Y_pred1 = []
#for index, score in enumerate(best_scores):
#    predicted1 = score[1].predict(train1[nbr_entries + 5000:nbr_entries + 10000])
#    Y_pred1.append(predicted1)
#    
#Y_predicted1 = np.transpose(Y_pred1)
#O_predicted = []
#
#for s_index, sample in enumerate(C_predicted1):
#    O_predicted.append([Y_predicted1[s_index][l_index] if choose_sing1[l_index] else value for l_index, value in enumerate(sample)])
#    
#t = np.arange(0, 150)
#true_positives = np.zeros(150)
#false_positives = np.zeros(150)
#false_negatives = np.zeros(150)
#for si, sample in enumerate(C_predicted):
#    for vi, val in enumerate(sample):
#        true_positives[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        false_positives[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 0) else 0
#        false_negatives[vi] += 1 if (val == 0 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        
#true_positives1 = np.zeros(150)
#false_positives1 = np.zeros(150)
#false_negatives1 = np.zeros(150)
#for si, sample in enumerate(Y_predicted):
#    for vi, val in enumerate(sample):
#        true_positives1[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        false_positives1[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 0) else 0
#        false_negatives1[vi] += 1 if (val == 0 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        
#true_positives2 = np.zeros(150)
#false_positives2 = np.zeros(150)
#false_negatives2 = np.zeros(150)
#for si, sample in enumerate(O_predicted):
#    for vi, val in enumerate(sample):
#        true_positives2[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        false_positives2[vi] += 1 if (val == 1 and target[nbr_entries + 5000 + si][vi] == 0) else 0
#        false_negatives2[vi] += 1 if (val == 0 and target[nbr_entries + 5000 + si][vi] == 1) else 0
#        
#precision = []
#precision1 = []
#precision2 = []
#c_score = []
#c_score1 = []
#c_score2 = []
#recall = []
#recall1 = []
#recall2 = []
#
#choose_sing = []
#for vi, val in enumerate(true_positives):
#    precision.append(true_positives[vi] / (true_positives[vi] + false_negatives[vi]))
#    precision1.append(true_positives1[vi] / (true_positives1[vi] + false_negatives1[vi]))
#    precision2.append(true_positives2[vi] / (true_positives2[vi] + false_negatives2[vi]))
#    recall.append(true_positives[vi] / (true_positives[vi] + false_positives[vi]))
#    recall1.append(true_positives1[vi] / (true_positives1[vi] + false_positives1[vi]))
#    recall2.append(true_positives2[vi] / (true_positives2[vi] + false_positives2[vi]))
#    c_score.append(precision[vi] * recall[vi])
#    c_score1.append(precision1[vi] * recall1[vi])
#    c_score2.append(precision2[vi] * recall2[vi])
#    #choose_sing.append(1 if c_score1[vi] > c_score[vi] else 0)
#
#plt.figure(1)
#plt.subplot(211)
#plt.plot(t, c_score, 'g-', t, c_score1, 'r-', t, c_score2, 'b-')
#plt.xticks(t, selected_labels, rotation=80)
##plt.subplot(311)
##plt.plot(t, true_positives, 'g-', t, false_positives, 'r--', t, false_negatives, 'b:')
##plt.xticks(t, selected_labels, rotation=80)
##
##plt.subplot(312)
##plt.plot(t, true_positives1, 'g-', t, false_positives1, 'r--', t, false_negatives1, 'b:')
##plt.xticks(t, selected_labels, rotation=80)
#
#plt.subplot(212)
#predicted_t = np.transpose(C_predicted)
#predicted_yt = np.transpose(Y_predicted)
#predicted_ot = np.transpose(O_predicted)
#target_t = np.transpose(target[nbr_entries:nbr_entries + 5000])
#f1_scores = [f1_score(target_t[index], predicted_t[index], average="weighted") for index, label in enumerate(predicted_t)]
#f1_scores1 = [f1_score(target_t[index], label, average="weighted") for index, label in enumerate(predicted_yt)]
#f1_scores2 = [f1_score(target_t[index], label, average="weighted") for index, label in enumerate(predicted_ot)]
##choose_sing1 = [1 if score < f1_scores1[index] else 0 for index, score in enumerate(f1_scores)]
#plt.plot(t, f1_scores, 'g-', t, f1_scores1, 'r-', t, f1_scores2, 'b-')
#plt.xticks(t, selected_labels, rotation=80)
#
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#plt.show()


#%%
#ts0 = time.time()
#
#predicted = random_search.best_estimator_.predict(train2[nbr_entries:nbr_entries + 500])
#accuracy_data = []
##decoded_predictions = mlb.inverse_transform(predicted)
##decoded_categories = mlb.inverse_transform(target[nbr_entries:nbr_entries + 500])
#
#for prediction, category, data in zip(predicted, target[nbr_entries:nbr_entries + 500], train2[nbr_entries:nbr_entries + 500]):
#    accuracy_data.append({
#            "binary": (prediction, category),
#            #"decoded": (depreds, decategories),
#            "data": data,
#            "raw_accuracy": prediction == category,
#            "accuracy": np.sum(prediction == category)/len(prediction),
#            "correct_positives": np.sum((prediction == category) & (prediction == 1))/np.sum(prediction == 1),
#            "missed_positives": np.sum((prediction != category) & (category == 1))/np.sum(category == 1)
#        })
#
#best_results = []
#for data in accuracy_data:
#    if data["correct_positives"] > 0.7:
#        best_results.append(data)

#print(metrics.classification_report(y_test, predicted))
##metrics.confusion_matrix(y_test, predicted)
#
#print("Phase 5 complete")
#print("Time elapsed: {0}s".format(time.time() - ts0))
#print("All phases complete")