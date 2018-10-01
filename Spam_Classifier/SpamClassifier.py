
# coding: utf-8

# In[1]:


import nltk
import pandas as pd, numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
import re


df = pd.DataFrame(columns=['Label','Message'])

with open("SMSSpamCollection") as SMS_array:
    for num,line in enumerate(SMS_array):
        fields = line.split("\t")
        df.at[num,'Message']=fields[1]
        if fields[0]== 'ham':
            df.at[num,'Label']= 0
        else :
            df.at[num,'Label']=1               


# In[2]:


#Preprocessing

df['Message'] = df['Message'].apply(lambda x: x.lower()) #to lowercase
df['Message'] = df['Message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) #remove punctuations

#removing stopwords not optimal
#df['Message'] = df['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))

#stemming is not optimal
#stemmer = PorterStemmer()
#df['Message'] = df['Message'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()  ]))


# In[3]:


#split train and test data

train_data=df.sample(frac=0.8,random_state=200)
test_data=df.drop(train_data.index)

train_data.reset_index(inplace = True)
test_data.reset_index(inplace = True)

train_data=train_data.drop(['index'], axis=1)
test_data=test_data.drop(['index'], axis=1)


# In[4]:


spam_words_list = list(train_data[train_data['Label']==1]['Message'])
ham_words_list = list(train_data[train_data['Label']==0]['Message'])

#make dictionary of words with its frequency

dict_of_spam = {} #count of word given spam
dict_of_ham = {} #count of word given ham

for sentence in spam_words_list:
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in dict_of_spam:
            dict_of_spam[token] = 1.0
        else:
            dict_of_spam[token]+=1
          


# In[5]:


for sentence in ham_words_list:
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in dict_of_ham:
            dict_of_ham[token] = 1.0
        else:
            dict_of_ham[token]+=1

#calculating prior probability
count_spam_class=0
count_ham_class=0

for num,line in enumerate(train_data['Message']):
    if train_data.at[num,'Label'] == 0:
        count_ham_class+=1
    else :
        count_spam_class+=1
        
prob_spam_class = count_spam_class/(count_spam_class + count_ham_class)
prob_ham_class = count_ham_class/(count_spam_class + count_ham_class)


# In[6]:


precision= 0
recall =0
fscore =0

#preds of test_data

preds= {}

for idx in range(0, len(test_data)):
    i = test_data.iloc[idx]['Message']
    prob_spam=0
    prob_ham=0
    tokenized_words = [word_tokenize(i)]
    flat_tokenized_words= [item for sublist in tokenized_words for item in sublist]
    for j in flat_tokenized_words:
        if j not in dict_of_spam:
            dict_of_spam[j]=0
            prob_spam+=math.log((dict_of_spam[j] + 0.1)/(count_spam_class + 20000*0.1))
        else:
            prob_spam+=math.log((dict_of_spam[j] + 0.1)/(count_spam_class + 20000*0.1)) # product of prob(word|spam) in a message
        
        if j not in dict_of_ham:
            dict_of_ham[j]=0
            prob_ham+=math.log((dict_of_ham[j] + 0.1)/(count_ham_class + 20000*0.1))
        else:
            prob_ham+=math.log((dict_of_ham[j] + 0.1)/(count_ham_class + 20000*0.1)) # product of prob(word|ham) in a message
    
    prob_spam += math.log(prob_spam_class) # multiply by prior probability
    prob_ham += math.log(prob_ham_class)
    
    prob_spam = math.exp(prob_spam) # taking anti-log
    prob_ham = math.exp(prob_ham)
    
    if prob_ham > prob_spam:
        preds[idx]= 0 #ham
    else:
        preds[idx]= 1 #spam
    
        


# In[7]:


#confusion matrix

true_positive=0
true_negative=0
false_positive=0
false_negative=0

for index,i in enumerate(preds):
    if list(preds.values())[index] == test_data.at[index,'Label'] :
        if test_data.at[index, 'Label'] == 1:
            true_positive+=1
        else:
            true_negative+=1
    else :
        if list(preds.values())[index]== 0 and test_data.at[index,'Label'] == 1 :
            false_negative+=1
        else:
            false_positive+=1
            


# In[8]:


precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
fscore = 2 * (precision*recall)/(precision + recall)

accuracy= (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)

print('______________Naive_Bayes_Classifier______________')
print('Accuracy = ',accuracy)
print('Precision = ',precision)
print('Recall = ',recall)
print('F-score = ',fscore)


# In[9]:


#part b
#split train and test data

train_data=df.sample(frac=0.8,random_state=200)
test_data=df.drop(train_data.index)

train_data.reset_index(inplace = True)
test_data.reset_index(inplace = True)

train_data=train_data.drop(['index'], axis=1) 
test_data=test_data.drop(['index'], axis=1)


# In[10]:


spam_words_list = list(train_data[train_data['Label']==1]['Message'])
ham_words_list = list(train_data[train_data['Label']==0]['Message'])

#make dictionary of words with its frequency

dict_of_spam = {} #count of word given spam
dict_of_ham = {} #count of word given ham

for sentence in spam_words_list:
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in dict_of_spam:
            dict_of_spam[token] = 1.0
        else:
            dict_of_spam[token]+=1
          


# In[11]:


for sentence in ham_words_list:
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in dict_of_ham:
            dict_of_ham[token] = 1.0
        else:
            dict_of_ham[token]+=1

#calculating prior probability
count_spam_class=0
count_ham_class=0

for num,line in enumerate(train_data['Message']):
    if train_data.at[num,'Label'] == 0:
        count_ham_class+=1
    else :
        count_spam_class+=1
        
prob_spam_class = count_spam_class/(count_spam_class + count_ham_class)
prob_ham_class = count_ham_class/(count_spam_class + count_ham_class)


# In[12]:


def test_accuracy_fscore(alpha):
    #preds of test_data
    preds= {}

    for idx in range(0, len(test_data)):
        i = test_data.iloc[idx]['Message']
        prob_spam=0
        prob_ham=0
        tokenized_words = [word_tokenize(i)]
        flat_tokenized_words= [item for sublist in tokenized_words for item in sublist]
        for j in flat_tokenized_words:
            if j not in dict_of_spam:
                dict_of_spam[j]=0
                prob_spam+=math.log((dict_of_spam[j] + alpha)/(count_spam_class + 20000*alpha))
            else:
                prob_spam+=math.log((dict_of_spam[j] + alpha)/(count_spam_class + 20000*alpha)) # product of prob(word|spam) in a message
        
            if j not in dict_of_ham:
                dict_of_ham[j]=0
                prob_ham+=math.log((dict_of_ham[j] + alpha)/(count_ham_class + 20000*alpha))
            else:
                prob_ham+=math.log((dict_of_ham[j] + alpha)/(count_ham_class + 20000*alpha)) # product of prob(word|ham) in a message
    
        prob_spam += math.log(prob_spam_class) # multiply by prior probability
        prob_ham += math.log(prob_ham_class)
    
        prob_spam = math.exp(prob_spam) # taking anti-log
        prob_ham = math.exp(prob_ham)
    
        if prob_ham > prob_spam:
            preds[idx]= 0 #ham
        else:
            preds[idx]= 1 #spam

    #confusion matrix test data

    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    
    for index,i in enumerate(preds):
        if list(preds.values())[index] == test_data.at[index,'Label'] :
            if test_data.at[index, 'Label'] == 1:
                true_positive+=1
            else:
                true_negative+=1
        else :
            if list(preds.values())[index]== 0 and test_data.at[index,'Label'] == 1 :
                false_negative+=1
            else:
                false_positive+=1

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    test_fscore = 2 * (precision*recall)/(precision + recall)

    test_accuracy= (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    
    return test_accuracy, test_fscore


# In[13]:


def train_accuracy_fscore(alpha):
    #preds of test_data
    preds= {}

    for idx in range(0, len(train_data)):
        i = train_data.iloc[idx]['Message']
        prob_spam=0
        prob_ham=0
        tokenized_words = [word_tokenize(i)]
        flat_tokenized_words= [item for sublist in tokenized_words for item in sublist]
        for j in flat_tokenized_words:
            if j not in dict_of_spam:
                dict_of_spam[j]=0
                prob_spam+=math.log((dict_of_spam[j] + alpha)/(count_spam_class + 20000*alpha))
            else:
                prob_spam+=math.log((dict_of_spam[j] + alpha)/(count_spam_class + 20000*alpha)) # product of prob(word|spam) in a message
        
            if j not in dict_of_ham:
                dict_of_ham[j]=0
                prob_ham+=math.log((dict_of_ham[j] + alpha)/(count_ham_class + 20000*alpha))
            else:
                prob_ham+=math.log((dict_of_ham[j] + alpha)/(count_ham_class + 20000*alpha)) # product of prob(word|ham) in a message
    
        prob_spam += math.log(prob_spam_class) # multiply by prior probability
        prob_ham += math.log(prob_ham_class)
    
        prob_spam = math.exp(prob_spam) # taking anti-log
        prob_ham = math.exp(prob_ham)
    
        if prob_ham > prob_spam:
            preds[idx]= 0 #ham
        else:
            preds[idx]= 1 #spam

    #confusion matrix train data

    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    
    for index,i in enumerate(preds):
        if list(preds.values())[index] == train_data.at[index,'Label'] :
            if train_data.at[index, 'Label'] == 1:
                true_positive+=1
            else:
                true_negative+=1
        else :
            if list(preds.values())[index]== 0 and train_data.at[index,'Label'] == 1 :
                false_negative+=1
            else:
                false_positive+=1

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    train_fscore = 2 * (precision*recall)/(precision + recall)

    train_accuracy= (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
    
    return train_accuracy, train_fscore


# In[14]:


test_accuracy = {}
test_fscore = {}
train_accuracy = {}
train_fscore = {}

for i in range(-5, 1):
    #print(i)
    test_accuracy[i], test_fscore[i]=test_accuracy_fscore(2**i)
    train_accuracy[i], train_fscore[i]=train_accuracy_fscore(2**i)


# In[17]:


#plot for accuracy

import matplotlib.pyplot as plt

plt.plot([-5,-4, -3, -2 ,-1, 0], [train_accuracy[-5], train_accuracy[-4], train_accuracy[-3], train_accuracy[-2], train_accuracy[-1], train_accuracy[0]],'b-', label = "train_accuracy")

plt.plot([-5,-4, -3, -2 ,-1, 0], [test_accuracy[-5], test_accuracy[-4], test_accuracy[-3], test_accuracy[-2], test_accuracy[-1], test_accuracy[0]], 'g-', label = "test_accuracy")

plt.xlabel('i')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


plt.plot([-5,-4, -3, -2 ,-1, 0], [train_fscore[-5], train_fscore[-4], train_fscore[-3], train_fscore[-2], train_fscore[-1], train_fscore[0]],'b-', label= "train_fscore")

plt.plot([-5,-4, -3, -2 ,-1, 0], [test_fscore[-5], test_fscore[-4], test_fscore[-3], test_fscore[-2], test_fscore[-1], test_fscore[0]], 'y-', label = 'test_fscore')

plt.xlabel('i')
plt.ylabel('F-score')
plt.legend()
plt.show()

