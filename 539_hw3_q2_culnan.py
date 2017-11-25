

###################################################################
####################      QUESTIONS 1/2      ######################
###################################################################
#############      GREEDY L-R POS TAGGING      ####################
###################################################################

######load required modules
import numpy as np
import pickle

np.set_printoptions(threshold=np.nan)

#####give names to test and train set
train = 'PTBSmall/train.tagged'
test  = 'PTBSmall/test.tagged'

################## KEY FUNCTIONS ##################################
def connect_pos(dataset):
    """gets a list of all words, a corresponding list of all POS 
    tags, and a list of all word-pos pairs. Returns a list with 
    three elements corresponding to the previously enumerated 
    lists."""
    words    = ['sentstart']
    pos      = ['sentstart']
    w_t_pair = [('sentstart', 'sentstart')]
    with open(dataset, 'r') as datafile:
        for line in datafile:
            line = line.lower()
            line = line.strip()
            if len(line) > 1:
                wd_pos = line.split('\t')
                w_t_pair.append(tuple(wd_pos))
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
            else:
                wd_pos = ['sentbreak', 'sentbreak']     
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
                w_t_pair.append(tuple(wd_pos))
                wd_pos = ['sentstart', 'sentstart']
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
                w_t_pair.append(tuple(wd_pos))
        return [words, pos, w_t_pair]

def remove_sent_cap(dataset):
    ##takes the data and removes sentence-initial capitalization 
    sent_groups = connect_pos(dataset)
    words = sent_groups[0]
    pairs = sent_groups[2]
    for i in range(len(words)-1):
        if words[i] == 'sentstart':
            words[i+1] = words[i+1].lower()
        if 'sentstart' in pairs[i]:
            pairs[i+1] = (pairs[i+1][0].lower(), pairs[i+1][1])
    return [words, pairs]

def word_dict(dataset):
	##creates a dictionary of words and counts from the text
    word_dict = {}
    wordset = remove_sent_cap(dataset)[0]
    wordset.remove('sentbreak')
    for word in wordset:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict

def w_t_dict(dataset):
	##creates a dict of word, tag pairs and counts from the text
    w_t_dict = {}
    wt_set = remove_sent_cap(dataset)[1]
    wt_set.remove(('sentbreak', 'sentbreak'))
    for item in wt_set:
        if item not in w_t_dict:
            w_t_dict[item] = 1
        else:
            w_t_dict[item] += 1
    return w_t_dict

def tag_dict(dataset):
    ##creates a dict of tags and counts from the text
    tag_dict = {}
    tag_set   = connect_pos(dataset)[1]
    tag_set.remove('sentbreak')
    for item in tag_set:
        if item not in tag_dict:
            tag_dict[item] = 1
        else:
            tag_dict[item] += 1
    return tag_dict

def create_tagset(dataset):
    tags = tag_dict(dataset)
    tagset = []
    [tagset.append(key) for key in tags if key not in tagset]
    tagset.remove('sentbreak')
    return tagset

def create_wordset(dataset):
    words = word_dict(dataset)
    wordset = []
    [wordset.append(key) for key in words if key not in wordset]
    wordset.remove('sentbreak')
    return wordset

def sentences(dataset):
    """makes a list of all sentences in the dataset as well as a corresponding 
    one for pos tags"""
    wd_set = remove_sent_cap(dataset)[0]
    wd_set2 = []
    pos_set = connect_pos(dataset)[1]
    pos_set2 = []
    wd_set = ' '.join(wd_set)
    pos_set = ' '.join(pos_set)
    wd_set = wd_set.split('sentbreak')
    pos_set = pos_set.split('sentbreak')
    [wd_set2.append(item.strip(' ')) for item in wd_set]
    [pos_set2.append(item.strip(' ')) for item in pos_set]
    return [wd_set2, pos_set2]

def split_sentences(dataset):
    #takes sentences and makes a list of words in each
    data = sentences(dataset)
    sentenceset = data[0]
    sentenceset2 = []
    tagset = data[1]
    tagset2 = []
    for sentence in sentenceset:
        sentence = sentence.split()
        sentenceset2.append(sentence)
    for tag in tagset:
        tag = tag.split()
        tagset2.append(tag)
    return sentenceset2, tagset2

def tag_prevt_counts(dataset):
    tags = connect_pos(dataset)[1]
    for tag in tags:
        if tag == 'sentbreak':
            tags.remove(tag)
    tagset  = create_tagset(dataset)
    tagset2 = create_tagset(dataset)
    tt_counts = {}
    for i in range(len(tags)):
        current = tags[i]
        last    = tags[i-1]
        if (current, last) not in tt_counts:
            tt_counts[(current,last)] = 1
        else:
            tt_counts[(current,last)] += 1
    return tt_counts

def create_tag_prevt_matrix(dataset, t1t2_pickle='t1t2.p'):
    ###first half of training, which makes matrix of tag transitions
    tags       = connect_pos(dataset)[1]
    tt_dict    = tag_prevt_counts(dataset)
    t_dict     = tag_dict(dataset)
    for tag in tags:
        if tag == 'sentbreak':
            tags.remove(tag)
    tagset     = create_tagset(dataset)
    tagset2    = create_tagset(dataset)
    tag_matrix = np.zeros((len(tagset),len(tagset)))
    for prev_tag in tagset:
        for tag in tagset2:
            if (tag, prev_tag) in tt_dict:
                tag_matrix[tagset.index(prev_tag)][tagset2.index(tag)] = (tt_dict[(tag, prev_tag)] + 1) / (t_dict[prev_tag] + len(tagset))
            else:
                tag_matrix[tagset.index(prev_tag)][tagset2.index(tag)] = 1 / (t_dict[prev_tag] + len(tagset))
    pickle.dump(tag_matrix, open(t1t2_pickle, "wb"))
    print('tt-matrix complete!')
    return tag_matrix

def create_wd_tag_matrix(dataset, wt_pickle='wt.p'):
	###second half of training, which makes matrix of word-tag pair probabilities
    t_dict       = tag_dict(dataset)
    wt_ct_dict   = w_t_dict(dataset)
    tagset       = create_tagset(dataset)
    wordset      = create_wordset(dataset)
    k            = len(wordset)
    wt_matrix    = np.zeros((len(wordset), len(tagset)))
    for word in wordset:
        if (len(wordset) - wordset.index(word)) % 50 == 0:
            print(len(wordset) - wordset.index(word))
        for tag in tagset:
            if (word,tag) in wt_ct_dict:
                wt_matrix[wordset.index(word)][tagset.index(tag)] = (wt_ct_dict[(word,tag)] + 1) / (t_dict[tag] + k)
            else:
                wt_matrix[wordset.index(word)][tagset.index(tag)] = 1 / (t_dict[tag] + k)
    pickle.dump(wt_matrix, open(wt_pickle, "wb"))
    print('wt-matrix complete!')
    return wt_matrix.shape

# 9:24 correct
# X:XX with wrong counts for wt probs
def training(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    ##this function puts the two halves of training together to create both matrices
    create_wd_tag_matrix(dataset, wt_pickle)
    create_tag_prevt_matrix(dataset, t1t2_pickle)
    return 'Done! Your pickles have been created'

#3:47 greedy 
def greedy_test(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    """This function takes care of the testing using pretrained matrices; the entire list
    of tags selected is given as an output, and this may be used with the following
    functions to get total word, sentence, and unknown word accuracies"""
    #load testing data
    sents       = split_sentences(dataset)[0]
    #get tagset and wordset from training
    tagset      = create_tagset(train)
    wordset     = create_wordset(train)
    #create tagdict
    t_dict      = tag_dict(dataset)
    k           = len(wordset)
    #create empty list for predictions
    tag_preds   = []
    unknowns    = []
    savespot    = 0
    ##this is the last thing you added to see if it would make a difference
    single_pred = []
    #load training matrices
    wt_matrix   = pickle.load(open(wt_pickle, 'rb'))
    t1t2_matrix = pickle.load(open(t1t2_pickle, 'rb'))
    #run the calculations
    for sent in sents:
        for word in sent:
            if word == 'sentstart':
                tag_preds.append('sentstart')
            else:   
                prob = 0.0
                ##this is the last thing you added (270 - 277)
                if word in wordset:
                    [single_pred.append(wt_matrix[wordset.index(word)][i] * t1t2_matrix[tagset.index(tag_preds[-1])][i]) for i in range(len(tagset))]
                elif word not in wordset:
                    [single_pred.append((1 / (t_dict[tagset[i]] + k)) * t1t2_matrix[tagset.index(tag_preds[-1])][i]) for i in range(len(tagset))]
                tag_preds.append(tagset[single_pred.index(max(single_pred))])
                single_pred = []
    return tag_preds

def word_accuracy(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    ##returns the overall word accuracy for a given dataset
    test_results = greedy_test(dataset, t1t2_pickle, wt_pickle)
    tag_preds    = test_results
    wordset      = create_wordset(train)
    data         = connect_pos(test)
    sentences    = data[0]
    sents_no_br  = []
    gold_tags    = data[1]
    unknowns     = []
    for item in gold_tags:
        if item == 'sentbreak':
            gold_tags.remove(item)
    for item in sentences:
        if item != 'sentbreak':
            sents_no_br.append(item)
    for item in sents_no_br:
        if item not in wordset:
            unknowns.append(sents_no_br.index(item))
    correct_count = 0
    total_count   = 0
    unknown_corr  = 0
    accuracy      = 0.0
    for item in unknowns:
        if tag_preds[item] == gold_tags[item]:
            unknown_corr += 1
    print(unknown_corr)
    unknown_total = len(unknowns)
    print(unknown_total)
    for i in range(len(tag_preds)):
        total_count += 1
        if tag_preds[i] == gold_tags[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    unk_acc  = unknown_corr / unknown_total
    return correct_count, total_count, accuracy, unknown_corr, unknown_total, unk_acc

def unknown_wd_acc(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    unknowns = greedy_test(dataset, t1t2_pickle, wt_pickle)[1]

#def sent_accuracy(dataset, t1t2_pickle='t1t2.p'):

###################PRACTICE/TEST COMMANDS###################################

#print(create_wd_tag_matrix(train))

print(training(train, t1t2_pickle='t1t2_q2_altered.p', wt_pickle='wt_q2_altered.p'))
print(word_accuracy(test, t1t2_pickle='t1t2_q2_altered.p', wt_pickle='wt_q2_altered.p'))

#51158, 59017, 86.6835% with 'q2' pickles
#1210, 4030, 30.02% unknown word accuracy with add_one smoothing
