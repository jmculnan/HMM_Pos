

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
            #line = line.lower()
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

def count_wt_probability(tag, word, dataset, t_dict='NONE',wt_ct_dict='NONE'):
    ##this returns prob of a word given the tag 
    tag_and_word_count   = 0
    tag_count            = 0
    wt_prob              = 0.0
    if t_dict=='NONE' and wt_ct_dict=='NONE':
        t_dict   = tag_dict(dataset)
        wt_ct_dict = w_t_dict(dataset)
    try:
        wt_prob = float(wt_ct_dict[(word,tag)] / t_dict[tag])
    except:
        wt_prob = 0.0
    return wt_prob

def add_one_wt(tag, word, dataset, t_dict='NONE',wt_ct_dict='NONE'):
    ##this returns the prob of word given tag with add-one smoothing
    ####ADDED FOR QUESTION 2
    tag_given_word_count = 0
    word_count           = 0
    ##need a k to add to denominator
    wordset              = create_wordset(train)
    k                    = len(wordset)
    k_list               = []
    wt_ct_list           = []
    wt_prob              = 0.0
    if t_dict=='NONE' and wt_ct_dict=='NONE':
        t_dict       = tag_dict(dataset)
        wt_ct_dict   = w_t_dict(dataset)
    ##calculate k using all word types that occur with the tag t
    if (word,tag) in wt_ct_dict:
        wt_prob = (wt_ct_dict[(word,tag)] + 1) / (t_dict[tag] + k)
    else:
        wt_prob = 1 / (t_dict[tag] + k)
    return wt_prob

def add_one_tags(tag, prev_tag, dataset, tags='NONE'):
    ##counts probability of a tag given a previous tag
    tag_and_prev_count      = 0
    prev_t_count            = 0
    count_prob              = 0.0
    if tags == 'NONE':
        tags = connect_pos(dataset)[1]
    for i in range(len(tags)):
        if tags[i-1] == prev_tag:
            if tags[i] == tag:
                tag_and_prev_count += 1
            prev_t_count += 1
    count_prob = (tag_and_prev_count + 1) / (prev_t_count + 46)
    return count_prob

def count_tag_probability(tag, prev_tag, dataset, tags='NONE'):
    ##counts probability of a tag given a previous tag
    tag_and_prev_count      = 0
    prev_t_count            = 0
    count_prob              = 0.0
    if tags == 'NONE':
        tags = connect_pos(dataset)[1]
    for i in range(len(tags)):
        if tags[i-1] == prev_tag:
            if tags[i] == tag:
                tag_and_prev_count += 1
            prev_t_count += 1
    count_prob = tag_and_prev_count / prev_t_count
    return count_prob

def create_tag_prevt_matrix(dataset, t1t2_pickle='t1t2.p'):
	###first half of training, which makes matrix of tag transitions
    tags       = connect_pos(dataset)[1]
    for tag in tags:
        if tag == 'sentbreak':
            tags.remove(tag)
    tagset     = create_tagset(dataset)
    tag_matrix = np.zeros((len(tagset),len(tagset)))
    for prev_tag in tagset:
        for tag in tagset:
            tag_matrix[tagset.index(prev_tag)][tagset.index(tag)] = count_tag_probability(tag,prev_tag,dataset,tags)
    pickle.dump(tag_matrix, open(t1t2_pickle, "wb"))
    return tag_matrix

def create_wd_tag_matrix(dataset, wt_pickle='wt.p'):
	###second half of training, which makes matrix of word-tag pair probabilities
    t_dict       = tag_dict(dataset)
    wt_ct_dict   = w_t_dict(dataset)
    tagset       = create_tagset(dataset)
    wordset      = create_wordset(dataset)
    wt_matrix    = np.zeros((len(wordset), len(tagset)))
    for word in wordset:
        for tag in tagset:
            wt_matrix[wordset.index(word)][tagset.index(tag)] = count_wt_probability(tag,word,dataset,t_dict,wt_ct_dict)
    pickle.dump(wt_matrix, open(wt_pickle, "wb"))
    return wt_matrix.shape

def training(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
	##this function puts the two halves of training together to create both matrices
	create_wd_tag_matrix(dataset, wt_pickle)
	create_tag_prevt_matrix(dataset, t1t2_pickle)
	return 'Done! Your pickles have been created'

def greedy_test(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    """This function takes care of the testing using pretrained matrices; the entire list
    of tags selected is given as an output, and this may be used with the following
    functions to get total word, sentence, and unknown word accuracies"""
    #load testing data
    sents       = split_sentences(dataset)[0]
    #get tagset and wordset from training
    tagset      = create_tagset(train)
    wordset     = create_wordset(train)
    #create empty list for predictions
    tag_preds   = []
    unknowns    = []
    savespot    = 0
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
                for i in range(len(tagset)):
                    ##without smoothing
                    if word in wordset:
                        if wt_matrix[wordset.index(word)][i] * t1t2_matrix[tagset.index(tag_preds[-1])][i] > prob:
                            #prob = wt_matrix[wordset.index(word)][i] * t1t2_matrix[tagset.index(tag_preds[-1])][i]
                            savespot = i
                    #for unknown words--make prediction based only on previous tag
                    else:
                        punctuation = ['.', ',','#','``',"''",':', 'sentstart']
                        if t1t2_matrix[tagset.index(tag_preds[-1])][i] > prob:
                            if tagset[i] in punctuation:
                                continue
                            else: 
                            #prob = t1t2_matrix[tagset.index(tag_preds[-1])][i]
                                savespot = i
                #if word not in wordset:
                 #   print(word, tagset[savespot])
                tag_preds.append(tagset[savespot])
    return tag_preds

def word_accuracy(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    ##returns the overall word accuracy for a given dataset
    test_results = greedy_test(dataset, t1t2_pickle, wt_pickle)
    tag_preds    = test_results
    gold_tags = connect_pos(test)[1]
    for item in gold_tags:
        if item == 'sentbreak':
            gold_tags.remove(item)
    correct_count = 0
    total_count   = 0
    accuracy      = 0.0
    for i in range(len(tag_preds)):
        total_count += 1
        if tag_preds[i] == gold_tags[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    return correct_count, total_count, accuracy

def unknown_wd_acc(dataset, t1t2_pickle='t1t2.p', wt_pickle='wt.p'):
    unknowns = greedy_test(dataset, t1t2_pickle, wt_pickle)[1]

#def sent_accuracy(dataset, t1t2_pickle='t1t2.p'):

###################PRACTICE/TEST COMMANDS###################################

#print(create_wd_tag_matrix(train))

print(training(train, t1t2_pickle='t1t2_rev.p', wt_pickle='wt_rev.p'))
print(word_accuracy(test, t1t2_pickle='t1t2_rev.p', wt_pickle='wt_rev.p'))

