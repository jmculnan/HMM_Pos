import numpy as np
import dynet_config
dynet_config.set(
    mem=4096,
    autobatch=True,      # utilize autobatching
    random_seed=1978     # simply for reproducibility here
)
import dynet as dy

import sys
sys.path.append("..")
import capizziutils

#####give names to test and train set
train = 'PTBSmall/train.tagged'
test  = 'PTBSmall/test.tagged'
embeddings_center = 'deps.words'
embeddings_context = 'deps.contexts'
embeddings = 'pretrained_embeddings.txt'

def format_data(dataset):
    words    = ['sentstart']
    pos      = ['sentstart']
    w_t_pair = [('sentstart', 'sentstart')]
    with open(dataset, 'r') as datafile:
        for line in datafile:
            line = line.strip()
            if len(line) > 1:
                wd_pos = line.split('\t')
                w_t_pair.append(tuple(wd_pos))
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
            else:
                wd_pos = ['sentstart', 'sentstart']
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
                w_t_pair.append(tuple(wd_pos))
        for i in range(len(w_t_pair)):
            if 'sentstart' in w_t_pair[i]:
                w_t_pair[i-1][0] = w_t_pair[i-1][0].lower()
        for pair in w_t_pair:
            pair = '/'.join(pair)
        return w_t_pair

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

def make_data(dataset):
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
                wd_pos = ['sentstart', 'sentstart']
                words.append(wd_pos[0])
                pos.append(wd_pos[1])
                w_t_pair.append(tuple(wd_pos))
        for i in range(len(words)-1):
            if words[i] == 'sentstart':
                words[i+1] = words[i+1].lower()
            if pos[i] == 'sentstart':
                pos[i+1] = pos[i+1].lower()
            if 'sentstart' in w_t_pair[i]:
                w_t_pair[i+1] = (w_t_pair[i+1][0].lower(), w_t_pair[i+1][1])
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

def create_tagset(dataset):
    tags = tag_dict(dataset)
    tagset = []
    taglabel = {}
    [tagset.append(key) for key in tags if key not in tagset]
    tagset.sort()
    tagset.remove('sentbreak')
    for i in range(len(tagset)):
    	taglabel[tagset[i]] = i
    return taglabel

def indexed_wordset(dataset):
	wordset = create_wordset(dataset)
	indexed_wordset = {}
	count = 0
	for word in wordset:
		if word not in indexed_wordset:
			indexed_wordset[word] = count
			count += 1
	return indexed_wordset

def words2indexes(seq_of_words, w2i_lookup):
    """
    This function converts our sentence into a sequence of indexes that correspond to the rows in our embedding matrix
    :param seq_of_words: the document as a <list> of words
    :param w2i_lookup: the lookup table of {word:index} that we built earlier
    """
    seq_of_idxs = []
    for w in seq_of_words:
        w = w.lower()            # lowercase
        i = w2i_lookup.get(w, 0) # we use the .get() method to allow for default return value if the word is not found
                                 # we've reserved the 0th row of embedding matrix for out-of-vocabulary words
        seq_of_idxs.append(i)
    return seq_of_idxs

labeled_pos = create_tagset(train)
reverse_labels = dict((value, key) for key, value in labeled_pos.items())

###from Mike Capizzi's tutorial
train_sents = split_sentences(train)[0]
train_labels = [[labeled_pos[label] for label in sent] for sent in split_sentences(train)[1]]

test_sents = split_sentences(test)[0]
test_labels = [[labeled_pos[label] for label in sent] for sent in split_sentences(test)[1]]

#building the model
RNN_model = dy.ParameterCollection()

#hyperparameter settings
embedding_size = 300 #size from pretrained embeddings
hidden_size    = 200
num_layers     = 1

###from Mike Capizzi's tutorial
emb_matrix_pretrained, w2i_pretrained = capizziutils.load_pretrained_embeddings(embeddings)
embedding_dimensions = emb_matrix_pretrained.shape[1]
embedding_params = RNN_model.lookup_parameters_from_numpy(emb_matrix_pretrained)
word_indices = w2i_pretrained

RNN = dy.GRUBuilder(num_layers, embedding_dimensions, hidden_size, RNN_model)

#add projection layer (to convert hidden size to size of tagset)
###from Mike Capizzi's tutorial
pW = RNN_model.add_parameters((hidden_size, len(list(labeled_pos.keys()))))
pb = RNN_model.add_parameters(len(list(labeled_pos.keys())))

def convert_words(wordlist):
	idx_wordset = indexed_wordset(train)
	indices = []
	for word in wordlist:
		word = word.lower()
		num  = idx_wordset.get(word, 0)
		indices.append(num)
	return indices


###from Mike Capizzi's tutorial
def forward_pass(word_indices):
	input_sequence = [embedding_params[i] for i in word_indices]
	W = dy.parameter(pW)
	b = dy.parameter(pb)
	rnn_seq = RNN.initial_state()
	rnn_hidden_outputs = rnn_seq.transduce(input_sequence)
	rnn_outputs = [dy.transpose(W) * h + b for h in rnn_hidden_outputs]
	return rnn_outputs

###from Mike Capizzi's tutorial
def predict(output_list):
    prediction_probs = [dy.softmax(out) for out in output_list]
    prediction_probs_np = [out.npvalue() for out in prediction_probs]
    prediction_probs_idx = [np.argmax(out) for out in prediction_probs_np]
    return prediction_probs_idx

trainer = dy.SimpleSGDTrainer(m=RNN_model, learning_rate=0.01)

def word_accuracy(dataset, results):
    ##returns the overall word accuracy for a given dataset
    tag_preds    = results
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
    unknown_total = len(unknowns)
    for i in range(len(tag_preds)):
        total_count += 1
        if tag_preds[i] == gold_tags[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    unk_acc  = unknown_corr / unknown_total
    return correct_count, total_count, accuracy, unknown_corr, unknown_total, unk_acc

batch_size = 256
num_batches_training = int(np.ceil(len(split_sentences(train)[0]) / batch_size))
num_batches_testing  = int(np.ceil(len(split_sentences(test)[0]) / batch_size))

####you need to make a different data structure in order to get the list of tokens without sentbreak
####and not split up by sentences--you need to use that probably everywhere where you see split sentences
###in this question and also wherever it asks for the data

sample_sentence = ['sentstart', 'this', 'is', 'my', 'favorite', 'thing', 'to', 'do', '.']

#from capizzi's tutorial
def evaluate(nested_preds, nested_true):
    word_scores = []
    sentence_scores = []
    for i in range(len(nested_true)):
        scores = []
        pred = nested_preds[i]
        true = nested_true[i]
        for p,t in zip(pred,true):
            score = check_score(p,t)
            scores.append(score)
        sentence_scores.append(check_sentence_score(scores))
        word_scores.extend(scores)
    overall_accuracy = get_accuracy(word_scores)
    sentence_accuracy = get_accuracy(sentence_scores)
    return overall_accuracy, sentence_accuracy

def test():
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)
    all_predictions = []

    for j in range(num_batches_testing):
        # begin a clean computational graph
        dy.renew_cg()
        # build the batch
        batch_tokens = test_sents[j*batch_size:(j+1)*batch_size]
        batch_labels = test_labels[j*batch_size:(j+1)*batch_size]
        # iterate through the batch
        for k in range(len(batch_tokens)):
            # prepare input: words to indexes
            seq_of_idxs = convert_words(batch_tokens[k], word_indices)
            # make a forward pass
            preds = forward_pass(seq_of_idxs)
            label_preds = predict(preds)
            all_predictions.append(label_preds)
    return all_predictions

final_predictions = test()

overall_accuracy, sentence_accuracy = evaluate(final_predictions, test_labels)
print("overall accuracy: {}".format(overall_accuracy))
print("sentence accuracy (all tags in sentence correct): {}".format(sentence_accuracy))

################
# HYPERPARAMETER
################
num_epochs = 5

def train():

    # i = epoch index
    # j = batch index
    # k = sentence index (inside batch j)
    # l = token index (inside sentence k)

    epoch_losses = []
    overall_accuracies = []
    sentence_accuracies = []
    
    for i in range(num_epochs):
        epoch_loss = []
        for j in range(num_batches_training):
            # begin a clean computational graph
            dy.renew_cg()
            # build the batch
            batch_tokens = train_sents[j*batch_size:(j+1)*batch_size]
            batch_labels = train_labels[j*batch_size:(j+1)*batch_size]
            # iterate through the batch
            for k in range(len(batch_tokens)):
                # prepare input: words to indexes
                seq_of_idxs = words2indexes(batch_tokens[k], word_indices)
                # make a forward pass
                preds = forward_pass(seq_of_idxs)
                # calculate loss for each token in each example
                loss = [dy.pickneglogsoftmax(preds[l], batch_labels[k][l]) for l in range(len(preds))]
                # sum the loss for each token
                sent_loss = dy.esum(loss)
                # backpropogate the loss for the sentence
                sent_loss.backward()
                trainer.update()
                epoch_loss.append(sent_loss.npvalue())
            # check prediction of sample sentence
            if j % 250 == 0:
                print("epoch {}, batch {}".format(i+1, j+1))
                sample = forward_pass(words2indexes(sample_sentence, word_indices))
                predictions = [reverse_labels[p] for p in predict(sample)]
                print(list(zip(sample_sentence, predictions)))
        # record epoch loss
        epoch_losses.append(np.sum(epoch_loss))
        trainer.save('RNN_epoch' + i + '.model')
        # get accuracy on test set
        print("testing after epoch {}".format(i+1))
        epoch_predictions = test()
        epoch_overall_accuracy, epoch_sentence_accuracy = evaluate(epoch_predictions, test_labels)
        overall_accuracies.append(epoch_overall_accuracy)
        sentence_accuracies.append(epoch_sentence_accuracy)
        
    return epoch_losses, overall_accuracies, sentence_accuracies

losses, overall_accs, sentence_accs = train()

