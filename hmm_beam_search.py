import nltk
from nltk.corpus import brown
from nltk.tag import hmm
import numpy as np
from mxpost import MaxentPosTagger

train_data = brown.tagged_sents()[:56000]
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
brown_news_tagged = brown.tagged_words()
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
most_freq_l = [tag for (tag, _) in tag_fd.most_common()]
possible_tags = {}

for sentence in train_data:
	for word, tag in sentence:
		if word in possible_tags:
			if tag not in possible_tags[word]:
				possible_tags[word].append(tag)
		else:
			possible_tags[word] = [tag]


def _k_best_sequence(words, sequences, k):
	prob = []
	k_best = []
	for seq in sequences:
		test_seq = list(zip(words, seq))
		prob.append(tagger.log_probability(test_seq))
	arr = np.array(prob)
	indices = arr.argsort()[-k:][::-1]
	for i in indices:
		k_best.append(sequences[i])
	return k_best

def k_best(sentence, k):
	k_best_sequence = [[]]
	words = []
	for word, tag in sentence:
		s = []
		for seq in k_best_sequence:
			if word not in possible_tags:
				possible_tags[word] = most_freq_l
			for tag_seq in possible_tags[word]:
				s.append(seq + [tag_seq])
		words.append(word)
		k_best_sequence = _k_best_sequence(words, s, k)
	return k_best_sequence

def accuracy_score(y_pred, y_true):
	if len(y_pred)!=len(y_true):
		print('Error: length of predictions and the true values is not same')
		return
	else:
		correct_predictions = 0
		total = 0
		for i in range(len(y_pred)):
			for j in range(len(y_pred[i])):
				if y_pred[i][j] == y_true[i][j]:
					correct_predictions+=1
			total+=len(y_pred[i])
		return 100*correct_predictions/total



maxent_train = brown.tagged_sents()[:500]
maxent_tagger = MaxentPosTagger()
maxent_tagger.train(maxent_train)
test_sents = brown.tagged_sents()[-1340:]

# discriminative reranking step 2 and testing
def re_rank_and_test(k):
	predictions = []
	true_values = []
	for se in test_sents:
		word_list = []
		tag_list = []
		for word, t in se:
			word_list.append(word)
			tag_list.append(t)
		score_list = []
		k_b = k_best(se, k)
		for tag in k_b:
			score_list.append(maxent_tagger.score(word_list, tag))
		scores = np.array(score_list)
		best_i = np.argmax(scores)
		best_tag_seq = k_b[best_i]
		true_values.append(tag_list)
		predictions.append(best_tag_seq)

	accuracy = accuracy_score(predictions, true_values)
	print('Accuracy for beam width', k, 'is', accuracy)

for x in range(1,6):
	re_rank_and_test(x)