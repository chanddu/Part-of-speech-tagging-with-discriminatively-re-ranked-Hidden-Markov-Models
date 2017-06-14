import nltk
from nltk.corpus import brown
from nltk.tag import hmm
import numpy as np

train_data = brown.tagged_sents()[:56000]
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(train_data)
print('Training ended')

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
		
predictions = []
true_values = []
for se in brown.tagged_sents()[-1340:]:
	word_list = []
	tag_list = []
	tag = []
	for word, t in se:
		word_list.append(word)
		tag_list.append(t)
	for w,tt in tagger.tag(word_list):
		tag.append(tt)
	
	true_values.append(tag_list)
	predictions.append(tag)

accuracy = accuracy_score(predictions, true_values)
print(accuracy)