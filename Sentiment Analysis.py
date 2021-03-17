import os
import numpy as np
import pandas as pd
import csv




####################
#Part 1 functions
####################

def remove_punct_special(word):
	punctuations_special_chars = ''';:'"\,<>./?@#$%^&_~()[]{}'''
	new_word=""
	for char in word:
	   if char not in punctuations_special_chars:
	       new_word = new_word + char
	return new_word

#Extracting feature vectors for a given review.
def feature_vector(given_line,positive_words,negative_words,pronouns,pos_neg):
	x1=x2=x3=x4=x5=x6=0
	
	line = given_line
	if "!" in line:
		x5 = 1
		line = given_line.replace("!","")	

	words = line.split()
	id_value = words.pop(0)
	
	word_count = len(words)

	for word in words:
		word = word.lower()
		word = remove_punct_special(word)

		if word in positive_words:
			x1 = x1 +1
		elif word in negative_words:
			x2 = x2 + 1
		elif word == "no":
			x3 = 1	
		elif word in pronouns:
			x4 = x4 + 1

	x6 = np.log(word_count)	
	final_vector = [id_value,x1,x2,x3,x4,x5,x6,pos_neg]
	return final_vector
				
#Sigmoid function to calculate class probability
def get_class_prob(raw_score):
	raw_score = raw_score[0]
	return 1/(1+np.exp((-1)*raw_score))			

####################
#Part 2 functions
####################

#Implementation of Stochastic Gradient Descent Algorithm
def get_SGD_weights(weights,training_set):
	old_weights = weights
	iter = 0

	while True:
		new_weights = get_updated_weights(old_weights,training_set)
		diff_vector = pd.DataFrame([0.001,0.001,0.001,0.001,0.001,0.001,0.001])
		boolean_vector = abs(old_weights - new_weights)<diff_vector

		if False not in boolean_vector.values:
			#All values are True, meaning all features having 'happy' weights, i.e., less than 0.01
			break
		else:
			old_weights = new_weights
			iter = iter +1
	
	return new_weights

def get_updated_weights(weights,df):
	random_row = df.sample()
	actual_class=random_row.iloc[0]['class']
	random_row = random_row.drop('class',1)
	
	predicted_value = get_predicted_value(random_row.iloc[0],weights)
	
	gradient = (predicted_value - actual_class)*random_row.to_numpy()

	gradient = gradient.transpose()
	learning_rate = 1

	new_weights = weights - learning_rate*(gradient)

	return new_weights

def get_predicted_value(random_row,weights):	
	raw_score = random_row.to_numpy().dot(weights.to_numpy())
	return get_class_prob(raw_score)
				


#################################################
#Main code for part 1 (Feature extraction) starts
#################################################


pfile = open("positive-words.txt","r")
positive_words = pfile.read()
positive_words = positive_words.split()
pfile.close()


nfile = open("negative-words.txt","r")
negative_words = nfile.read()
negative_words = negative_words.split()
nfile.close()

pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]

ptfile = open("hotelPosT-train.txt","r")
positive_train_set = ptfile.readlines()
ptfile.close()


ntfile = open("hotelNegT-train.txt","r",encoding="utf8")
negative_train_set = ntfile.readlines()
ntfile.close()

training_feature_vectors = ["ID","x1","x2","x3","x4","x5","x6","class"]

for pline in positive_train_set:
	pvector = feature_vector(pline,positive_words,negative_words,pronouns,1)
	training_feature_vectors = np.vstack([training_feature_vectors,pvector])

for nline in negative_train_set:
	nvector = feature_vector(nline,positive_words,negative_words,pronouns,0)
	training_feature_vectors = np.vstack([training_feature_vectors,nvector])


with open("sannidhi-nikhith-assgn2-part1.csv","w") as my_csv:
	writer = csv.writer(my_csv)
	writer.writerows(training_feature_vectors)
	my_csv.close()

#################################################
#Main code for part 1 (Feature extraction) ends
#################################################
#################################################
#Main code for part 2 (SGD implementation) starts
#################################################

with open("sannidhi-nikhith-assgn2-part1.csv", newline='') as my_csv:
	complete_data = list(csv.reader(my_csv))

complete_data = list(filter(None,complete_data))
complete_data.remove(['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'class'])

df = pd.DataFrame(complete_data, columns=['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'class'])

df = df.drop('ID',1)

df.insert(loc=6,column='feature_for_bias',value=1.0)

df['x1']=df['x1'].astype(float)
df['x2']=df['x2'].astype(float)
df['x3']=df['x3'].astype(float)
df['x4']=df['x4'].astype(float)
df['x5']=df['x5'].astype(float)
df['x6']=df['x6'].astype(float)
df['class']=df['class'].astype(float)

df = df.sample(frac=1) #Shuffling the data


training_set = df.sample(frac=0.8)

dev_set = df.drop(training_set.index)

weights = [0.0,0.0,0.0,0.0,0.0,0.0,1.0]
weights = pd.DataFrame(weights)

final_weights = get_SGD_weights(weights,training_set)


finalized_and_frozen_weights = [6.268941e+00, -6.193176e+00, -7.310586e-01, -1.193176e+00, 1.612843e-10, -3.111539e+00, 2.689414e-01] 
finalized_and_frozen_weights = pd.DataFrame(finalized_and_frozen_weights)

##################################################
#Code to evaluate the final weights on the dev set
##################################################

true_value_count = 0
total_count = 0

for row_id, row in dev_set.iterrows():
	
	actual_class = row['class']
	
	row = pd.DataFrame(row)
	row = row.T
	row = row.drop('class',1)
	
	predicted_class = get_predicted_value(row, final_weights)

	if predicted_class > 0.5:
		predicted_class = 1.0
	else:
		predicted_class = 0.0	

	if(predicted_class == actual_class):
		true_value_count = true_value_count + 1
	total_count = total_count +1

accuracy = true_value_count/total_count


#############################################################
#Code for predicting reviews of homework test data starts
#############################################################

pfile = open("positive-words.txt","r")
positive_words = pfile.read()
positive_words = positive_words.split()
pfile.close()

nfile = open("negative-words.txt","r")
negative_words = nfile.read()
negative_words = negative_words.split()
nfile.close()

pronouns = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]


test_file = open("HW2-testset.txt","r",encoding="utf8")
test_set = test_file.readlines()
test_file.close()

test_feature_vectors = ["ID","x1","x2","x3","x4","x5","x6","feature_for_bias"]

for tline in test_set:
	tvector = feature_vector(tline,positive_words,negative_words,pronouns,1)
	test_feature_vectors = np.vstack([test_feature_vectors,tvector])

test_true_value_count = 0
test_total_count = 0


test_feature_vectors = np.delete(test_feature_vectors, (0), axis=0)
test_feature_vectors = pd.DataFrame(test_feature_vectors, columns=['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'feature_for_bias'])

test_feature_vectors['x1']=test_feature_vectors['x1'].astype(float)
test_feature_vectors['x2']=test_feature_vectors['x2'].astype(float)
test_feature_vectors['x3']=test_feature_vectors['x3'].astype(float)
test_feature_vectors['x4']=test_feature_vectors['x4'].astype(float)
test_feature_vectors['x5']=test_feature_vectors['x5'].astype(float)
test_feature_vectors['x6']=test_feature_vectors['x6'].astype(float)
test_feature_vectors['feature_for_bias']=test_feature_vectors['feature_for_bias'].astype(float)


for row_id, row in test_feature_vectors.iterrows():
	
	ID_number = row.iloc[0]

	row = row.T
	row = row.drop('ID',axis=0)

	predicted_class = get_predicted_value(row, finalized_and_frozen_weights)

	output_class=""

	if predicted_class > 0.5:
		output_class = "POS"
	else:
		output_class = "NEG"

		
	output_string = ID_number+"\t"+output_class + "\n"
	f = open("sannidhi-nikhith-assgn2-out.txt","a")
	f.write(output_string)
	f.close()

			
#################################
#END OF CODE FOR ASSIGNMENT 2 
#################################




