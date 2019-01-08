from collections import Counter

from keras import Sequential
from keras.layers import Dense
from sklearn import svm
import math as m
import numpy as np

file = open("sms.txt")
sms_lines = file.readlines()
sms_filtered_list = []

# Will be used to eliminate the punctuation from messages
PUNCTUATION = [",", ";", ".", "'", ":", "?", "!"]


# Return type of sms and msg in a tuple, given the corresponding line of textfile
# Input: a raw sms line of form "<msg_type>    <msg>"
# Output: A tuple, (<type_of_msg>, <msg>)
def get_sms_type(sms):
    space = False
    sms = sms.replace("\n", "")
    list_sms = list(sms)
    type_of_msg = ""
    msg_start_index = -1
    for k in range(len(sms)):
        if list_sms[k] in [" ", "\t"] and not space:
            space = True
        elif space and list_sms[k] != " ":
            msg_start_index = k
            break
        else:
            type_of_msg += list_sms[k]
    msg = ""
    for k in range(msg_start_index, len(sms)):
        msg += list_sms[k]
    return type_of_msg, msg


# Return the list of all sms types
# Input: a list of sms tuples, (<type_of_msg>, <msg>)
# Output: a list which contains types in the same order as the input list
def get_sms_types(sl):
    types = []
    for sms in sl:
        types += [get_sms_type(sms)[0]]
    return types


# Return the list of all sms messages
# Input: a list of sms tuples, (<type_of_msg>, <msg>)
# Output: a list which contains messages in the same order as the input list
def get_sms_list(sl):
    return [get_sms_type(sms)[1] for sms in sl]


# Remove punctuation from a sms list
# Input: a list of messages
# Output: a list of the same messages, in the same order but without punctuation
def remove_punctuation(sl):
    for k in range(len(sl)):
        for p in PUNCTUATION:
            sl[k] = sl[k].replace(p, " ")
    return sl


# Clean spaces in messages
# Input: a list of messages
# Output: a list of messages where the multiple spaces are replaced with simple ones
def remove_useless_space(sl):
    space = False
    msg_list = []
    for k in range(len(sl)):
        tested = 0
        index = 0
        msg = ""
        sms = list(sl[k])
        while tested < len(sl[k]):
            if sms[index] == " " and space:
                del sms[index]
            elif sms[index] == " " and not space:
                index += 1
                space = True
            else:
                index += 1
                space = False
            tested += 1
        for l in sms:
            msg += l
        msg_list += [msg]
        space = False
    return msg_list


# Return a list of all words used in a list of messages
# Input: a list of messages
# Output: a list which contains all words used in the provided messages
def get_list_of_words(sl):
    wl = []
    for sms in sl:
        for word in sms.split(" "):
            wl += [word]
    return wl


# Filters words, eliminate numeric ones
# Input: a list of words
# Output: a list of sorted words, with the fake ones removed
def filter_words(wl):
    final_word_list = []
    for word in wl:
        if word not in "0123456789":
            final_word_list += [word]
    return final_word_list


# Give the features vector of a word
# Input: a message as the first variable and a dictionary as the second one
# Output: a list which correspond to a vector of features
def get_vector(msg, dic):
    list_msg = msg.split(" ")
    sort_msg = list(set(list_msg))
    index_msg = []
    spotted_words = []
    vector = [0 for _ in range(len(dic))]
    for k in range(len(sort_msg)):
        if sort_msg[k] in dic:
            index_msg += [dic.index(sort_msg[k])]
            spotted_words += [sort_msg[k]]
    for k in range(len(index_msg)):
        if index_msg[k] >= 0:
            vector[index_msg[k]] = list_msg.count(spotted_words[k])
    return vector


# Return the list of words that are presents in dictionary
# Input: a dictionary (Counter type of object)
# Output: a list which contains all words presents in dictionary
def get_word_list_from_dic(dic):
    wl = []
    for word in dic:
        wl += [word[0]]
    return wl


# Separate features from labels given a dataset
# Input: a dataset which consists of a list of tuples (<features>, <labels>)
# Output: a unique tuple with the list of the features as the first element and the list of the labels as the second one
def separate_features_from_labels(data, dic):
    features = []
    labels = []
    for d in data:
        features += [get_vector(d[1], dic)]
        labels += [0 if "ham" in d[0] else 1]
    return features, labels


# Return the average of the length of the given messages
# Input: a list of sms, tuples of (<type_of_msg>, <msg>)
# Output: an integer, the average of the length of the spams
def get_average_spam_length(data):
    average = 0
    nb_val = 0
    for d in data:
        if "ham" not in d[0]:
            average += len(d[1])
            nb_val += 1
    average /= nb_val
    return int(average)


# Return a filtered list without repetitive words (such as include, includes, included ==> include)
# Input: a list of words
# Output: a list of words with repetitive ones removed
def eliminate_repetetive(wl):
    new_wl = []
    for w in wl:
        if w not in new_wl:
            new_wl += [w]
    return new_wl

# Preparing sms_types
sms_types = get_sms_types(sms_lines)

# Preparing sms_list
sms_list = get_sms_list(sms_lines)
sms_list = remove_punctuation(sms_list)
sms_list = remove_useless_space(sms_list)

# Zipping them together to build our dataset
dataset = list(zip(sms_types, sms_list))

# Preparing word_list
word_list = get_list_of_words(sms_list)
word_list = filter_words(word_list)
word_list = eliminate_repetetive(word_list)

# Creating dictionary according to word_list
dictionary = Counter(word_list)

# Getting the most commons words in the dictionary
selected_words = 3000
most_commons = dictionary.most_common(selected_words)
most_commons_list = get_word_list_from_dic(most_commons)

# Getting train_model_data which represents 80% of the whole dataset
train_model_data = dataset[0:int(0.8 * len(sms_list))]
train_features, train_labels = separate_features_from_labels(train_model_data, most_commons_list)

# Getting test_model_data which represents 20% of the whole dataset
test_model_data = dataset[int(0.8 * len(sms_list))::]
test_features, test_labels = separate_features_from_labels(test_model_data, most_commons_list)


###############################################
##      PREDICTION USING MESSAGE LENGTH      ##
###############################################

average_spam_length = get_average_spam_length(train_model_data)
errors = 0
for i in range(len(test_features)):
    errors += 1 if (1 if len(test_features[i]) >= average_spam_length else 0) != test_labels[i] else 0

print("Using message length: ")
print("Errors: "+str(errors))
print("Total tests: "+str(len(test_features)))
print("Percentage of errors: "+str(m.floor(errors / len(test_features) * 1000) / 10)+"%")

# For readability only
print("")

####################################
##      PREDICTION USING SVM      ##
####################################

# Creating the SVM model
clf = svm.SVC(gamma="scale")

# Training the model
clf.fit(train_features, train_labels)

errors = 0
for i in range(0, len(test_features)):
    prediction = clf.predict([test_features[i]])
    errors += 1 if (0 if test_labels[i] != "ham" else 1) != prediction[0] else 0

print("Using SVM: ")
print("Errors: "+str(errors))
print("Total tests: "+str(len(test_features)))
print("Percentage of errors: "+str(m.floor(errors / len(test_features) * 1000) / 10)+"%")

# Creating the model
model = Sequential([
    Dense(50, activation='relu', input_dim=selected_words),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(1, activation='relu'),
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

tr_features = np.array(train_features)
tr_labels = train_labels

# Define the number of training tests per batch
epochs = 1000

model.fit(tr_features, tr_labels, epochs=epochs, batch_size=10)

te_features = np.array(test_features)
te_labels = test_labels
score = model.evaluate(te_features, te_labels, batch_size=10)


print("Using Dense layers: ")
print("Percentage of errors: "+str(100 - (m.floor(score[1] * 1000) / 10))+"%")

print()
print()
print()

print("############################")
print("#          RESULTS         #")
print("############################")
print()
print("SVM Model:")
print("- Accuracy: "+str(100 - (m.floor(errors / len(test_features) * 1000) / 10)) + "%")
print()
print("Dense layer network:")
print("- Accuracy: "+str(m.floor(score[1] * 1000) / 10) + "%")

