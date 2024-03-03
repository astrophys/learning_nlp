### 1.1 Gender Identification
print("1.1 Gender Identification")
import nltk
from nltk.corpus import names
import random
def gender_features(word):
    return {'last_letter': word[-1]}

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
# (Pdb) train_set[0]
# ({'last_letter': 'e'}, 'female')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))

## What is most informative
classifier.show_most_informative_features(5)        # likelihood ratios


### 1.2 Choosing Right Features
print("\n\n1.2 Choosing Right Features")
def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('first/last letter classifier : {} '.format(nltk.classify.accuracy(classifier,
                                                  test_set)))

# Break up into 3 groups - train, devtest, test
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]

#
train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Test with dev-test : {}'.format(nltk.classify.accuracy(classifier, devtest_set)))

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

# Commented out for shortened stdout
#for (tag, guess, name) in sorted(errors):
#    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))

def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}

train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))


### 1.3 Document Classification
print('\n\n1.3 Document Classification')
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
#print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)


print('\n\n1.4 Part-of-Speech Tagging')
print('...skipping')
from nltk.corpus import brown
## suffix_fdist = nltk.FreqDist()
## for word in brown.words():
##     word = word.lower()
##     suffix_fdist[word[-1:]] += 1
##     suffix_fdist[word[-2:]] += 1
##     suffix_fdist[word[-3:]] += 1
## common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
## print(common_suffixes)
##
## def pos_features(word):
##     features = {}
##     for suffix in common_suffixes:
##         features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
##     return features
##
## # IMPORTANT PART HERE!!!
## tagged_words = brown.tagged_words(categories='news')
## featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
##
## size = int(len(featuresets) * 0.1)
## train_set, test_set = featuresets[size:], featuresets[:size]
##
## classifier = nltk.DecisionTreeClassifier.train(train_set)
## print(nltk.classify.accuracy(classifier, test_set))
## print(classifier.classify(pos_features('cats')))
## print(classifier.pseudocode(depth=4))


### Untested
print('\n\n1.5  Exploiting Context')
def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features

pos_features(brown.sents()[0], 8)

tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append( (pos_features(untagged_sent, i), tag) )

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


### Untested
print('\n\n1.6  Sequence Classification')





print('the end')
