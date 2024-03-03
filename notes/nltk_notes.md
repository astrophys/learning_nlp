### Reading NLTK Book
Ch 1: 
=======================
1. Reviewing NLTK...
    a) Token == word, emoji, etc
    #) Pronoun resolution - semantic role labelling

Ch 2 : 
=======================
1. https://www.nltk.org/book/ch02.html
#. Corpora in other languages 
#. Use Wordnet to navigate logical structures
    a) Is there a spanish version?
#. Brown corpus (from Brown University in 1961)
    a) 1 million words
    #) 500 sourceso
#. Reuters Corpus
    a) 10788 news documents
    #) 1.3 million words
    #) training and test sets
#. Inaugural Address Corpus
    a) Presidents
#. Table 1.2 - list of corpus
#. Spanish coprora
    a) nltk.corpus.cess_esp.words()
#. Corpora Structure
    a) Isoloated   - eg Gutenberg
    #) Categorized - eg Brown
    #) Overlapping - Reuters
    #) Temporal    - Inaugural addresses
    #) See help(nltk.corpus.reader)
#. Load your own corpora w/ PlaintextCorpusReader()
#. Conditional Frequency Distributions  


Ch5 : 
=======================
1. nltk.Text.similar('word') :
    a) Show words that show up in same context as 'word'
    #) 'distributional' similarity
    #) tends to find other words w/ similar part of speech
#. Tagging parts of speach


#. Ch6 : Learning to Classify Text
=======================
1. Intro
    a) 'ed' ending words tends to be past tense
    #) 'will' is indicative of news text
#. Goals 
    a) How can we identify particular features of language data that are salient
       for classifying it?
    #) How can we construct models of language that can be used to perform language
       processing tasks automatically?
    #) What can we learn about language from these models?
#. 1 - Supervised Classification
    a) Gender Identification
        #. Names ending with a, e and i tend to be female
        #. Names ending with k, o, r, s, t tend to be in male
    #) First step
        #. deciding what features are relevant and HOW to encode it
        #. feature extractor : e.g.
            gender_features(word){return{'last_letter' : word[-1]}}
        #. Use nltk.corpus import names, names.words('male.txt')
        #. greate list with tuples(firstname, class)
    #) Train 
        #. nltk.NaiveBayesClassifier.train(train_set)
    #) Test 
        #. nltk.classify.accuracy(classifier, test_set)
    #) nltk.classify.apply_features   ## smart with large objects, optimize memory
#. 1.2 - Choosing the right features
    a) Typically build features via trial and error...
    #) Can try 'kitchen sink' approach and whittle it down
        #. Beware of overfitting
    #) Have 3 sets :
        #. train set
        #. dev-test set (ie for error analysis)
        #. test set
    #) Pick features
        #. Run analysis
        #. Look for patterns in errors and build additional features
        #. For names, 1 letter suffix isn't sufficient
            + 2 letter suffix (e.g. -yn for female)
    #) Process
        #. Train on train set
        #. Use dev set to analyze errors 
        #. Build additional features based off dev-set errors
        #. Retrain 
        #. Use Test for overall accuracy, this minimizes overfitting.
#. 1.3 - Document Classification
    a) Build a feature extractor, get list of 2000 most frequent words in
       overall corpus
#. 1.4 - Part of Speech Tagging
    a) Use nltk.DecisionTreeClassifier
    #) Get suffixes
    #) Used tag words.
    
#. Thoughts
    a) Consider Lexicagol diversity
    #) Exclude non-unique words in English and Spanish.
    #) Look at collocations / bigrams, e.g. red wine
    #) normalize the text... lower()
    #) The syntax of passing a list with features and classes to
       nltk.NaiveBayesClassifier is really ugly and unpleasant
    #) Good book to understand the concepts, feels outdated

#. QUESTIONS :
    a) Why use nltk.DecisionTreeClassifier vs. nltk.NaiveBayesClassifier
    #) How do I use floating point features w/in the context of the NLTK library?
        #. 


