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


### scikit-learn

Working With Text Data
=======================
1. [Link](https://scikit-learn.org/1.3/tutorial/text_analytics/working_with_text_data.html#working-with-text-data)
#. Installation 
    a) pip install scikit-learn


### tensorflow

Text and natural language processing with TensorFlow
=======================
1. [Link](https://developers.google.com/machine-learning/guides/text-classification)
#. Intro
    a) Before train, need to process/pre-process the text
        #. Tokenized
        #. Vectorized
        #. Normalize
        #. Feature
    #) Two TF libs
        #. KerasNLP 
            * Contains high-level NLP lib
            * Has latest transformer based models 
            * Has lower-level tokenization utilities
            * Built on TensorFlow Text
            * Designed
            * Recommended soln for most NLP use cases
        #. TensorFlow Text
#. KerasNLP
    a) Has state-of-the-art preset weights and architectures
        #. Can use w/ out-of-the-box configuration
        #. Can easily customize components
    #) Contains end-to-end implementations of popular models (eg BERT and FNet)
    #) Tasks that can complete
        #. Machine Translation
        #. Text generation
        #. Text classfication
        #. Transformer model training
#. TensorFlow Text
    a) Lower level tools used by KerasNLP
        #. Use for working with raw text strings and documents
        #. Preprocessing tools 
    #) Tasks
        #. Apply feature-rich tokenizers to split strings, separate words and punctuation
        #. Check if token matches a specified string pattern
        #. Combine tokens into n-grams
        #. Process text w/ TF graph s.t. tokenization during training matches
           tokenization at inference
    #) Where to Start
        #. [Text processing tools for TensorFlow](https://www.tensorflow.org/text)
        #. [KerasNLP](https://keras.io/keras_nlp/)
            * [Getting Started with KerasNLP](https://keras.io/guides/keras_nlp/getting_started/)
            * [Pretraining a Transformer from scratch with KerasNLP](https://keras.io/guides/keras_nlp/transformer_pretraining/)
        #. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
            * [Basic text classification](https://www.tensorflow.org/tutorials/keras/text_classification)
            * [Text classification with TensorFlow Hub: Movie reviews](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
            * [Load Text](https://www.tensorflow.org/tutorials/load_data/text)
#. Thoughts
    a) This feels much more feature rich than NLTK


### scikit-learn





### Academic Articles
Feature extraction approaches from natural language requirements for reuse in software product lines: A systematic literature review by Bakar
=======================
1. [Link](https://www.sciencedirect.com/science/article/pii/S0164121215001004)
#. Lots of feature selection is done by hand
    a) Yuck!



17-november-2023
=======================
1. Thoughts :
    a) Where do they get 
        #. reach, nfollowers


20-december-2023
=======================
1. Installed exiftool so I could extract metadata, i.e. gps location form images
    a) Evidently FB strips that out...which is why I haven't found a photo w/ it
        #. https://webapps.stackexchange.com/a/46815/173098
    #) Same w/ Twitter
        #. https://stackoverflow.com/a/10591799
