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
        #. [Google Machine Learning: Text Classification guide](https://developers.google.com/machine-learning/guides/text-classification)
#. Thoughts
    a) This feels much more feature rich than NLTK


Text and natural language processing with TensorFlow
=======================
1. [Link](https://developers.google.com/machine-learning/guides/text-classification)
#. Introduction
    a) Common applications 
        #. Sentiment analysis, e.g. 
            * Determine if Twitter posts liked Black Panther movie 
            * Walmart reviews of Nike shoes
        #. Spam vs Inbox classification
    #) This guide will help you learn 
        #. 
#. Step 1 : Gather Data
    a) Text classifier can only be as good as the dataset it is built from
        #. Use Twitter API or NYT API 
    #) Caveats
        #. If you are using public API, understand the limitations of API before
           using them
            * e.g. Rate limits
        #. More training examples you have the better
            * Model generalization
        #. Make sure numbe rof samples for every class or topic is not overly imbalanced
            * You should have a comparable number of samples in each class
        #. Make sure your samples adquately cover the \emph{space of possible inputs},
           not just common cases
#. Step 2 : Explore your data
    #) Important Metrics / Sanity check
        #. Number of samples: Total number of examples you have in the data.
        #. Number of classes: Total number of topics or categories in the data.
        #. Number of samples per class: Number of samples per class (topic/category).
           In a balanced dataset, all classes will have a similar number of samples;
           in an imbalanced dataset, the number of samples in each class will vary widely.
        #. Number of words per sample: Median number of words in one sample.
        #. Frequency distribution of words: Distribution showing the frequency
           (number of occurrences) of each word in the dataset.
        #. Distribution of sample length: Distribution showing the number of words per
           sample in the dataset.
#. Step 2.5 : Choose a Model
    a) Questions to consider
        #. How do we present the text data to an algorithm that expects numeric input?
            * Called data preprocessing and vectorization
        #. What type of model should we use?
        #. What configuration parameters should we use for our model?
    #) Naiive soln
        #. Very large array of preprocessing and model configuraiton options
        #. Try every possible option exhaustively, pruning choices through intuition
        #. Very time consuming / expensive
            * EXACTLY! This is my issue w/ NLTK
    #) Goal
        #. For a given dataset, find an algorithm that achieves maximum accuracy
           while minimizing computational time
        #. They ran 450k experiments across differnt types of problems across 12 datasets
            * E.g. sentiment analysis / topic classification
    #) Algorithm for Data Preparation and Model Building
        #. Calculate the number of samples/number of words per sample ratio.
        #. If this ratio is less than 1500, tokenize the text as n-grams and use a
           simple multi-layer perceptron (MLP) model to classify them (left branch in the
           flowchart below):
            * Split the samples into word n-grams; convert the n-grams into vectors.
            * Score the importance of the vectors and then select the top 20K using
              the scores.
            * Build an MLP model.
        #. If the ratio is greater than 1500, tokenize the text as sequences and use a
           sepCNN model to classify them (right branch in the flowchart below):
            * Split the samples into words; select the top 20K words based on their
              frequency.
            * Convert the samples into word sequence vectors.
            * If the original number of samples/number of words per sample ratio is
              less than 15K, using a fine-tuned pre-trained embedding with the sepCNN
              model will likely provide the best results.
        #. Measure the model performance with different hyperparameter values to find
           the best model configuration for the dataset.
        #. See : TextClassificationFlowchart.png
    #) Models can be classified into two broad models
        #. sequence models - i.e. using word ordering to extract information 
            * Convolutional Neural Networks
            * Recurrent Neural Networks
        #. bags (sets) of words - i.e. n-gram models
            * Logistic Regression
            * Simple Multi-layer Perceptrons
            * Gradient boosted trees
            * Support Vector Machines
    #) Based off of experiments,
       ratio of "number of samples" (S) : "number of words per sample" (W)
        #. If S/W < 1500, n-gram models work better
        

