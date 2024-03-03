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


