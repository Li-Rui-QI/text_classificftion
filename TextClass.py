import csv
import io
import string


import numpy
import pandas
import pandas as pd
import textblob
from tensorflow.keras import layers, models, optimizers
from keras.preprocessing import text, sequence
from sklearn import model_selection, metrics, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def Word_Count_Vectors(Train_x, Valid_x):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])
    # 使用向量计数器对象转换训练集和验证集
    xtrain_count = count_vect.transform(Train_x)
    xvalid_count = count_vect.transform(Valid_x)
    return xtrain_count, xvalid_count


def tf_idf(train_x, valid_x):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)
    return xtrain_tfidf_ngram, xvalid_tfidf_ngram


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        for i in range(predictions.shape[0]):
            if (predictions[i, 0] >= 0.5):
                predictions[i, 0] = 1
        else:
            predictions[i, 0] = 0

    return metrics.accuracy_score(predictions, valid_y)


def Word_Embeddings_Create_cnn(Train_x, Valid_x):
    embeddings_index = {}
    fin = io.open('wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embeddings_index[tokens[0]] = numpy.asarray(tokens[1:], dtype='float32')

    token = text.Tokenizer()
    token.fit_on_texts(trainDF['text'])
    word_index = token.word_index

    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(Train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(Valid_x), maxlen=70)
    # 创建分词嵌入映射
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model

    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return model, train_seq_x, valid_seq_x


if __name__ == '__main__':
    data = open('corpus').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split(" ", 1)
        labels.append(content[0])
        texts.append(content[1])

    # 创建一个dataframe，列名为text和label
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    # Divide the data set into a training set and a validation set
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

    # label target variable
    # encoder = preprocessing.LabelEncoder()
    # train_y = encoder.fit_transform(train_y)
    # valid_y = encoder.fit_transform(valid_y)
    # tmp = tf_idf(train_x, valid_x)
    # accuracy = train_model(svm.SVC(), tmp[0], train_y, tmp[1])
    # print("SVM, N-Gram Vectors: ", accuracy)

    #
    classifier = Word_Embeddings_Create_cnn(train_x, valid_x)
    accuracy = train_model(classifier[0], classifier[1], train_y, classifier[2], is_neural_net=True)
    print("CNN, Word Embeddings", accuracy)
