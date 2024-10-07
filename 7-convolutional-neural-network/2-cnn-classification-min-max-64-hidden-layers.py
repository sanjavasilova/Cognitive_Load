from cnn_classification import *

cnn_classification_using_hidden_layer([len(feature_columns_empatica), 64, 32, 2],
                                      [len(feature_columns_samsung), 64, 32, 2], "min-max")
