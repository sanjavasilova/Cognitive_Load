from cnn_classification import *

cnn_classification_using_hidden_layer([len(feature_columns_empatica), 16, 8, 2],
                                      [len(feature_columns_samsung), 16, 8, 2], "min-max")
