from cnn_classification import *

cnn_classification_using_hidden_layer([len(feature_columns_empatica), 126, 32, 2],
                                      [len(feature_columns_samsung), 128, 32, 2], "z-score")
