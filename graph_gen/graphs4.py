import csv
import os

import pandas as pd
import matplotlib.pyplot as plt

input_filename = 'data_10_minmax.csv'
df = pd.read_csv(input_filename)
df.loc[df['vectorizer'] == 'Pt-BR_Word2Vec', 'vectorizer'] = 'Word2Vec'
df.loc[df['vectorizer'] == 'Pt-BR_GloVe', 'vectorizer'] = 'GloVe'
df.loc[df['vectorizer'] == 'Pt-BR_FastText', 'vectorizer'] = 'fastText'
df.loc[df['classifier'] == 'cnn', 'classifier'] = 'CNN'
df.loc[df['classifier'] == 'lstm', 'classifier'] = 'LSTM'
df.loc[df['classifier'] == 'mixed', 'classifier'] = 'CNN+LSTM'

metrics = ['acurácia', 'precisão', 'recall']
metric_columns = {
    'acurácia': ['vectorizer', 'classifier', 'dimensions', 'mean_accuracy', 'min_accuracy', 'max_accuracy'],
    'precisão': ['vectorizer', 'classifier', 'dimensions', 'mean_precision', 'min_precision', 'max_precision'],
    'recall':   ['vectorizer', 'classifier', 'dimensions', 'mean_recall', 'min_recall', 'max_recall'],
}
dimensions = [50, 100, 300, 600, 1000]
style = {
    'Word2Vec': '-',
    'GloVe': '--',
    'fastText': ':',
    'CNN': '-',
    'LSTM': '--',
    'CNN+LSTM': ':'
}
color = {
    'Word2Vec': '#0000FF',
    'GloVe': '#00FF00', 
    'fastText': '#FF0000',
    'CNN': '#0000FF',
    'LSTM': '#00FF00', 
    'CNN+LSTM': '#FF0000'
}

for key, columns in metric_columns.items():
    for vectorizer in df.vectorizer.unique():
        fig, ax = plt.subplots()
        output_filename = os.path.join('graphs', 'redone', '_'.join([vectorizer, key]))

        vec_df = df.loc[df['vectorizer']==vectorizer, columns]
        for classifier in vec_df.classifier.unique():
            class_df = vec_df[vec_df['classifier']==classifier]

#             ax.errorbar(
#                 dimensions,
#                 means,
#                 yerr=errors,
#                 label=classifier,
#                 color=color[classifier],
#                 linewidth=4,
#                 linestyle=style[classifier],
#                 antialiased=True
#             )
#             plt.fill_between(
#                 dimensions,
#                 minimum,
#                 maximum,
#                 alpha=0.2, 
#                 edgecolor=color[classifier],
#                 facecolor=color[classifier],
#                 linestyle=style[classifier],
#                 antialiased=True
#             )
