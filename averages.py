import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame

df = pd.read_csv('data_final_10.csv')
df_pca = pd.read_csv ('results/data_final_thesis_redux.csv')

vectors = ['Pt-BR_Word2Vec', 'Pt-BR_GloVe', 'Pt-BR_FastText',]
classifiers = ['cnn', 'lstm', 'mixed',]
work_columns = ['vector_time (s)', 'train_time (s)', 'accuracy', 'precision', 'recall',]
percent_metric = ['Acurácia (%)', 'Precisão (%)', 'Recall (%)',]

name_class = {
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'mixed': 'CNN+LSTM',
}

name_vector = {
    'Pt-BR_Word2Vec': 'Word2Vec',
    'Pt-BR_GloVe': 'GloVe',
    'Pt-BR_FastText': 'fastText',
}

columns = [
    'vectorizer',
    'classifier',
    'metric',
    #'min_300_orig',
    'média 300 original',
    #'max_300_orig',
    #'min_600_reduce',
    'média reduzido de 600',
    #'max_600_reduce',
    #'min_1000_reduce',
    'média reduzido de 1000',
    #'max_1000_reduce',
]

data = []
for work_column in work_columns:
    for vector in vectors:
        for classifier in classifiers:
            df_work = df.loc[
                (df['vectorizer'] == vector) &
                (df['classifier'] ==  classifier) &
                (df['dimensions'] == 300),
            ][work_column].mean()
            df_pca_work = df_pca.loc[
                (df_pca['vectorizer'] == vector) &
                (df_pca['classifier'] ==  classifier),
            ].groupby('dimensions')[work_column].mean()
            
            data.append([vector, classifier, work_column, df_work, df_pca_work[600], df_pca_work[1000]])

dataframe = pd.DataFrame(data)
dataframe.columns = columns
dataframe.loc[dataframe['vectorizer'] == 'Pt-BR_Word2Vec', 'vectorizer'] = 'Word2Vec'
dataframe.loc[dataframe['vectorizer'] == 'Pt-BR_FastText', 'vectorizer'] = 'fastText'
dataframe.loc[dataframe['vectorizer'] == 'Pt-BR_GloVe', 'vectorizer'] = 'GloVe'
dataframe.loc[dataframe['classifier'] == 'cnn', 'classifier'] = 'CNN'
dataframe.loc[dataframe['classifier'] == 'lstm', 'classifier'] = 'LSTM'
dataframe.loc[dataframe['classifier'] == 'mixed', 'classifier'] = 'CNN+LSTM'
dataframe.loc[dataframe['metric'] == 'vector_time (s)', 'metric'] = 'Tempo de vetorização (s)'
dataframe.loc[dataframe['metric'] == 'train_time (s)', 'metric'] = 'Tempo de treinamento (s)'
dataframe.loc[dataframe['metric'] == 'accuracy', 'metric'] = 'Acurácia (%)'
dataframe.loc[dataframe['metric'] == 'precision', 'metric'] = 'Precisão (%)'
dataframe.loc[dataframe['metric'] == 'recall', 'metric'] = 'Recall (%)'
dataframe['label'] = dataframe[['vectorizer', 'classifier',]].agg('\n'.join, axis=1)

plot_metrics = dataframe['metric'].unique()

for plot_metric in plot_metrics:
    plot_dataset = dataframe[dataframe['metric'] == plot_metric]
    plot_dataset.reset_index(drop=True, inplace=True)
    if plot_metric in percent_metric:
        plot_dataset['média 300 original'] *=100
        plot_dataset['média reduzido de 600'] *=100
        plot_dataset['média reduzido de 1000'] *=100

    ax = plot_dataset.plot(kind='bar', rot=0, figsize=(10, 8))
    ax.set_xticks(plot_dataset.index)
    ax.set_xticklabels(plot_dataset.label)
    if plot_metric in percent_metric:
        ax.set_ylim(94, 98)
    elif plot_metric == 'Tempo de vetorização (s)':
        ax.set_ylim(260, 350)
    else:
        ax.set_ylim(0, 3000)
    ax.set_ylabel(plot_metric)
    ax.set_xlabel("Vetorizador/Classificador")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join('graphs', 'pca', plot_metric))
    plt.close()