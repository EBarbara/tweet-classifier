import csv
import os

import matplotlib.pyplot as plt

configs = [
    ('Pt-BR_Word2Vec','cnn'),
    ('Pt-BR_Word2Vec','lstm'),
    ('Pt-BR_Word2Vec','mixed'), 
    #('Pt-BR_GloVe','cnn'),
    #('Pt-BR_GloVe','lstm'),
    #('Pt-BR_GloVe','mixed'),
    #('Pt-BR_FastText','cnn'),
    #('Pt-BR_FastText','lstm'),
    #('Pt-BR_FastText','mixed'),
]

elements = 3
input_filename = 'results/data_train_time.csv'

labels = {
    'Pt-BR_Word2Vec': 'W2V',
    'Pt-BR_GloVe': 'GloVe',
    'Pt-BR_FastText': 'fastText',
    'cnn': 'CNN', 
    'lstm': 'LSTM', 
    'mixed': 'CNN+LSTM',
}

style = {
    'Pt-BR_Word2Vec': '-',
    'Pt-BR_GloVe': '--',
    'Pt-BR_FastText': ':',
    'cnn': '-',
    'lstm': '--',
    'mixed': ':',
}

color = {
    'Pt-BR_Word2Vec': '#0000FF', 
    'Pt-BR_GloVe': '#00FF00', 
    'Pt-BR_FastText': '#FF0000',
    'cnn': '#0000FF', 
    'lstm': '#00FF00', 
    'mixed': '#FF0000',
}

fontsize_label = 16
fontsize_legend = 16

with open(input_filename) as csv_file:
    dataset = csv.DictReader(csv_file, delimiter=',')

    fig, ax = plt.subplots()
    for config in configs:
        dimensions = []
        error_low = []
        error_high = []
        value = []

        for data in dataset:
            if data['vectorizer'] == config[0] and data['classifier'] == config[1]:
                dimensions.append(float(data['dimensions']))
                error_low.append(float(data['diff_under']))
                error_high.append(float(data['diff_over']))
                value.append(float(data['mean']))

        csv_file.seek(0)

        x = dimensions
        y = value
        error = [error_low, error_high]
        ax.errorbar(
            x,
            y,
            yerr=error,
            label=labels[config[1]],
            color=color[config[1]],
            linewidth=4,
            linestyle=style[config[1]],
            antialiased=True
        )
        plt.fill_between(
            x,
            [y - error for y, error in zip(y, error[0])],
            [y + error for y, error in zip(y, error[1])],
            alpha=0.2, 
            edgecolor=color[config[1]],
            facecolor=color[config[1]],
            linestyle=style[config[1]],
            antialiased=True
        )

    plt.xlabel('Dimensões [n]', fontsize=fontsize_label)
    plt.ylabel('Tempo treinamento (s)', fontsize=fontsize_label)
    plt.legend(fontsize=fontsize_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # plt.ylim(94, 98)
    plt.savefig('graphs/W2V_time_train')
    plt.close()
