import csv
import os

import matplotlib.pyplot as plt

configs = [
    'Pt-BR_Word2Vec',
    'Pt-BR_GloVe',
    'Pt-BR_FastText',
]

elements = 3
input_filename = 'data_vect_time_redux.csv'

labels = {
    'Pt-BR_Word2Vec': 'W2V',
    'Pt-BR_GloVe': 'GloVe',
    'Pt-BR_FastText': 'fastText',
}

style = {
    'Pt-BR_Word2Vec': '-',
    'Pt-BR_GloVe': '--',
    'Pt-BR_FastText': ':',
}

color = {
    'Pt-BR_Word2Vec': '#0000FF', 
    'Pt-BR_GloVe': '#00FF00', 
    'Pt-BR_FastText': '#FF0000',
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
            if data['vectorizer'] == config:
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
            label=labels[config],
            color=color[config],
            linewidth=4,
            linestyle=style[config],
            antialiased=True
        )
        plt.fill_between(
            x,
            [y - error for y, error in zip(y, error[0])],
            [y + error for y, error in zip(y, error[1])],
            alpha=0.2, 
            edgecolor=color[config],
            facecolor=color[config],
            linestyle=style[config],
            antialiased=True
        )

    plt.xlabel('Dimensões [n]', fontsize=fontsize_label)
    plt.ylabel('Tempo treinamento (s)', fontsize=fontsize_label)
    plt.legend(fontsize=fontsize_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # plt.ylim(94, 98)
    plt.savefig('graphs/time_train_redux')
    plt.close()
