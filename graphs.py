import csv
import itertools
import os

import matplotlib.pyplot as plt


elements = 3

VECTORIZER = 'vectorizer'
CLASSIFIER = 'classifier'

vectorizers = ['W2V', 'GloVe', 'fastText']
classifiers = ['CNN', 'LSTM', 'CNN+LSTM']
configs = [
    (VECTORIZER, vectorizers[0]),
]

errors = [
    [
        [[0.07, 0.08, 0.07, 0.03, 0.06],[0.08, 0.07, 0.07, 0.04, 0.07]], #CNN
        [[0.12, 0.06, 0.08, 0.14, 0.08],[0.11, 0.07, 0.17, 0.13, 0.12]], #LSTM
        [[0.07, 0.10, 0.05, 0.14, 0.08],[0.20, 0.04, 0.05, 0.15, 0.09]], #MIX
    ], # W2V
       [
        [[0.11, 0.08, 0.07, 0.03, 0.06],[0.15, 0.07, 0.07, 0.04, 0.07]], #CNN
        [[0.95, 0.06, 0.08, 0.14, 0.08],[0.54, 0.07, 0.17, 0.13, 0.12]], #LSTM
        [[0.46, 0.10, 0.05, 0.14, 0.08],[0.68, 0.04, 0.05, 0.15, 0.09]], #MIX
    ], # GloVe
    [
        [[0.07, 0.08, 0.07, 0.03, 0.06],[0.08, 0.07, 0.07, 0.04, 0.07]], #CNN
        [[0.12, 0.06, 0.08, 0.14, 0.08],[0.11, 0.07, 0.17, 0.13, 0.12]], #LSTM
        [[0.07, 0.10, 0.05, 0.14, 0.08],[0.20, 0.04, 0.05, 0.15, 0.09]], #MIX
    ], # FastText
]

metrics = ['accuracy', 'precision', 'recall', 'f1']

input_filename = 'average.csv'
fontsize_label = 22
fontsize_legend = 22

def build_plot(dataset, vectorizer, classifier):
    tt = []
    dim = []
    acc = []
    pre = []
    rec = []
    f1 = []
    for row in dataset:
        if row['vectorizer'] == vectorizer and row['classifier'] == classifier:
            dim.append(float(row['dimensions']))
            tt.append(float(row['train_time (s)']))
            acc.append(float(row['accuracy']))
            pre.append(float(row['precision']))
            rec.append(float(row['recall']))
            f1.append(float(row['f1']))
    return {
        'dimensions': dim,
        'train': tt,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }

def plot_data(config, sub_errors):
    plot_focus = config[0]
    main_data = config[1]

    labels = vectorizers if plot_focus == CLASSIFIER else classifiers
    style = ['-', '--', ':']
    color = ['#0000FF', '#00FF00', '#FF0000']
  
    data = [None] * elements
    with open(input_filename) as csv_file:
        dataset = csv.DictReader(csv_file, delimiter=',')
        for i in range(elements):
            if plot_focus == VECTORIZER:
                data[i] = build_plot(dataset, main_data, classifiers[i])
            elif plot_focus == CLASSIFIER:
                data[i] = build_plot(dataset, vectorizers[i], main_data)
            else:
                return
            csv_file.seek(0)

    for metric in metrics:
        fig, ax = plt.subplots()
        
        output_filename = os.path.join('graphs', '_'.join([main_data, metric]))

        for i in range(elements):
            x = data[i]['dimensions']
            y = data[i][metric]
            error = sub_errors[i]
            ax.errorbar(
                x,
                y,
                yerr=error,
                label=labels[i],
                color=color[i],
                linewidth=4,
                linestyle=style[i],
                antialiased=True
            )
            plt.fill_between(
                x,
                [y - error for y, error in zip(y, error[0])],
                [y + error for y, error in zip(y, error[1])],
                alpha=0.2, 
                edgecolor=color[i],
                facecolor=color[i],
                linestyle=style[i],
                antialiased=True
            )
           
        # reduzir fonte dos labels
        # aumentar dos eixos

        plt.xlabel('Dimensions [n]', fontsize=fontsize_label)
        plt.ylabel(metric.capitalize() + ' (%)', fontsize=fontsize_label)
        plt.legend(fontsize=fontsize_legend)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.ylim(94, 98)
        plt.savefig(output_filename)
        plt.close()

for config in configs:
    plot_data(config=config, sub_errors=errors[0])
