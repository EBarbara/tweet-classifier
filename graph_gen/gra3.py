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
    (VECTORIZER, vectorizers[1]),
    (VECTORIZER, vectorizers[2]),
    (CLASSIFIER, classifiers[0]),
    (CLASSIFIER, classifiers[1]),
    (CLASSIFIER, classifiers[2]),
]

metrics = ['accuracy', 'precision', 'recall']

input_filename = 'data_stats.csv'
fontsize_label = 18
fontsize_numbers = 12
fontsize_legend = 16

def build_metric(minimum, average, maximum):
    return {
        'min': minimum,
        'avg': average,
        'max': maximum,
    }

def build_plot(dataset, vectorizer, classifier):
    dim = []
    acc = []
    pre = []
    rec = []
    for row in dataset:
        if row['vectorizer'] == vectorizer and row['classifier'] == classifier:
            dim.append(float(row['dimensions']))
            acc.append(build_metric(
                float(row['min_acc']) * 100,
                float(row['avg_acc']) * 100,
                float(row['max_acc']) * 100
            ))
            pre.append(build_metric(
                float(row['min_precision']) * 100,
                float(row['avg_precision']) * 100,
                float(row['max_precision']) * 100
            ))
            rec.append(build_metric(
                float(row['min_recall']) * 100,
                float(row['avg_recall']) * 100,
                float(row['max_recall']) * 100
            ))
    return {
        'dimensions': dim,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
    }

def plot_data(config):
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
        output_filename = os.path.join('graphs', '_'.join([main_data, metric]))
        
        plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()

        for i in range(elements):
            x = data[i]['dimensions']
            y = [part['avg'] for part in data[i][metric]]
            limits = [
                [part['min'] for part in data[i][metric]],
                [part['max'] for part in data[i][metric]]
            ]
            error = [
                [avg - lower for avg, lower in zip(y, limits[0])],
                [higher - avg for avg, higher in zip(y, limits[1])],
            ]
            ax.errorbar(x, y, yerr=error, label=labels[i], color=color[i], linewidth=4, linestyle=style[i], antialiased=True)
            plt.fill_between(
                x, [y - error for y, error in zip(y, error[0])], [y + error for y, error in zip(y, error[1])],
                alpha=0.2, edgecolor=color[i], facecolor=color[i], linestyle=style[i], antialiased=True
            )
     
        plt.xlabel('Dimensions [n]', fontsize=fontsize_label)
        plt.ylabel(metric.capitalize() + ' (%)', fontsize=fontsize_label)
        plt.legend(fontsize=fontsize_legend)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([100, 300, 600, 1000])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_numbers) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_numbers)

        plt.ylim(88, 100)
        plt.savefig(output_filename)
        plt.close()

for config in configs:
    plot_data(config=config)