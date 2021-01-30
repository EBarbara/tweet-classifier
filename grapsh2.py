import csv

import matplotlib.pyplot as plt
import numpy as np


def build_plot(dataset, samples, entity):
    smp = []
    mdl = []
    ent = []
    pre = []
    rec = []
    f1 = []
    tp = []
    fp = []
    fn = []
    for row in dataset:
        if row['Sample'] in samples and row['Entity'] == entity:
            smp.append(row['Sample'])
            mdl.append(row['Model'])
            ent.append(row['Entity'])
            pre.append(float(row['Precision']))
            rec.append(float(row['Recall']))
            f1.append(float(row['F1']))
            tp.append(float(row['TP']))
            fp.append(float(row['FP']))
            fn.append(float(row['FN']))
    return {
        'sample': smp,
        'model': mdl,
        'entity': ent,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'fn': fn,
        'fp': fp,
        'tp': tp,
    }


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        text = ax.annotate(
            '{:.2f}%'.format(height * 100),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom'
        )
        text.set_fontsize(10)


filename = 'C:\\Users\\yami_\\Downloads\\resultadosartigoleonardo.csv'
with open(filename) as csv_file:
    dataset = csv.DictReader(csv_file, delimiter=',')
    sample = build_plot(dataset, ('2.1', '2.2', '2.3'), 'ALL')

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111)

x = np.arange(len(sample['model']))  # the label locations
width = 0.30  # the width of the bars

plt.ylim(0.7, 1.0)
plt.yticks([0.70, 0.80, 0.90, 1.00])
ax.set_yticklabels(['70%', '80%', '90%', '100%'])
# plt.ylim(0, 1)
# plt.yticks([0.0, 0.1, 0.2, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
# ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

text = plt.xlabel('Models')
text.set_fontsize(20)
text = plt.ylabel('Measures')
text.set_fontsize(20)
ax.set_xticks(x)
ax.set_xticklabels(sample['model'])

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)

rects1 = ax.bar(x - width, sample['precision'], width, label='Precision')
rects2 = ax.bar(x, sample['recall'], width, label='Recall')
rects3 = ax.bar(x + width, sample['f1'], width, label='F1 Measure')

# plt.plot(sample['model'], sample['precision'], 'k-', color='red', label='Precision')
# plt.plot(sample['model'], sample['recall'], 'k--', color='blue', label='Recall')
# plt.plot(sample['model'], sample['f1'], 'k-*', color='purple', label='F1 Measure')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

plt.legend(fontsize=16, loc=2)

fig.tight_layout()
plt.show()
