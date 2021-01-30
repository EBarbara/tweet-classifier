import matplotlib.pyplot as plt
import pandas as pd

input_filename = 'data_10_minmax.csv'
df = pd.read_csv(input_filename)
df.loc[df['vectorizer'] == 'Pt-BR_Word2Vec', 'vectorizer'] = 'Word2Vec'
df.loc[df['vectorizer'] == 'Pt-BR_GloVe', 'vectorizer'] = 'GloVe'
df.loc[df['vectorizer'] == 'Pt-BR_FastText', 'vectorizer'] = 'fastText'
df.loc[df['classifier'] == 'cnn', 'classifier'] = 'CNN'
df.loc[df['classifier'] == 'lstm', 'classifier'] = 'LSTM'
df.loc[df['classifier'] == 'mixed', 'classifier'] = 'CNN+LSTM'

fontsize_label = 22
fontsize_legend = 22

def plot_by_vectorizer(dataframe, vectorizer_name, metric_name, metric_name_pt, legend_loc):
	df_vec = df.loc[df['vectorizer'] == vectorizer_name]
	classifiers = df_vec.classifier.drop_duplicates().values
	dimensions = df_vec.dimensions.drop_duplicates().values
	fig, ax = plt.subplots(1, 1)
	
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
	
	for classifier in classifiers:
		row_classifier = df_vec.loc[df_vec.classifier == classifier]
		errors = [
			(row_classifier[f'mean_{metric_name}'].values - row_classifier[f'min_{metric_name}'].values)  * 100,
			(row_classifier[f'max_{metric_name}'].values - row_classifier[f'mean_{metric_name}'].values)  * 100
		]
		ax.errorbar(
			dimensions,
            row_classifier[f'mean_{metric_name}'] * 100,
            yerr=errors,
            label=classifier,
            color=color[classifier],
            linewidth=4,
            linestyle=style[classifier],
            antialiased=True
        )
		ax.fill_between(
			dimensions,
            row_classifier[f'min_{metric_name}'].values * 100,
            row_classifier[f'max_{metric_name}'].values * 100,
            alpha=0.2, 
            edgecolor=color[classifier],
            facecolor=color[classifier],
            linestyle=style[classifier],
            antialiased=True
		)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(94, 98)
	fig.set_size_inches(10, 8)
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.set_xticks(dimensions)
	plt.xlabel('Dimensions [n]', fontsize=fontsize_label)
	plt.ylabel(metric_name_pt.capitalize() + ' (%)', fontsize=fontsize_label)
	plt.legend(fontsize=fontsize_legend, loc=legend_loc)
	fig.show()
	fig.savefig(f'graphs/redone/{metric_name}_{vectorizer_name}.png', dpi=150)

def plot_by_classifier(dataframe, classifier_name, metric_name, metric_name_pt, legend_loc):
	df_cls = dataframe.loc[dataframe['classifier'] == classifier_name]
	vectorizers = df_cls.vectorizer.drop_duplicates().values
	dimensions = df_cls.dimensions.drop_duplicates().values
	fig, ax = plt.subplots(1, 1)
	
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
	
	for vectorizer in vectorizers:
		row_vectorizer = df_cls.loc[df_cls.vectorizer == vectorizer]
		errors = [
			(row_vectorizer[f'mean_{metric_name}'].values - row_vectorizer[f'min_{metric_name}'].values)  * 100,
			(row_vectorizer[f'max_{metric_name}'].values - row_vectorizer[f'mean_{metric_name}'].values)  * 100
		]
		ax.errorbar(
			dimensions,
            row_vectorizer[f'mean_{metric_name}'] * 100,
            yerr=errors,
            label=vectorizer,
            color=color[vectorizer],
            linewidth=4,
            linestyle=style[vectorizer],
            antialiased=True
        )
		ax.fill_between(
			dimensions,
            row_vectorizer[f'min_{metric_name}'].values * 100,
            row_vectorizer[f'max_{metric_name}'].values * 100,
            alpha=0.2, 
            edgecolor=color[vectorizer],
            facecolor=color[vectorizer],
            linestyle=style[vectorizer],
            antialiased=True
		)

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.ylim(94, 98)
	fig.set_size_inches(10, 8)
	ax.tick_params(axis='both', which='major', labelsize=16)
	ax.set_xticks(dimensions)
	plt.xlabel('Dimensions [n]', fontsize=fontsize_label)
	plt.ylabel(metric_name_pt.capitalize() + ' (%)', fontsize=fontsize_label)
	plt.legend(fontsize=fontsize_legend, loc=legend_loc)
	fig.show()
	fig.savefig(f'graphs/redone/{metric_name}_{classifier_name}.png', dpi=150)

#for vectorizer in ['Word2Vec', 'GloVe', 'fastText']:
#	plot_by_vectorizer(df, vectorizer, 'recall', 'recall', 0)

for classifier in ['CNN', 'LSTM', 'CNN+LSTM']:
	plot_by_classifier(df, classifier, 'recall', 'recall', 0)