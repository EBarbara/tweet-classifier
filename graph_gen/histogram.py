import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

remove_special_chars = re.compile("[^A-Za-z0-9 ]+")
filename = 'dataset/Merged_classified/revised.csv'
stopwords = stopwords.words('portuguese')
fontsize_label = 18

def clean(sentence):
    return re.sub(remove_special_chars, "", sentence.lower())


with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    tweet_words = []
    stopped_words = []
    for tweet in reader:
        cleaned_sentence = clean(tweet[4])
        word_tokens = word_tokenize(cleaned_sentence)
        filtered_sentence = [w for w in word_tokens if not w in stopwords]
        tweet_words.append(len(word_tokens))
        stopped_words.append(len(filtered_sentence))
    print('done')

fig, ax = plt.subplots()
plt.hist(
    stopped_words, bins=60, range=(1, 60),
    density=True, cumulative=True,
    facecolor='none', edgecolor='blue', lw=0.7
)
plt.xlabel('Words per tweet', fontsize=fontsize_label)
# plt.ylabel('Relative frequency', fontsize=fontsize_label) 
plt.ylabel('Empirical cummulative distribution', fontsize=fontsize_label)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('histogram cumulative (2)')
