from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import inaugural
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import Text
from nltk import collocations
from nltk.sentiment import SentimentIntensityAnalyzer

# Requires installation of nltk (pip install nltk) and some nltk.download("something").

##################################
### EXAMPLE: ARTICLE FROM FILE ###
##################################

# Open and read a file into a variable. Choose your own path/file.
file = open("articles_bi/Are nurses more altruistic than real estate br.txt", "r")
article_altruism = file.read()
file.close()

words = word_tokenize(article_altruism)
sentences  = sent_tokenize(article_altruism)

# Print sentences with the word "good" in it.
for sentence in sentences:
	ws = word_tokenize(sentence)
	if "good" in [w.lower() for w in ws]:
		print(sentence,"\n")





###################################
### EXAMPLE: INAUGURAL SPEECHES ###
###################################

# Building lists of words in three speeches of choice
washington_words = [w for w in inaugural.words("1789-Washington.txt") if w.isalpha()]
trump_words = [w for w in inaugural.words("2017-Trump.txt") if w.isalpha()]
biden_words = [w for w in inaugural.words("2021-Biden.txt") if w.isalpha()]

# List of stopwords
stopwords = stopwords.words('english')

# Removing stopwords
washington_words = [w for w in washington_words if w.lower() not in stopwords]
trump_words = [w for w in trump_words if w.lower() not in stopwords]
biden_words = [w for w in biden_words if w.lower() not in stopwords]

# Building a frequency distribution of words - counting words
washington_fd = FreqDist([w.lower() for w in washington_words])
trump_fd = FreqDist([w.lower() for w in trump_words])
biden_fd = FreqDist([w.lower() for w in biden_words])

# Printing a table of the three most common words
washington_fd.tabulate(3)
trump_fd.tabulate(3)
biden_fd.tabulate(3)



# Investigating concordances - word occurence with it's context
my_word = "great"

washington_text = Text(inaugural.words("1789-Washington.txt"))
washington_conc_list = washington_text.concordance_list(my_word, lines=3, width=100)

trump_text = Text(inaugural.words("2017-Trump.txt"))
trump_conc_list = trump_text.concordance_list(my_word, lines=3, width=100)

biden_text = Text(inaugural.words("2021-Biden.txt"))
biden_conc_list = biden_text.concordance_list(my_word, lines=3, width=100)

print("---WASHINGTON:")
for entry in washington_conc_list:
	print(entry.line)

print("---TRUMP:")
for entry in trump_conc_list:
	print(entry.line)

print("---BIDEN:")
for entry in biden_conc_list:
	print(entry.line)



# Finding Collocations with two words - Bigrams
print("---WASHINGTON:")
washington_finder = collocations.BigramCollocationFinder.from_words(washington_words)
washington_ngrams = washington_finder.ngram_fd.most_common(3)
print(washington_ngrams)

print("---TRUMP:")
trump_finder = collocations.BigramCollocationFinder.from_words(trump_words)
trump_ngrams = trump_finder.ngram_fd.most_common(3)
print(trump_ngrams)

print("---BIDEN:")
biden_finder = collocations.BigramCollocationFinder.from_words(biden_words)
biden_ngrams = biden_finder.ngram_fd.most_common(3)
print(biden_ngrams)



"""
Using NLTK’s Pre-Trained Sentiment Analyzer:
VADER (Valence Aware Dictionary for Sentiment Reasoning)
Motivation:
We won’t try to determine if a sentence is objective or subjective, fact or opinion. 
Rather, we care only if the text expresses a positive, negative or neutral opinion.
"""

sia = SentimentIntensityAnalyzer()

# Testing SIA ...
print("TESTING.. with two extreme sentences:")
print(sia.polarity_scores("I am happy happy happy, and everything is excellent and good! My life is perfect!"))
print(sia.polarity_scores("I feel bad, everything is bad bad bad. My life is the worst. All is lost, I am sad sad sad."))


print("---WASHINGTON:")
print(sia.polarity_scores(inaugural.raw("1789-Washington.txt")))
print("---TRUMP:")
print(sia.polarity_scores(inaugural.raw("2017-Trump.txt")))
print("---BIDEN:")
print(sia.polarity_scores(inaugural.raw("2021-Biden.txt")))


print(sia.polarity_scores(article_altruism))

