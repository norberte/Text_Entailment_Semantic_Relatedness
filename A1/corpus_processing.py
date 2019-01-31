import nltk
from gensim import models
from text_cleaning import textCleaning
import urllib

"""
Data format is assumed to be one tweet per line

Input: path to file as a string
Output: list of tweets, each element being 1 line (tweet) from the data set
"""
def import_data(filename):
    data = []
    with open(filename, encoding="utf8") as input_file:
        for line in input_file:
           data.append(line)
    return data


def get_stopwords(link):
    f = urllib.urlopen(link)
    myfile = f.read()

    stopwords = []
    with open(myfile, 'r') as f:
        for line in f:
            stopwords.append(line.rstrip())
    return stopwords

def part_C(outputFile, freq_distr):
    # output token frequency distribution
    print("Part C")
    with open(outputFile, 'w', encoding="utf8") as f:
        f.truncate()
        f.write(freq_distr.tabulate(True) + "\n")

    print()
    print(freq_distr.unicode_repr())
    print()

def part_B(freq_distr):
    print("Part B")

    print("Q1: How many tokens did you find in the corpus?")
    number_of_tokens = freq_distr.B()
    print("A1: ", "There are ", str(number_of_tokens), "tokens in the corpus.")
    print()

    print("Q2: How many types (unique tokens) did you have?")
    unique_tokens = set(freq_distr.keys())
    number_of_types = len(unique_tokens)
    print("A2: ", "There are ", str(number_of_tokens), "tokens in the corpus.")
    print()

    print("Q3: What is the type / token ratio for the corpus?")
    ratio = float(number_of_types / number_of_tokens)
    print("A3: The type / token ratio is ", str(ratio))
    print()


def part_D(freq_distr):
    print("part D:")
    print("Question: How many tokens appeared only once in the corpus?")
    print("Answer: There are ", len(freq_distr.hapaxes()), "tokens that appeared only once (also called hapax legomena)")
    print()

def part_A(fileName, outputFile):
    # import data
    tweets = import_data(fileName)

    # text preprocessing
    cleanTweets = textCleaning(tweets)

    # define a frequency distribution counter
    freq_distr = nltk.probability.FreqDist()

    # tokenization
    with open(outputFile, 'w', encoding="utf8") as f:
        f.truncate()
        for line in cleanTweets:
            tokens = nltk.word_tokenize(line)
            f.write(''.join(tokens) + "\n")
            for word in tokens:
                freq_distr[word] += 1

    return freq_distr


def main():
    # import data
    freq_distr = part_A("../data/microblog2011.txt", "../output/microblog2011_tokenized.txt")

    part_B(freq_distr)
    part_C("Tokens.txt", freq_distr)
    part_D(freq_distr)

    #print(get_stopwords("http://www.site.uottawa.ca/~diana/csi5180/StopWords"))

if __name__ == '__main__':
    main()
