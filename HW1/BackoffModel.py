import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    previous_word = "<s>"
    for sentence in corpus.corpus:
       for datum in sentence.data:
          word = datum.word
          self.unigramCounts[word] += 1
          #tuple consisting of previous_word and the current word
          bigram = (previous_word,word)   
          self.bigramCounts[bigram] += 1
          self.total += 1
          previous_word = word

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0
    previous_word = "<s>"
    for word in sentence:
      bigram = (previous_word,word)
      count = self.bigramCounts[bigram]
      if count > 0:
        score += math.log(count)
        score -= math.log(self.unigramCounts[previous_word])
      else:
        count = self.unigramCounts[word]
        score +=math.log(count + 1)
        score -=math.log(self.total + len(self.unigramCounts))
        score +=math.log(1.6) #growth factor of 1.6. I read research papers and 1.6-1.8 was a good range. So I started off with 1.6
      previous_word = word
    return score