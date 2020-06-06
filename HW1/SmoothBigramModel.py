import math, collections
class SmoothBigramModel:
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    #self.total = 0
    self.train(corpus)
  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    for sentence in corpus.corpus:
      #print(sentence.data)
      SD_len = len(sentence.data)
      for datum in range(SD_len):
        word= sentence.data[datum].word
        self.unigramCounts[word] += 1
        if datum>0:
          bit = (sentence.data[datum-1].word, sentence.data[datum].word)
          self.bigramCounts[bit] += 1
        #self.total += 1

  def score(self, sentence):
    """Takes a list of strings, returns a score of that sentence."""
    score = 0.0
    uni_len=len(self.unigramCounts)
    #for word in sentence:
    for datum in range(len(sentence)-1):
      word=(sentence[datum], sentence[datum+1])
      count = self.bigramCounts[word]
      score += math.log(count+1)
      score -= math.log(self.unigramCounts[sentence[datum]]+uni_len)
    return score