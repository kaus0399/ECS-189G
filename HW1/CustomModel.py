import math, collections
# I will implement a trigram model as it was easiest for me to understand. I know N-gram is usually not the best, but usually we can get away with it. 
# I will use the bigram model as a basis to create this one. 
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.unigram_count = collections.defaultdict(lambda: 0) #one word sequence
    self.bigram_count = collections.defaultdict(lambda: 0) #two word sequence
    self.trigram_count = collections.defaultdict(lambda: 0) #three word sequence
    self.unigram_words = 0 #initialze unigram words
    self.vocabulary = 0
    self.train(corpus)
  

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    #following same for loop structure as other models. 
    for sentence in corpus.corpus:
      first_previous_word = None
      second_previous_word = None
      third_previous_word = None
      for datum in sentence.data:
        word = datum.word
        self.unigram_count[tuple([word])] += 1 #increment one word sequence by one as word = datum.word
        if first_previous_word != None:
          self.bigram_count[tuple([first_previous_word,word])] += 1 #increment two-word sequence as the previous word and current word
        if second_previous_word != None:
          self.trigram_count[tuple([second_previous_word,first_previous_word,word])] += 1 #increment three-word sequence as previous 2 words and currenrt word
        third_previous_word = second_previous_word #Passing on the words as follows
        second_previous_word = first_previous_word
        first_previous_word = word
        
    self.unigram_words=sum(self.unigram_count.values()) #setting unigram words to amount of unigram words
    self.vocabulary=len(self.unigram_count) #Vocabulary dict will be length of unigram count


  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    score = 0 #initialize score
    first_previous_word = None
    second_previous_word = None
    third_previous_word = None
    #Iterate over words in sentence and establish tupule with the approrpiate word
    for word in sentence:
      num_tri_words = self.trigram_count[tuple([second_previous_word, first_previous_word, word])]
      num_bi_words = self.bigram_count[tuple([second_previous_word, first_previous_word])]
      #Consideiring that num tri words isnt empty. 
      if (num_tri_words > 0):
        score += math.log(num_tri_words)
        score -= math.log(num_bi_words)
      else: # in the scenario where doesnt qualify for trigram
        num_bi_words = self.bigram_count[tuple([first_previous_word, word])]
        num_uni_words = self.unigram_count[tuple([first_previous_word])]
        #Consiering that bi words isn't empty 
        if (num_bi_words > 0):
          score += math.log(num_bi_words)
          score -= math.log(num_uni_words)
        else:
            score += math.log(self.unigram_count[tuple([word])] + 1.0)
            score -= math.log(self.vocabulary + self.unigram_words)
      third_previous_word = second_previous_word
      second_previous_word = first_previous_word
      first_previous_word = word
    return score
