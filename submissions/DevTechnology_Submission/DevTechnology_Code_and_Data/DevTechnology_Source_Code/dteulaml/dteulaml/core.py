# -*- coding: utf-8 -*-
"""dteulaml library"""

# Standard library imports
import logging
import os
from collections import Counter
import pickle
import sys

# 3rd-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import brier_score_loss
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import matplotlib.pyplot as plt
from scipy.special import softmax

# dteulaml imports
from . import helpers

eulamodel = None

class AggregateBayes():
  """
  A simple Bayesian bag of words classifier.
  Mimics sklearn's classifier base class, but does not extend it formally, as
  the desire is to accept strings directly as opposed to converting to numeric
  representations.  This is 'duck typed'
  """

  def __init__(self, mingram=1, maxgram=3, textcol='text', cased=False): # type: (AggregateBayes, int, int, str, bool) -> None
    """
    Initialize the classifier

    :param mingram: the minimum size of n-grams to use
    :type mingram: int
    :param maxgram: the maximum size of n-grams to use
    :type maxgram: int
    :param textcol: the name of the text column
    :type textcol: str
    :param cased: preserve case or not
    :type textcol: bool
    """

    self.mingram = mingram
    self.maxgram = maxgram
    self.textcol = textcol
    self.cased = cased

  def get_params(self, deep=True): # type: (AggregateBayes, bool) -> {}
    """
    Gets the init parameters

    :param deep: deep copy of each parameter value or not
    :type deep: bool

    :return: a dict of init parameters
    :rtype: {}
    """

    return {'mingram': self.mingram, 'maxgram': self.maxgram, 'textcol': self.textcol, 'cased': self.cased}

  def set_params(self, **parameters): # type: (AggregateBayes, {}) -> AggregateBayes
    """
    Sets the init parameters (overwrites)

    :param parameters: a dict of parameters
    :type parameters: {}

    :return: self for compatability
    :rtype: AggregateBayes
    """

    for name, value in parameters.items():
      setattr(self, name, value)

    return self

  def counttokens(self, text, ctr): # type: (AggregateBayes, str, Counter) -> int
    """
    Tokenizes the text, increments the counter and returns the number of tokens found

    :param text: the text to tokenize
    :type text: str
    :param ctr: the running total of occurrences of the tokens
    :type ctr: Counter

    :return: the number of tokens found in the text
    :rtype: int
    """

    safetext = text
    if not self.cased:
      safetext = safetext.lower()
    toklist = helpers.allngrams(safetext, self.mingram, self.maxgram)
    tokset = set(toklist)
    ctr.update(tokset)

    return len(toklist)

  def bayesianlikelihood(self, token): # type: (AggregateBayes, str) -> float
    """
    Uses Bayes' Rule to calculate the probability that a text has the target
    classification if it has the token.

    :param token: the token to calculate probability on
    :type token: str

    :return: the probability that a text that has this token is of the target class
    :rtype: float
    """

    numtargets = self.targets_.shape[0]
    numbenign = self.benign_.shape[0]
    probA = numtargets / (numtargets + numbenign)
    probBA = self.targetngramcounts_[token] / numtargets
    probB = (self.targetngramcounts_[token] + self.benignngramcounts_[token]) / (numtargets + numbenign)
    probAB = (probBA * probA) / probB

    return probAB

  def fit(self, X, y): # type: (AggregateBayes, pd.DataFrome, np.ndarray) -> AggregateBayes
    """
    Calculates the Bayesian probability of each n-gram for the texts with label=1

    :param X: An array of texts
    :type X: pd.DataFrame
    :param y: The answers
    :type y: np.ndarray

    :return: self, for compatibility
    :rtype: AggregateBayes
    """

    self.X_ = X
    self.y_ = y
    self.targets_ = X.iloc[np.where(y == 1)[0], :]
    self.benign_ = X.iloc[np.where(y == 0)[0], :]
    self.targetngramcounts_ = Counter()
    self.benignngramcounts_ = Counter()
    self.targets_.apply(lambda row: self.counttokens(row[self.textcol], self.targetngramcounts_), axis=1)
    self.benign_.apply(lambda row: self.counttokens(row[self.textcol], self.benignngramcounts_), axis=1)
    self.probindex_ = {curtargettoken: self.bayesianlikelihood(curtargettoken) for curtargettoken in self.targetngramcounts_}

    return self

  def predict(self, X): # type: (AggregateBayes, pd.DataFrome) -> np.ndarray
    """
    Predicts the classification for X texts with a confidence 0.0-1.0

    :param X: An array of texts
    :type X: pd.DataFrame

    :return: the classifications
    :rtype: np.ndarray
    """

    # for each clause in the original data, sum probabilities of its ngrams
    # divide this by the number of tokens in the clause
    workingdata = X.copy(deep=True)
    retlist = []
    if self.cased:
      toklists = workingdata.apply(lambda row: helpers.allngrams(row[self.textcol], self.mingram, self.maxgram), axis=1).values.tolist()
      retlist = [sum(self.probindex_[curtok] for curtok in toklist if curtok in self.probindex_) / len(toklist) if len(toklist) > 0 else 0.0 for toklist in toklists]
    else:
      toklists = workingdata.apply(lambda row: helpers.allngrams(row[self.textcol].lower(), self.mingram, self.maxgram), axis=1).values.tolist()
      retlist = [sum(self.probindex_[curtok] for curtok in toklist if curtok in self.probindex_) / len(toklist) if len(toklist) > 0 else 0.0 for toklist in toklists]

    return np.array(retlist)

def testABC(outfolder): # type: (str) -> None
  """
  See if the classifier can work at all

  :param outfolder: output folder to write results to
  :type outfolder: str
  """

  sourcedata = helpers.refdf.copy(deep=True)
  print('Raw data: ' + str(sourcedata.shape))
  sourcedata = sourcedata[sourcedata['Clause Text'].map(helpers.goodsize)]
  print('Cleaned data: ' + str(sourcedata.shape))
  sourcedata.set_index('Clause ID', inplace=True)

  traindf, testdf = train_test_split(sourcedata, test_size=0.15, random_state=18)
  print('Training data: ' + str(traindf.shape))
  print('Test data: ' + str(testdf.shape))

  trainanswers = traindf.pop('Classification').values
  testanswers = testdf.pop('Classification').values
  abc = AggregateBayes(mingram=1, maxgram=2, textcol='Clause Text', cased=False)
  abc.fit(traindf, trainanswers)
  testpredictions = abc.predict(testdf)
  locked = np.where(testpredictions > 0.35, 1, 0)
  print('PrecRec: ' + str(precision_recall_fscore_support(testanswers, locked, average='binary')))
  print('Brier: ' + str(brier_score_loss(testanswers, testpredictions)))
  precs, recs, threshes = precision_recall_curve(testanswers, testpredictions)
  plt.plot(recs, precs, marker='.')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend()
  plt.show()

  outputdf = helpers.refdf.copy(deep=True)
  allpredictions = abc.predict(outputdf)
  outputdf['Predictions'] = allpredictions
  if os.path.isdir(outfolder):
    helpers.writedftocsv(os.path.join(outfolder, helpers.filenames['abcoutput']), outputdf)

def findconflicts(outpath): # type: (str) -> None
  """
  Analyzes texts of EULA clauses to figure out character counts, cases, punctuation, tokens, etc.

  :param outpath: the directory to write the results in
  :type outpath: str
  """

  highmatches = []
  blacklist = set()
  accs = [curdict for curdict in helpers.accExamples]
  unaccs = [curdict for curdict in helpers.unaccExamples]
  for dict in accs:
    dict['Clause Tokens'] = helpers.allngrams(dict['Clause Text'].lower(), 1, 3)
  for dict in unaccs:
    dict['Clause Tokens'] = helpers.allngrams(dict['Clause Text'].lower(), 1, 3)

  for curAcc in accs:
    for curUnacc in unaccs:
      curscore = helpers.dicescore(curAcc['Clause Tokens'], curUnacc['Clause Tokens'])
      if curscore > 0.8:
        newfact = {}
        newfact['Acceptable Clause ID'] = curAcc['Clause ID']
        newfact['Unacceptable Clause ID'] = curUnacc['Clause ID']
        newfact['Similarity'] = curscore
        newfact['Acceptable Len'] = len(curAcc['Clause Text'])
        newfact['Unacceptable Len'] = len(curUnacc['Clause Text'])
        highmatches.append(newfact)
      if curscore > 0.95:
        blacklist.add(curAcc['Clause ID'])
        blacklist.add(curUnacc['Clause ID'])

  unduped = [curdict for curdict in helpers.refData if curdict['Clause ID'] not in blacklist]
  unduped = [{key: value for key, value in curdict.items() if key != 'Clause Tokens'} for curdict in unduped]

  helpers.writeCSV(highmatches, helpers.headerlists['conflicts'], os.path.join(outpath, helpers.filenames['conflicts']))
  helpers.writeCSV(unduped, helpers.headerlists['base'], os.path.join(outpath, helpers.filenames['deduped']))

def counttextfeatures(outpath): # type: (str) -> None
  """
  Analyzes texts of EULA clauses to figure out character counts, cases, punctuation, tokens, etc.

  :param outpath: the directory to write the results in
  :type outpath: str
  """

  workingdata = [curline for curline in helpers.refData]
  for curline in workingdata:
    curline['Chars'] = len(curline['Clause Text'])
    curline['Tokens'] = len(curline['Clause Text'].split())
    curline['Allcaps'] = sum(1 for curtok in curline['Clause Text'].split() if curtok.isupper())
  unacctoks = [curline['Tokens'] for curline in workingdata if curline['Classification'] == '1' and curline['Tokens'] < 500]
  print(str(len(unacctoks)) + ' unacceptable clauses')
  plt.hist(unacctoks, bins=50)
  plt.xlabel('Tokens in a Clause')
  plt.ylabel('Number of Clauses')
  plt.title('Histogram of Length of Unacceptable Clauses')
  plt.show()
  acctoks = [curline['Tokens'] for curline in workingdata if curline['Classification'] == '0' and curline['Tokens'] < 500]
  print(str(len(acctoks)) + ' acceptable clauses')
  plt.hist(acctoks, bins=50)
  plt.xlabel('Tokens in a Clause')
  plt.ylabel('Number of Clauses')
  plt.title('Histogram of Length of Acceptable Clauses')
  plt.show()
  for curline in workingdata:
    if len(curline['Clause Text']) > 20000:
      curline['Clause Text'] = 'Truncated'
  helpers.writeCSV(workingdata, helpers.headerlists['textanalysis'], os.path.join(outpath, helpers.filenames['textanalysis']))

def buildbertargs(): # type: () -> ClassificationArgs
  """
  Builds arguments for the classifier.  Must be the same for
  training and predicting.

  :return: the arguments
  :rtype: ClassificationArgs
  """

  accargs = ClassificationArgs()
  accargs.num_train_epochs = 5
  accargs.fp16 = False
  accargs.overwrite_output_dir = True
  accargs.evaluate_during_training = False
  accargs.sliding_window = True
  accargs.max_seq_length = 256
  accargs.stride = 0.9
  accargs.labels_list = [1, 0]
  accargs.save_model_every_epoch = False
  accargs.silent = True
  accargs.manual_seed = 18

  return accargs

def trainroberta(): # type: () -> None
  """
  Uses a default BERT language model to train a classifier on clauses
  provided in the training CSV
  """

  sourcedata = helpers.refdf.copy(deep=True)
  print('Raw data: ' + str(sourcedata.shape))
  # sourcedata['Clause Text'] = sourcedata['Clause Text'].str.lower()
  sourcedata = sourcedata[sourcedata['Clause Text'].map(helpers.goodsize)]
  print('Cleaned data: ' + str(sourcedata.shape))
  sourcedata.set_index('Clause ID', inplace=True)

  bertdata = pd.DataFrame({
    'text': sourcedata['Clause Text'],
    'labels': sourcedata['Classification']
  }, index=sourcedata.index)

  traindf, testdf = train_test_split(bertdata, test_size=0.2, random_state=18)

  print('Data for BERT: ' + str(bertdata.shape))
  print('Data for training: ' + str(traindf.shape))
  print('Data for testing: ' + str(testdf.shape))

  accmodel = ClassificationModel('roberta', 'roberta-base', args=buildbertargs(), weight=[2, 1])
  accmodel.train_model(traindf, eval_df=testdf)

  print('---------------')
  print('Test Data Eval:')

  result, model_outputs, wrong_predictions = accmodel.eval_model(testdf)
  print(result)

#  model_outputs = [softmax(curclause, axis=1) for curclause in model_outputs]
#  print(str(model_outputs))

  print('---------------')
  print('Full Data Eval:')

  result, model_outputs, wrong_predictions = accmodel.eval_model(bertdata)
  print(result)

#  model_outputs = [softmax(curclause, axis=1) for curclause in model_outputs]

def finalmodel(outfolder): # type: (str) -> None
  """
  Trains the BERT model using the parameters currently set in
  buildbertargs().  The parameters have been explored with a
  train/test split, so this training is with the full dataset.

  :param outfolder: the folder to write the model to
  :type outfolder: str
  """

  rawdata = helpers.refdf.copy(deep=True)
  print('Raw data: ' + str(rawdata.shape))
  rawdata.set_index('Clause ID', inplace=True)
  # sourcedata = helpers.dedupdf.copy(deep=True)
  # print('Deduped data: ' + str(sourcedata.shape))
  sourcedata = helpers.refdf.copy(deep=True)
  print('Raw data: ' + str(sourcedata.shape))
  sourcedata = sourcedata[sourcedata['Clause Text'].map(helpers.goodsize)]
  print('Sized data: ' + str(sourcedata.shape))
  sourcedata.set_index('Clause ID', inplace=True)

  traindata = pd.DataFrame({
    'text': sourcedata['Clause Text'],
    'labels': sourcedata['Classification']
  }, index=sourcedata.index)

  evaldata = pd.DataFrame({
    'text': rawdata['Clause Text'],
    'labels': rawdata['Classification']
  }, index=rawdata.index)

  print('Data for BERT: ' + str(traindata.shape))

  accargs = buildbertargs()
  accargs.output_dir = outfolder
  accmodel = ClassificationModel('roberta', 'roberta-base', args=accargs, weight=[2, 1])
  accmodel.train_model(traindata)

  print('---------------')
  print('Training Data Eval:')

  result, model_outputs, wrong_predictions = accmodel.eval_model(traindata)
  print(result)

  print('---------------')
  print('Full Data Eval:')

  result, model_outputs, wrong_predictions = accmodel.eval_model(evaldata)
  # {'mcc': 0.9062028924099057, 'tp': 4835, 'tn': 1368, 'fp': 74, 'fn': 140, 'eval_loss': 0.18330956540325125}
  print(result)

def hyperargs(): # type: () -> {}
  """
  Builds different sets of arguments for the classifier.  Must be the same for
  training and predicting.

  :return: the labeled arguments
  :rtype: {}
  """

  retdict = {}

  for curwindow in [128, 64, 32, 256]:
    for curstride in [0.7, 0.8, 0.9]:
      accargs = ClassificationArgs()
      accargs.num_train_epochs = 5
      accargs.fp16 = False
      accargs.overwrite_output_dir = True
      accargs.evaluate_during_training = False
      accargs.sliding_window = True
      accargs.max_seq_length = curwindow
      accargs.stride = curstride
      accargs.labels_list = [1, 0]
      accargs.save_eval_checkpoints = False
      accargs.save_model_every_epoch = False
      accargs.silent = True
      accargs.manual_seed = 18
      retdict['basic5epochs' + str(curwindow) + 'win' + str(int(curstride * 10.0)) + 'stride'] = accargs

  return retdict

def hyperparamsearch(): # type: () -> None
  """
  Trains various BERT models to see what combination of parameters might
  be best.  Unfortunately, the weight parameter is not available in the
  built-in grid search algorithm, so we roll our own.
  """

  sourcedata = helpers.refdf.copy(deep=True)
  print('Raw data: ' + str(sourcedata.shape), file=sys.stderr)
  # sourcedata['Clause Text'] = sourcedata['Clause Text'].str.lower()
  sourcedata = sourcedata[sourcedata['Clause Text'].map(helpers.goodsize)]
  print('Cleaned data: ' + str(sourcedata.shape), file=sys.stderr)
  sourcedata.set_index('Clause ID', inplace=True)

  bertdata = pd.DataFrame({
    'text': sourcedata['Clause Text'],
    'labels': sourcedata['Classification']
  }, index=sourcedata.index)

  traindf, testdf = train_test_split(bertdata, test_size=0.2, random_state=18)
  print('Train data: ' + str(traindf.shape), file=sys.stderr)
  print('Eval data: ' + str(testdf.shape), file=sys.stderr)

  arglist = hyperargs()

  for label, curargs in arglist.items():
    print(label, file=sys.stderr)
    accmodel = ClassificationModel('roberta', 'roberta-base', args=curargs, weight=[2, 1])
    accmodel.train_model(traindf)
    result, model_outputs, wrong_predictions = accmodel.eval_model(testdf)
    print(label + ': ' + str(result))

def loadeulamodel(inpath): # type: (str) -> None
  """
  Loads a trained BERT-based model for classifying EULA clause acceptability

  :param inpath: the path to the model folder
  :type inpath: str
  """

  global eulamodel

  accargs = buildbertargs()
  accargs.output_dir = inpath
  eulamodel = ClassificationModel('roberta', inpath, args=accargs, weight=[2, 1])

def predicteula(inclauses): # type: ([]) -> {}
  """
  Predicts if texts represent unacceptable EULA clauses

  :param inclause: the list of clause texts
  :type inclause: []

  :return: the acceptability decision and softmax probabilities for each window for each clause as a dict of lists
  :rtype: {}
  """

  global eulamodel

  retdict = {}

  cleanclauses = [helpers.cleantext(curclause) for curclause in inclauses]
  predictions, raw_outputs = eulamodel.predict(cleanclauses)

  probs = [softmax(curoutput, axis=1) for curoutput in raw_outputs]

  retdict['clauselen'] = [len(inclause) for inclause in inclauses]
  retdict['prediction'] = predictions
  retdict['windows'] = [prob.tolist() for prob in probs]

  return retdict

def predictall(inpath, outpath): # type: (str, str) -> None
  """
  Predicts acceptability for each clause in the training data.  Writes
  back some details to the CSV.  Assumes that there is a column called
  'Clause Text' in the CSV.  If there is a column called
  'Classification', will display a precision/recall curve.

  :param inpath: the path to the input CSV file
  :type inpath: str
  :param outpath: the path to the output CSV file
  :type outpath: str
  """

  sourcedata = pd.read_csv(inpath, encoding='utf8', converters={'Clause Text': helpers.cleantext})
  texts = sourcedata['Clause Text'].tolist()
  corpusresults = predicteula(texts)
  sourcedata['Predictions'] = corpusresults['prediction']
  sourcedata['Windows'] = [len(curwins) for curwins in corpusresults['windows']]
  sourcedata['Unacc Windows'] = [sum(1 for win in curwins if win[0] > win[1]) for curwins in corpusresults['windows']]
  sourcedata['Acc Windows'] = [sum(1 for win in curwins if win[0] <= win[1]) for curwins in corpusresults['windows']]
  unaccs = [[win[0] for win in curwins] for curwins in corpusresults['windows']]
  sourcedata['Max Prob'] = [max(curwins) for curwins in unaccs]
  sourcedata.loc[sourcedata['Clause Text'].str.len() > 10000, 'Clause Text'] = 'Too long. Redacted.'
  helpers.writedftocsv(outpath, sourcedata)

  if 'Classification' in sourcedata:
    precs, recs, threshes = precision_recall_curve(sourcedata['Classification'], sourcedata['Max Prob'])
    plt.plot(recs, precs, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    print('MCC: ' + str(matthews_corrcoef(sourcedata['Classification'], corpusresults['prediction'])))
    print('Brier: ' + str(brier_score_loss(sourcedata['Classification'], corpusresults['prediction'])))
    print('PrecRec: ' + str(precision_recall_fscore_support(sourcedata['Classification'], corpusresults['prediction'], average='binary')))

def validation(modelfolder, inputfile, outputfile): # type: (str, str, str) -> None
  """
  Runs the trained model on the validation data, writes the predictions
  to the validation output format.

  :param modelfolder: the folder with the latest BERT model
  :type modelfolder: str
  :param inputfile: the validation data
  :type inputfile: str
  :param outputfile: the prediction file
  :type outputfile: str
  """

  sourcedata = pd.read_csv(inputfile, encoding='utf8', converters={'Clause Text': helpers.cleantext})
  texts = sourcedata['Clause Text'].tolist()
  corpusresults = predicteula(texts)
  sourcedata['Prediction'] = corpusresults['prediction']
  accs = [[win[1] for win in curwins] for curwins in corpusresults['windows']]
  sourcedata['Probability Acceptable'] = [max(curwins) for curwins in accs]
  sourcedata = sourcedata.drop('Clause Text', 1)
  helpers.writedftocsv(outputfile, sourcedata)

def exploretransfer(): # type: () -> None
  """
  Experiment with transfer learning model
  """

  the508 = pickle.load(open(helpers.gettransferfile(), 'rb'))
  for item in the508:
    print('----------')
    for attr, val in item.items():
      if type(val) == str:
        print(attr + ': ' + str(len(val)) + ' chars')
        print(val)
      else:
        print(attr + ': ' + str(val))
