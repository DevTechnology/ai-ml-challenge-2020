# -*- coding: utf-8 -*-
"""dteulaapp library"""

# Standard library imports
import logging
import base64
import random
import io
import os
import json
import atexit

# 3rd-party imports
import PyPDF2
import docx
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from scipy.special import softmax

# dteulaapp imports
from . import helpers

classifications = ['Acceptable', 'Unacceptable', 'Not Sure']
accsearch = []
unaccsearch = []
feedback = []
eulamodel = None

def calcdice(indict, intoks): # type: ({}, []) -> {}
  """
  Adds a dice score for a dict's Tokens and an input list
  of tokens to the dict

  :param indict: the dict
  :type indict: {}

  :return: the dict with dice score added
  :rtype: {}
  """

  if 'Tokens' not in indict:
    return indict

  dictset = set(indict['Tokens'])
  inset = set(intoks)

  dicescore = (2.0 * len(dictset.intersection(inset))) / (len(dictset) + len(inset))
  indict['Dice'] = dicescore

  return indict

def dicesearch(intext, inspace): # type (str, []) -> str
  """
  Searches a space of token lists for a close match
  with a given text

  :param intext: the text to use to find a match
  :type intext: str
  :param inspace: the token space to look in (list of dicts)
  :type inspace: []

  :return: the text of the closest match
  :rtype: str
  """

  cleantext = helpers.cleantext(intext)
  intoks = helpers.allngrams(cleantext, 1, 3)
  dicescored = [calcdice(curdict, intoks) for curdict in inspace]
  bestmatch = sorted(dicescored, key=lambda d: d['Dice'])
  return bestmatch[-1]['Clause Text']

def addtoks(indict): # type: ({}) -> {}
  """
  Add a list of tokens to a dict including Clause Text

  :param indict: the dict
  :type indict: {}

  :return: the dict with tokens added
  :rtype: {}
  """

  if 'Clause Text' not in indict:
    return indict

  cleantext = helpers.cleantext(indict['Clause Text'])
  indict['Tokens'] = helpers.allngrams(cleantext, 1, 3)

  return indict

#TODO: Read this from a config file
def buildbertargs(): # type: () -> ClassificationArgs
  """
  Builds arguments for the classifier.  Must be the same for
  training and predicting.  Duplicated from the dteulaml
  module.

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

def loadmodels():  # type: () -> None
  """
  Loads in-memory resources for classification and search
  """

  global accsearch, unaccsearch, eulamodel

  accsearch = [row for row in helpers.accExamples if helpers.goodsize(row['Clause Text'])]
  accsearch = [addtoks(row) for row in accsearch]
  unaccsearch = [row for row in helpers.unaccExamples if helpers.goodsize(row['Clause Text'])]
  unaccsearch = [addtoks(row) for row in unaccsearch]
  modeldir = helpers.getmodelfolder()
  accargs = buildbertargs()
  accargs.output_dir = modeldir
  eulamodel = ClassificationModel('roberta', modeldir, args=accargs, weight=[2, 1], use_cuda=False)

#TODO: Possibly try to identify non-clause text and omit it?
#TODO: Possibly take any clause > XXX length and force-split it?
def extractClauses(intext): # type: (str) -> []
  """
  Breaks a text into clauses if such can be identified.  Returns
  line-by-line if not.  Currently not lossy, so all non-empty
  lines of text must be in a clause of some sort.

  :param intext: the full text, presumably from a EULA
  :type intext: str

  :return: the clauses found in the text
  :rtype: []
  """

  lines = intext.splitlines()
  lines = [helpers.cleantext(line) for line in lines]
  numheaders = sum(1 for line in lines if helpers.clausestarter(line))
  if numheaders == 0:
    return lines
  if numheaders < 5 and (numheaders / len(lines)) < 0.1:
    return lines

  # simulate a little state machine
  clauses = []
  curclause = ''
  inclause = True
  for line in lines:
    # new clause starts with A. or ii or 3.4, etc.
    if helpers.clausestarter(line):
      if inclause:
        if len(curclause.strip()) > 0:
          clauses.append(curclause)
      curclause = line
      inclause = True
    # blank line indicates separation between clauses
    elif len(line.strip()) == 0:
      if inclause:
        if len(curclause.strip()) > 0:
          clauses.append(curclause)
      curclause = ''
      inclause = False
    # non-blank line - if we're in a clause, add it to the text, otherwise start a new clause
    else:
      if inclause:
        curclause += ' ' + line
      else:
        curclause = line
        inclause = True
  # clean up the last clause
  if inclause and len(curclause.strip()) > 0:
    clauses.append(curclause)

  clauses = [helpers.cleantext(clause) for clause in clauses]

  return clauses

def extractPDFClauses(intext): # type: (str) -> []
  """
  Breaks a text into clauses if such can be identified.  Returns
  line-by-line if not.  Currently not lossy, so all non-empty
  lines of text must be in a clause of some sort.  Tailored to
  text extracted from a PDF file which includes many more newlines
  than a simple copy-paste.

  :param intext: the full text, presumably from a PDF EULA
  :type intext: str

  :return: the clauses found in the text
  :rtype: []
  """

  lines = intext.splitlines()
  lines = [helpers.cleantext(line) for line in lines]
  # If there aren't enough potential clause headers, just return each line
  numheaders = sum(1 for line in lines if helpers.pdfclausestarter(line))
  if numheaders == 0:
    return lines
  if numheaders < 5 and (numheaders / len(lines)) < 0.1:
    return lines

  # simulate a little state machine
  clauses = []
  curclause = ''
  inclause = True
  for line in lines:
    # new clause starts with A. or ii or 3.4, etc.
    if helpers.pdfclausestarter(line):
      if inclause:
        if len(curclause.strip()) > 0:
          clauses.append(curclause)
      curclause = line + ' '
      inclause = True
    # if we're in a clause, add it to the text, otherwise start a new clause
    else:
      if inclause:
        curclause += line
      else:
        curclause = line
        inclause = True
  if inclause and len(curclause.strip()) > 0:
    clauses.append(curclause)

  clauses = [helpers.cleantext(clause) for clause in clauses]

  return clauses

def parsepdf(intext): # type: (str) -> str
  """
  Accepts base64 encoded PDF content, returns a plain text extraction

  :param intext: the base64 encoded PDF
  :type intext: str

  :return: plain text from the PDF
  :rtype: str
  """

  pdfbinarydata = base64.b64decode(intext.strip())
  pdfFileObj = io.BytesIO()
  pdfFileObj.write(pdfbinarydata)
  pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
  extractedText = ''
  for i in range(0, pdfReader.numPages):
    pageObj = pdfReader.getPage(i)
    extractedText = extractedText + pageObj.extractText()

  return extractedText.strip()

def parseword(intext): # type: (str) -> str
  """
  Accepts base64 encoded Word content, returns a plain text extraction

  :param intext: the base64 encoded Word
  :type intext: str

  :return: plain text from the Word
  :rtype: str
  """

  wordbinarydata = base64.b64decode(intext.strip())
  wordFileObj = io.BytesIO()
  wordFileObj.write(wordbinarydata)
  theword = docx.Document(wordFileObj)
  extractedText = ''
  for para in theword.paragraphs:
    extractedText = extractedText + para.text + '\n'

  return extractedText

def predicteula(inclauses): # type: ([]) -> {}
  """
  Predicts if texts represent unacceptable EULA clauses

  :param inclause: the list of clause texts
  :type inclause: []

  :return: the acceptability decision and softmax probabilities for each window for each clause as a dict of lists
  :rtype: {}
  """

  global eulamodel

  if len(inclauses) == 0:
    return {'prediction': [], 'windows': []}

  retdict = {}

  cleanclauses = [helpers.cleantext(curclause) for curclause in inclauses]
  predictions, raw_outputs = eulamodel.predict(cleanclauses)

  probs = [softmax(curoutput, axis=1) for curoutput in raw_outputs]

  retdict['prediction'] = predictions
  retdict['windows'] = [prob.tolist() for prob in probs]

  return retdict

def processClauseText(intext, mtype): # type: (str, str) -> []
  """
  Accepts a text input which is assumed to be newline separated clauses from a EULA.
  Parses from the origin format: text, pdf or word.
  Extracts individual clauses, classifies them as acceptable or unacceptable, finds
  close matches with known acceptable and unacceptable clauses and packages up a
  list of dicts to return.

  :param intext: the text of the clauses or a base64 encoding of a binary format
  :type intext: str
  :param mtype: text, pdf or word
  :type mtype: str

  :return: classifications and close matches of the input clauses
  :rtype: []
  """

  global classifications, accsearch, unaccsearch

  retlist = []
  texts = []
  if mtype == 'text' or mtype == 'txt':
    texts = extractClauses(intext)
  elif 'pdf' in mtype:
    plaintext = parsepdf(intext)
    texts = extractPDFClauses(plaintext)
  elif 'word' in mtype:
    plaintext = parseword(intext)
    texts = extractClauses(plaintext)

  results = predicteula(texts)
  if len(results['prediction']) == 0:
    return retlist

  accs = [[win[1] for win in curwins] for curwins in results['windows']]
  probs = [max(curwins) for curwins in accs]

  for text, prediction in zip(texts, probs):
    if len(text.split()) > 9:
      curclause = {}
      curclause['origclause'] = text
      curclause['classification'] = 'Not Sure'
      if prediction > 0.6:
        curclause['classification'] = 'Acceptable'
      elif prediction < 0.4:
        curclause['classification'] = 'Unacceptable'
      curclause['score'] = prediction
      curclause['accclause'] = dicesearch(text, accsearch)
      curclause['unaccclause'] = dicesearch(text, unaccsearch)
      retlist.append(curclause)

  return retlist

def batch(infolder, outfile): # type: (str, str) -> None
  """
  Processes each file in a folder, non-recursively.  If the file is
  .txt, .docx or .pdf, attempts to extract clauses and classify as
  acceptable or unacceptable.

  :param infolder: a location to find files in
  :type infolder: str
  :param outfile: a JSON file path to write results to
  :type outfile: str
  """

  if not os.path.isdir(infolder):
    return

  results = []

  for filename in os.listdir(infolder):
    print('Processing ' + filename)
    curresults = []
    if filename.endswith('.txt'):
      with open(os.path.join(infolder, filename), 'r') as curfile:
        curdata = curfile.read() + '\n'
        curresults = processClauseText(curdata, 'text')
    elif filename.endswith('.pdf'):
      with open(os.path.join(infolder, filename), 'rb') as curfile:
        curdata = base64.b64encode(curfile.read()).decode()
        curresults = processClauseText(curdata, 'pdf')
    elif filename.endswith('.docx'):
      with open(os.path.join(infolder, filename), 'rb') as curfile:
        curdata = base64.b64encode(curfile.read()).decode()
        curresults = processClauseText(curdata, 'word')
    if len(curresults) > 0:
      for result in curresults:
        result['filename'] = filename
      results.extend(curresults)

  if outfile is not None:
    with open(outfile, 'w') as outfile:
      json.dump(results, outfile, indent=2)

def dealwithfeedback(info): # type: ({}) -> None
  """
  Register a user feedback event from the app

  :param info: the information about what the user said
  :type info: {}
  """

  global feedback

  feedback.append(info)

  fileexists = os.path.isfile('feedback.json')

  if len(feedback) > 100:
    jsonstr = json.dumps(feedback, indent=2)
    with open('feedback.json', 'a+') as outfile:
      if fileexists:
        outfile.write(',\n' + jsonstr + '\n')
      if not fileexists:
        outfile.write('[\n')
        outfile.write(jsonstr + '\n')
    feedback = []

def exitfeedback(): # type: () -> None
  """
  Register an exit handler for any leftover feedback
  """

  global feedback

  fileexists = os.path.isfile('feedback.json')

  if len(feedback) > 0:
    jsonstr = json.dumps(feedback, indent=2)
    with open('feedback.json', 'a+') as outfile:
      if fileexists:
        outfile.write(',\n' + jsonstr + ']\n')
      if not fileexists:
        outfile.write('[\n')
        outfile.write(jsonstr + ']\n')
    feedback = []

atexit.register(exitfeedback)
