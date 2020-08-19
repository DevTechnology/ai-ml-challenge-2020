# -*- coding: utf-8 -*-

# Standard library imports
import os
import csv
import unicodedata
import re

# 3-rd party imports
import pandas as pd

filenames = {}
filenames['textanalysis'] = 'textanalysis.csv'
filenames['conflicts'] = 'conflicts.csv'
filenames['ngrampred'] = 'ngrampred.csv'
filenames['clauseprob'] = 'clauseprob.csv'
filenames['deduped'] = 'deduped.csv'
filenames['abcoutput'] = 'abcout.csv'

headerlists = {}
headerlists['base'] = ['Clause ID', 'Clause Text', 'Classification']
headerlists['textanalysis'] = ['Clause ID', 'Clause Text', 'Classification', 'Chars', 'Tokens', 'Allcaps']
headerlists['conflicts'] = ['Acceptable Clause ID', 'Unacceptable Clause ID', 'Similarity', 'Acceptable Len', 'Unacceptable Len']
headerlists['ngrampred'] = ['ngram', 'uses', 'probBA', 'probB', 'probA', 'probAB']
headerlists['clauseprob'] = ['Clause ID', 'Clause Text', 'Classification', 'Tokens', 'Prob Sum']

refdf = pd.DataFrame()
dedupdf = pd.DataFrame()
refData = []
dedupData = []
unaccExamples = []
accExamples = []

crappyspaces = re.compile(r"[\n\r\t\v]+")
multispaces = re.compile(r"  +")

spaceadjacentpunct = re.compile(r"( +[.,;:(){}\[\]<>\"'~!@#$%^&*|\\?_+\-]|[.,;:(){}\[\]<>\"'~!@#$%^&*|\\?_+\-] +)")
noalphanum = re.compile(r"^[^a-zA-Z0-9]+$")

def cleantext(intext): # type (str) -> str
  """
  Cleans up certain patterns in a text, presumed to be
  from a EULA.  Simplifies some Unicode, deals with white
  space, etc.

  :param intext: the potentially dirty text
  :type intext: str

  :return: a cleaner version of the text
  :rtype: str
  """

  rettext = unicodedata.normalize('NFKD', intext)
  rettext = rettext.replace(u'\u201c', '"').replace(u'\u201d', '"')
  rettext = rettext.replace(u'\u2018', "'").replace(u'\u2019', "'")
  rettext = crappyspaces.sub(' ', rettext)
  rettext = multispaces.sub(' ', rettext)

  return rettext

# Initialization loads reference data from resources CSV into both a DataFrame and a list of dicts
this_dir, this_filename = os.path.split(__file__)
refPath = os.path.join(this_dir, 'resources', 'sampleclauses.csv')
if (os.path.isfile(refPath)):
  refdf = pd.read_csv(refPath, encoding='utf8', converters={'Clause Text': cleantext})
  with open(refPath, encoding='utf8') as refFile:
    refData = [{colName: str(cellValue) for colName, cellValue in row.items()}
               for row in csv.DictReader(refFile, skipinitialspace=True)]
  refData = [row for row in refData if 'Classification' in row and row['Classification'] in ['1', '0']]
  for curdict in refData:
    curdict['Clause Text'] = cleantext(curdict['Clause Text'])
  unaccExamples = [curdict for curdict in refData if curdict['Classification'] == '1']
  accExamples = [curdict for curdict in refData if curdict['Classification'] == '0']

# Initialization loads deduplicated data from resources CSV into both a DataFrame and a list of dicts
this_dir, this_filename = os.path.split(__file__)
dedupPath = os.path.join(this_dir, 'resources', 'deduped.csv')
if (os.path.isfile(dedupPath)):
  dedupdf = pd.read_csv(dedupPath, encoding='utf8', converters={'Clause Text': cleantext})
  with open(dedupPath, encoding='utf8') as dedupFile:
    dedupData = [{colName: str(cellValue) for colName, cellValue in row.items()}
               for row in csv.DictReader(dedupFile, skipinitialspace=True)]
  dedupData = [row for row in dedupData if 'Classification' in row and row['Classification'] in ['1', '0']]
  for curdict in refData:
    curdict['Clause Text'] = cleantext(curdict['Clause Text'])

def getmodelfolder(): # type: () -> str
  """
  Returns the path of the folder that the EULA model is
  stored in.  This assumes the well-named structure of
  the git repository.  If that doesn't exist, uses the
  default name of the BERT model output folder.
  """

  this_dir, this_filename = os.path.split(__file__)
  parentfolder = os.path.dirname(this_dir)
  dtsourcefolder = os.path.dirname(parentfolder)
  dtcodedatafolder = os.path.dirname(dtsourcefolder)
  modelfolder = os.path.join(dtcodedatafolder, 'DevTechnology_Compiled_Models', 'eulabert')
  if os.path.isdir(modelfolder):
    return modelfolder

  return 'outputs'

def getvalidationfile(): # type: () -> str
  """
  Returns the path of the file with validation data
  in it.  This assumes the well-named structure of
  the git repository.  If that doesn't exist, uses the
  default name of the file in the current folder.
  """

  this_dir, this_filename = os.path.split(__file__)
  parentfolder = os.path.dirname(this_dir)
  dtsourcefolder = os.path.dirname(parentfolder)
  dtcodedatafolder = os.path.dirname(dtsourcefolder)
  dtsubmissionfolder = os.path.dirname(dtcodedatafolder)
  submissionfolder = os.path.dirname(dtsubmissionfolder)
  rootfolder = os.path.dirname(submissionfolder)
  validationfilename = os.path.join(rootfolder, 'data', 'AI_ML_Challenge_Validation_Data_Set_v1.csv')
  if os.path.isfile(validationfilename):
    return validationfilename

  return 'AI_ML_Challenge_Validation_Data_Set_v1.csv'

def gettransferfile(): # type: () -> str
  """
  Returns the path of the file with a model of 508
  compliance.  This assumes the well-named structure of
  the git repository.  If that doesn't exist, uses the
  default name of the file in the current folder.
  """

  this_dir, this_filename = os.path.split(__file__)
  parentfolder = os.path.dirname(this_dir)
  dtsourcefolder = os.path.dirname(parentfolder)
  dtcodedatafolder = os.path.dirname(dtsourcefolder)
  dtsubmissionfolder = os.path.dirname(dtcodedatafolder)
  submissionfolder = os.path.dirname(dtsubmissionfolder)
  rootfolder = os.path.dirname(submissionfolder)
  transferfilename = os.path.join(rootfolder, 'reference', 'train.pkl')
  if os.path.isfile(transferfilename):
    return transferfilename

  return 'train.pkl'

def getvalidationoutput(): # type: () -> str
  """
  Returns the path of the folder that the EULA model is
  stored in.  This assumes the well-named structure of
  the git repository.  If that doesn't exist, uses the
  default name of the BERT model output folder.
  """

  this_dir, this_filename = os.path.split(__file__)
  parentfolder = os.path.dirname(this_dir)
  dtsourcefolder = os.path.dirname(parentfolder)
  dtcodedatafolder = os.path.dirname(dtsourcefolder)
  dtsubmissionfolder = os.path.dirname(dtcodedatafolder)
  validationoutputfile = os.path.join(dtsubmissionfolder, 'DevTechnology_Validation_Data_File.csv')
  if os.path.isdir(dtsubmissionfolder):
    return validationoutputfile

  return 'DevTechnology_Validation_Data_File.csv'

def dicescore(toks1, toks2): # type: ([], []) -> float
  """
  Calculates Dice score for two lists of tokens.
  Does not modulate for text length or TF-IDF.

  :param toks1: the first list of tokens
  :type toks1: []
  :param toks2: the other list of tokens
  :type toks2: []

  :return: the Dice similarity score
  :rtype: float
  """

  set1 = set(toks1)
  set2 = set(toks2)

  score = (2.0 * len(set1.intersection(set2))) / (len(set1) + len(set2))

  return score

def segmenttext(intext): # type: (str) -> []
  """
  Finds strings of consecutive words separated by punctuation.
  Example input:
  '5.1 on its own, or permit any third party to, release or publicly post (including to foreign partner)'
  Example output:
  ['5.1 on its own', 'or permit any third party to', 'release or publicly post', 'including to foreign partner']

  :param intext: the text to segment
  :type intext: str

  :return: the list of segments
  :rtype: []
  """

  touse = ' ' + intext + ' '
  segs = spaceadjacentpunct.split(touse)
  segs = [seg.strip() for seg in segs if not noalphanum.match(seg) and len(seg) > 0]
  return segs

def allngrams(intext, min, max): # type (str, int, int) -> []
  """
  Extract n-grams of different sizes from the input text.
  Does not cross punctuation.
  Does not preserve duplicate grams.
  Sorts output alphabetically.

  Example input:
  '5.1 on its own, or permit any third party to,'
  Example output prior to sort (1, 2 and 3 grams):
  ['5.1', 'on', 'its', 'own', 'or', 'permit', 'any', 'third', 'party', 'to', '5.1 on', 'on its', 'its own',
   'or permit', 'permit any', 'any third', 'third party', 'party to', '5.1 on its', 'on its own', 'or permit any',
   'permit any third', 'any third party', 'third party to']

  :param intext: the text to extract ngrams from
  :type intext: str

  :return: the list of ngrams
  :rtype: []
  """

  results = set()
  segs = segmenttext(intext)
  for windowsize in range(min, max + 1):
    for seg in segs:
      toks = seg.split()
      if len(toks) >= windowsize:
        for curstartpos in range(0, len(toks) - windowsize + 1):
          curngram = ' '.join(toks[curstartpos:curstartpos + windowsize])
          results.add(curngram)

  retlist = list(results)
  retlist.sort()
  return retlist

def goodsize(instr): # type: (str) -> bool
  """
  Determines if the clause is a good size for training.  The assumption is
  that a text less thna 10 tokens long cannot represent enough information to
  decide if the clause is acceptabke, and thant a token more than 500 tokens
  long is likely to have both acceptable and unacceptable components, so
  training, particularly windowed, cannot be performed against one 'right'
  answer.

  :param instr: the clause text
  :type instr: str

  :return: is the clause a good size?
  :rtype: bool
  """

  toks = len(instr.split())

  return toks > 10 and toks < 500

def writeCSV(data, headers, outpath): # type: ([], [], str) -> None
  """
  Writes data to a CSV file with column headers.

  :param data: the rows to write.  Must be a list of dicts with keys matching headers
  :type data: []
  :param headers: the column headers
  :type headers: []
  :param outpath: the location on disk to write the results:
  :type outpath: str
  """

  with open(outpath, 'w', encoding='utf8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(data)

def writedftocsv(filename, data): # type: (str, pd.DataFrame) -> None
  """
  Writes a data frame to a CSV file

  :param filename: the path to the file
  :type filename: str
  :param data: the DataFrame
  :type data: pd.DataFrame
  """

  data.to_csv(filename, encoding='utf8', index=False, header=True)
