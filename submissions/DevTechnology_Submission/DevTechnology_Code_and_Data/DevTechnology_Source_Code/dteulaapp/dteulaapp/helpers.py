# -*- coding: utf-8 -*-

# Standard library imports
import os
import csv
import re
import unicodedata

refData = []
ddData = []
accExamples = []
unaccExamples = []

# Initialization loads reference data from resources CSV.
this_dir, this_filename = os.path.split(__file__)
refPath = os.path.join(this_dir, 'resources', 'sampleclauses.csv')
if (os.path.isfile(refPath)):
  with open(refPath, encoding='utf8') as refFile:
    refData = [{colName: str(cellValue) for colName, cellValue in row.items()}
               for row in csv.DictReader(refFile, skipinitialspace=True)]
dedupepath = os.path.join(this_dir, 'resources', 'deduped.csv')
if (os.path.isfile(dedupepath)):
  with open(dedupepath, encoding='utf8') as ddFile:
    ddData = [{colName: str(cellValue) for colName, cellValue in row.items()}
               for row in csv.DictReader(ddFile, skipinitialspace=True)]
    accExamples = [posDict for posDict in ddData if posDict['Classification'] == '0' and len(posDict['Clause Text'].split()) > 9]
    unaccExamples = [negDict for negDict in ddData if negDict['Classification'] == '1' and len(negDict['Clause Text'].split()) > 9]

clauseprefixes = re.compile(r"([A-Z]|[a-z]|[0-9.]{1,4}|[ivx]{2,3})\.?[ \t]")
pdfclauseprefixes = re.compile(r"^([A-Z]\.|[a-z]\.|[0-9]{1,2}(\.[0-9])?|[ivx]{2,3}\.?)$")
crappyspaces = re.compile(r"[\n\r\t\v]+")
multispaces = re.compile(r"  +")
spaceadjacentpunct = re.compile(r"( +[.,;:(){}\[\]<>\"'~!@#$%^&*|\\?_+\-]|[.,;:(){}\[\]<>\"'~!@#$%^&*|\\?_+\-] +)")
noalphanum = re.compile(r"^[^a-zA-Z0-9]+$")

def getresourcefolder(): # type: () -> str
  """
  Returns the path of the folder that module resources
  are stored in.
  """

  this_dir, this_filename = os.path.split(__file__)
  resourcePath = os.path.join(this_dir, 'resources')

  return resourcePath

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

def clausestarter(intext): # type: (str) -> bool
  """
  Determines if the line looks like it starts a clause

  :param intext: a line, presumably from a EULA
  :type intext: str

  :return: does this line start a clause?
  :rtype: bool
  """

  return clauseprefixes.match(intext) is not None

def pdfclausestarter(intext): # type: (str) -> bool
  """
  Determines if the line looks like it starts a clause
  from a PDF extraction - likely to be the only thing
  on the line.

  :param intext: a line, presumably from a PDF EULA
  :type intext: str

  :return: does this line start a clause?
  :rtype: bool
  """

  return pdfclauseprefixes.match(intext.strip()) is not None

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
  Does not preserve duplicates.

  Example input:
  '5.1 on its own, or permit any third party to,'
  Example output (1, 2 and 3 grams):
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
  Determines if the clause is a good size for training.

  :param instr: the clause text
  :type instr: str

  :return: is the clause a good size?
  :rtype: bool
  """

  toks = len(instr.split())

  return toks > 10 and toks < 500
