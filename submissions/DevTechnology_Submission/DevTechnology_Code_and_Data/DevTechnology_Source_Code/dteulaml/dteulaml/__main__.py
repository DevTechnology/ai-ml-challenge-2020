# -*- coding: utf-8 -*-
""" dteulaml module

Run --help for more specific instructions

Version:
--------
- dteulaml v0.0.1
"""

# Standard library imports
import logging
import argparse
import sys
import os

# 3rd-party imports
import numpy as np

# dteulaml imports
import dteulaml.core as DTEULAML
from . import helpers

# python -m dteulaml --operation textanalyze --output c:\data\temp
# python -m dteulaml --operation ngrampred --output c:\data\temp
# python -m dteulaml --operation roberta
# python -m dteulaml --operation abc

def main(): # type: () -> None
  logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.ERROR)
  leParser = argparse.ArgumentParser()
  leParser.add_argument('--operation', help='What do you want to do? (textanalyze|roberta|predictall|finalmodel|transfer|hyperparam)')
  leParser.add_argument('--output', help='The directory or file you want the results in')
  leParser.add_argument('--input', help='The file with the data you want to process')
  lesArgs = leParser.parse_args()
  if not hasattr(lesArgs, 'operation') or lesArgs.operation is None:
    logging.error('dteulaml needs to know what to do')
    leParser.print_help()
    sys.exit(2)
  if lesArgs.operation == 'conflicts':
    if hasattr(lesArgs, 'output') and os.path.isdir(lesArgs.output):
      DTEULAML.findconflicts(lesArgs.output)
  if lesArgs.operation == 'textanalyze':
    if hasattr(lesArgs, 'output') and os.path.isdir(lesArgs.output):
      DTEULAML.counttextfeatures(lesArgs.output)
  if lesArgs.operation == 'roberta':
    DTEULAML.trainroberta()
  if lesArgs.operation == 'abc':
    if hasattr(lesArgs, 'output') and os.path.isdir(lesArgs.output):
      DTEULAML.testABC(lesArgs.output)
  if lesArgs.operation == 'predictall':
    if hasattr(lesArgs, 'input') and os.path.isfile(lesArgs.input) and hasattr(lesArgs, 'output'):
      modelfolder = helpers.getmodelfolder()
      print('Loading EULA model')
      DTEULAML.loadeulamodel(modelfolder)
      print('Predicting')
      DTEULAML.predictall(lesArgs.input, lesArgs.output)
  if lesArgs.operation == 'transfer':
    DTEULAML.exploretransfer()
  if lesArgs.operation == 'hyperparam':
    DTEULAML.hyperparamsearch()
  if lesArgs.operation == 'finalmodel':
    modelfolder = helpers.getmodelfolder()
    DTEULAML.finalmodel(modelfolder)
  if lesArgs.operation == 'validation':
    modelfolder = helpers.getmodelfolder()
    DTEULAML.loadeulamodel(modelfolder)
    validationdata = helpers.getvalidationfile()
    outputfile = helpers.getvalidationoutput()
    DTEULAML.validation(modelfolder, validationdata, outputfile)

if __name__ == '__main__':
  main()
