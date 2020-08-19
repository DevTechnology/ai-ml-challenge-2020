# -*- coding: utf-8 -*-
""" dteulaapp module

Run --help for more specific instructions

Version:
--------
- dteulaapp v0.0.1
"""

# Standard library imports
import logging
import argparse
import sys
import os

# dteulaapp imports
import dteulaapp.eulaapp as DTEULAAPP
import dteulaapp.core as EULACORE
from . import create_app

# python -m dteulaapp
def main(): # type: () -> None
  leParser = argparse.ArgumentParser()
  leParser.add_argument('--operation', help='What do you want to do? (app|batch)')
  leParser.add_argument('--input', help='Where should eulaapp look for files? (folder with .txt, .docx, .pdf files)')
  leParser.add_argument('--output', help='Where should eulaapp write results? (.json)')
  lesArgs = leParser.parse_args()
  if not hasattr(lesArgs, 'operation') or lesArgs.operation is None:
    logging.error('dteulaapp needs to know what to do')
    leParser.print_help()
    sys.exit(2)
  if lesArgs.operation == 'app':
    app = create_app()
    app.run(debug=True)
  elif lesArgs.operation == 'batch':
    if not hasattr(lesArgs, 'input') or lesArgs.input is None:
      logging.error('dteulaapp needs to know where to find files to process')
      leParser.print_help()
      sys.exit(2)
    if not os.path.isdir(lesArgs.input):
      logging.error(lesArgs.input + ' doesn\'t seem to be a folder on your computer.')
      leParser.print_help()
      sys.exit(2)
    if not hasattr(lesArgs, 'output') or lesArgs.output is None:
      logging.error('dteulaapp needs to know where to write results')
      leParser.print_help()
      sys.exit(2)
    EULACORE.loadmodels()
    EULACORE.batch(lesArgs.input, lesArgs.output)

if __name__ == '__main__':
  main()
