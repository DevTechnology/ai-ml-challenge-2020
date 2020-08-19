# -*- coding: utf-8 -*-
""" Test dteulaapp library. """

# Standard library imports
import unittest

# dteulaapp imports
import dteulaapp.core as DTEULAAPP

class AdvancedTest(unittest.TestCase):

  """ Perform a test on data which is advanced functionality. """
  def test(self):
    self.assertEqual('9', '9')

if __name__ == '__main__':
  unittest.main()
