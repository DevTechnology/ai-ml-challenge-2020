# -*- coding: utf-8 -*-
""" Test dteulaml library. """

# Standard library imports
import unittest

# dteulaml imports
import dteulaml.core as DTEULAML
import dteulaml.helpers as helpers

class SegmentTest(unittest.TestCase):
  """ Does text segmentation work? """
  def test(self):
    segs = helpers.segmenttext('5.1 on its own, or permit any third party to, release or publicly post (including to foreign partner)')
    self.assertEqual(len(segs), 4, 'There should be 4 segments in the text.')

class NGramTest(unittest.TestCase):
  """ Does n-gram extraction work? """
  def test(self):
    ngrams = helpers.allngrams('5.1 on its own, or permit any third party to, release or publicly post (including to foreign partner)', 1, 3)
    self.assertEqual(len(ngrams), 40, 'There should be 40 n-grams in the text.')

if __name__ == '__main__':
  unittest.main()
