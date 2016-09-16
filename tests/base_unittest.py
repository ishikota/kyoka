import unittest

class BaseUnitTest(unittest.TestCase):

  def eq(self, expected, target):
    return self.assertEqual(expected, target)

  def neq(self, expected, target):
    return self.assertNotEqual(expected, target)

  def true(self, target):
    return self.assertTrue(target)

  def false(self, target):
    return self.assertFalse(target)

  def include(self, target, source):
    return self.assertIn(target, source)

  def not_include(self, target, source):
    return self.assertNotIn(target, source)

  def debug(self):
    from nose.tools import set_trace; set_trace()

