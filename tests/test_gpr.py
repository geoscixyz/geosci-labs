import os
import testipynb
import unittest

notebooks = "notebooks/gpr"
TESTDIR = os.path.abspath(__file__)
NBDIR = os.path.sep.join(TESTDIR.split(os.path.sep)[:-2] + [notebooks])

# test the DC notebooks
Test = testipynb.TestNotebooks(directory=NBDIR, timeout=2100)
TestNotebooks = Test.get_tests()

if __name__ == "__main__":
    unittest.main()

