import unittest
import sys
import os
import subprocess

# Testing for the notebooks - use nbconvert to execute all cells of the notebook

TestDir = os.path.abspath('../') # where are the notebooks?

def setUp():
    nbpaths = [] #list of notebooks, with file paths
    nbnames = [] # list of notebook names (for making the tests)

    # walk the test directory and find all notebooks
    for dirname, dirnames, filenames in os.walk(TestDir):
        for filename in filenames:
            if filename.endswith('.ipynb') and not filename.endswith('-checkpoint.ipynb'):
                nbpaths.append(os.path.abspath(dirname) + os.path.sep + filename)
                nbnames.append(''.join(filename[:-6]))
    return nbpaths, nbnames

def get(nbname, nbpath):
    def test_func(self):
        print '\nTesting {0}'.format(nbname)
        check = subprocess.call(['jupyter', 'nbconvert', '{0}'.format(nbpath), '--execute'])
        assert check == 0
        subprocess.call(['rm', '{0}.html'.format(nbname)])

    return test_func

attrs = dict()
nbpaths, nbnames = setUp()

# build test for each notebook
for i, nb in enumerate(nbnames):
    attrs['test_'+nb] = get(nb, nbpaths[i])

# create class to unit test notebooks
TestNotebooks = type('TestNotebooks', (unittest.TestCase,), attrs)

if __name__ == '__main__':
    unittest.main()
