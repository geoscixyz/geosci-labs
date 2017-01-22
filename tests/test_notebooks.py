import unittest
import sys
import os
import subprocess

# Testing for the notebooks - use nbconvert to execute all cells of the
# notebook


TESTDIR = os.path.abspath(__file__)
NBDIR = os.path.sep.join(TESTDIR.split(os.path.sep)[:-3] + ['em_apps/']) # where are the notebooks?

def setUp():
    nbpaths = []  # list of notebooks, with file paths
    nbnames = []  # list of notebook names (for making the tests)

    # walk the test directory and find all notebooks
    for dirname, dirnames, filenames in os.walk(NBDIR):
        for filename in filenames:
            if filename.endswith('.ipynb') and not filename.endswith('-checkpoint.ipynb'):
                nbpaths.append(os.path.abspath(dirname) + os.path.sep + filename) # get abspath of notebook
                nbnames.append(''.join(filename[:-6])) # strip off the file extension
    return nbpaths, nbnames


def get(nbname, nbpath):

    # use nbconvert to execute the notebook
    def test_func(self):
        print '\n--------------- Testing {0} ---------------'.format(nbname)
        print '   {0}'.format(nbpath)
        nbexe = subprocess.Popen(['jupyter', 'nbconvert', '{0}'.format(nbpath),
                                  '--execute',
                                  '--ExecutePreprocessor.timeout=120'],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        output, err = nbexe.communicate()
        check = nbexe.returncode
        if check == 0:
            print '\n ..... {0} Passed ..... \n'.format(nbname)
        else:
            print '\n <<<<< {0} FAILED >>>>> \n'.format(nbname)
            print 'Captured Output: \n {0}'.format(err)

        self.assertTrue(check == 0)

    return test_func


def tearDown():
    subprocess.call(['find', '{0}'.format(os.path.sep.join(TESTDIR.split(os.path.sep)[:-1])),
                    '-type', 'f', '-name', '*.html', '-delete'])


attrs = dict()
nbpaths, nbnames = setUp()

# build test for each notebook
for i, nb in enumerate(nbnames):
    attrs['test_'+nb] = get(nb, nbpaths[i])

# create class to unit test notebooks
TestNotebooks = type('TestNotebooks', (unittest.TestCase,), attrs)


if __name__ == '__main__':
    unittest.main()
    tearDown()
