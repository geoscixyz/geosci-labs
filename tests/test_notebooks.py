import unittest
import sys
import os
import subprocess
import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor, ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


# Testing for the notebooks - use nbconvert to execute all cells of the
# notebook

# add names of notebooks to ignore here.
py2Ignore = []

TESTDIR = os.path.abspath(__file__)
NBDIR = os.path.sep.join(
    TESTDIR.split(os.path.sep)[:-3] + ['em_apps/']
) # where are the notebooks?


def setUp():
    nbpaths = []  # list of notebooks, with file paths
    nbnames = []  # list of notebook names (for making the tests)

    # walk the test directory and find all notebooks
    for dirname, dirnames, filenames in os.walk(NBDIR):
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.endswith("-checkpoint.ipynb"):
                nbpaths.append(os.path.abspath(dirname) + os.path.sep + filename) # get abspath of notebook
                nbnames.append("".join(filename[:-6])) # strip off the file extension
    return nbpaths, nbnames


def get(nbname, nbpath):

    # use nbconvert to execute the notebook
    def test_func(self):
        passing = True
        print("\n--------------- Testing {0} ---------------".format(nbname))
        print("   {0}".format(nbpath))

        if nbname in py2Ignore and sys.version_info[0] == 2:
            print(" Skipping {}".format(nbname))
            return

        ep = ClearOutputPreprocessor()

        with open(nbpath) as f:
            nb = nbformat.read(f, as_version=4)

            ep.preprocess(nb, {})

            ex = ExecutePreprocessor(
                timeout=600,
                kernel_name='python{}'.format(sys.version_info[0]),
                allow_errors=True
            )

            out = ex.preprocess(nb, {})

            for cell in out[0]['cells']:
                if 'outputs' in cell.keys():
                    for output in cell['outputs']:
                        if output['output_type'] == 'error':
                            passing = False

                            err_msg = []
                            for o in output['traceback']:
                                err_msg += ["{}".format(o)]
                            err_msg = "\n".join(err_msg)

                            msg = """
\n <<<<< {} FAILED  >>>>> \n
{} in cell [{}] \n-----------\n{}\n-----------\n
----------------- >> begin Traceback << ----------------- \n
{}\n
\n----------------- >> end Traceback << -----------------\n
                            """.format(
                                nbname, output['ename'],
                                cell['execution_count'], cell['source'],
                                err_msg
                            )

                            assert passing, msg

            print("\n ..... {0} Passed ..... \n".format(nbname))
    return test_func


attrs = dict()
nbpaths, nbnames = setUp()

# build test for each notebook
for i, nb in enumerate(nbnames):
    attrs["test_"+nb] = get(nb, nbpaths[i])

# create class to unit test notebooks
TestNotebooks = type("TestNotebooks", (unittest.TestCase,), attrs)


if __name__ == "__main__":
    unittest.main()
