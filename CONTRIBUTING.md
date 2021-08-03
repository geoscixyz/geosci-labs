# Contributing to geosci-labs

:tada: **First of all, thank you for considering contributing to this project!** :tada:

This is a community-driven project, so it's people like you that make it useful and successful.
There are a number of ways fo contribute:

* :bug: Submitting bug reports and feature requests
* :mag: Fixing typo's or explinations of how to use the apps
* :memo: Writing new notebook apps
* :bulb: Improving the underlying code or writing new code

If you get stuck at any point you can create an issue on GitHub (look for the
[*Issues*](https://github.com/geoscixyz/geosci-labs/issues) tab in the repository)
 or contact us at one of the other channels mentioned below.

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.
Also, checkout the
[Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software
contributions.

## Ground Rules

The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**.
Everyone must abide by our [Code of Conduct](CODE_OF_CONDUCT.md) and we encourage all to
read it carefully.

## Contents

* [What Can I Do?](#what-can-i-do)
* [How Can I Talk to You?](#how-can-i-talk-to-you)
* [Reporting a Bug](#reporting-a-bug)
* [Contributing to the notebooks and code](#contributing-to-the-notebooks-and-code)
  - [General guidelines](#general-guidelines)
  - [Setting up your environment](#setting-up-your-environment)
  - [Notebook structure](#notebook-outline)
  - [Creating a new app](#creating-a-new-app)
  - [Code style](#code-style)
  - [Testing](#testing)
  - [Peer Review](#peer-review)

## What Can I Do?

* Tackle any issue that you wish! Some issues are labeled as **"good first issues"** to
  indicate that they are beginner friendly, meaning that they don't require extensive
  knowledge of the project.
* Provide feedback about how we can improve the project or about your particular use
  case.
* Contribute code or notebooks you already have. It doesn't need to be perfect! We will help you
  clean things up, test it, etc.

 ## How Can I Talk to You?

Discussion often happens in the issues and pull requests.
In addition, there is a [Slack chat room](http://slack.geosci.xyz) for the
GeoSci.xyz project where you can ask questions.


## Reporting bugs or typos

Find the *Issues* tab on the top of the Github repository and click *New Issue*.
You'll be prompted to choose between different types of issue, like bug reports and
feature requests.
Choose the one that best matches your need.
The Issue will be populated with one of our templates.
**Please try to fillout the template with as much detail as you can**.
Remember: the more information we have, the easier it will be for us to solve your
problem.

## Contributing to the notebooks and code

If you are browsing through the notebooks and notice a typo or an explination that could be
improved, please consider letting us [creating an issue](#reporting-a-bug) or
submitting a fix (even better :star2:).

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests (don't
hesitate to [ask questions](#how-can-i-talk-to-you)):

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).
* Aaron Meurer's [tutorial on the git workflow](http://www.asmeurer.com/git-workflow/)
* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)

### General guidelines

We follow the [git pull request workflow](http://www.asmeurer.com/git-workflow/) to
make changes to our codebase.
Every change made goes through a pull request, even our own, so that our
[continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) services
have a change to check that the code is up to standards and passes all our tests.
This way, the *main* branch is always stable.

General guidelines for pull requests (PRs):

* **Open an issue first** describing what you want to do. If there is already an issue
  that matches your PR, leave a comment there instead to let us know what you plan to
  do.
* Each pull request should consist of a **small** and logical collection of changes.
* Larger changes should be broken down into smaller components and integrated
  separately.
* Bug fixes should be submitted in separate PRs.
* Describe what your PR changes and *why* this is a good thing. Be as specific as you
  can. The PR description is how we keep track of the changes made to the project over
  time.
* Do not commit changes to files that are irrelevant to your feature or bugfix (eg:
  `.gitignore`, IDE project files, etc).
* Write descriptive commit messages. Chris Beams has written a
  [guide](https://chris.beams.io/posts/git-commit/) on how to write good commit
  messages.
* Be willing to accept criticism and work on improving your code; we don't want to break
  other users' code, so care must be taken not to introduce bugs.
* Be aware that the pull request review process is not immediate, and is generally
  proportional to the size of the pull request.

### Setting up your environment

We highly recommend using [Anaconda](https://www.anaconda.com/download/) and the `conda`
package manager to install and manage your Python packages.
It will make your life a lot easier!

The repository includes a conda environment file `environment-dev.yml` with the
specification for all development requirements to build and test the project.
Once you have forked and clone the repository to your local machine, you use this file
to create an isolated environment on which you can work.
Run the following on the base of the repository:

```bash
conda env create
```

Before building and testing the project, you have to activate the environment:

```bash
conda activate geosci-labs-dev
```

You'll need to do this every time you start a new terminal.

See the [`environment-dev.yml`](environment-dev.yml) file for the list of dependencies and the
environment name.

We have a [`Makefile`](Makefile) that provides commands for installing, running the
tests and coverage analysis, running linters, etc.
If you don't want to use `make`, open the `Makefile` and copy the commands you want to
run.

To install the current source code into your testing environment, run:

```bash
make install
```

This installs your project in *editable* mode, meaning that changes made to the source
code will be available when you import the package (even if you're on a different
directory).

### Notebook structure

To maintain consistency of the notebook-apps, please use the following structure

- **Title**: title of the notebook that include the geophysical method name
- **Purpose**: Motivation and key concepts to be addressed in this notebook. Can include links to relevant background material (e.g. from https://gpg.geosci.xyz or https://em.geosci.xyz)
- **Setup**: Overview of the relevant parameters in the problem
- **App**: interactive visualization that uses [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)
- (**Additional material**: supporting derivations, links to related case studies, ...)
- **Acknowledgement**: please include the below statement a markdown cell at the end of the notebook, appending any other geoscience
  package names that are being used in the notebook and should be acknowledged:
  ```
  # Acknowledgements

  This app is a part of the [GeoSci.xyz](https://geosci.xyz) project and relies on [SimPEG](https://simpeg.xyz). Thanks to all of the [contributors](https://github.com/geoscixyz/geosci-labs/graphs/contributors) for their work!

  # License

  <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>
  ```

Prior to creating a pull request with a new notebook, please make sure you restart the kernel and clear output so that it always starts from a clean state.

If you create a new notebook, please add it to the list in [index.ipynb](notebooks/index.ipynb)

### Conventions


**For colormaps** (http://matplotlib.org/examples/color/colormaps_reference.html)
- **fields** are plotted with the `viridis`
- **potentials** are plotted with `viridis`
- **sensitivities** are plotted with `viridis`
- **physical properties** are plotted with `jet`
- **charges** are plotted with `RdBu`

**Order of widgets:**

- geometry of survey
- geomerty target
- physical properties of target
- view options

### Code style

We use [Black](https://github.com/ambv/black) to format the code so we don't have to
think about it.
Black loosely follows the [PEP8](http://pep8.org) guide but with a few differences.
Regardless, you won't have to worry about formatting the code yourself.
Before committing, run it to automatically format your code:

```bash
make format
```

Don't worry if you forget to do it.
Our continuous integration systems will warn us and you can make a new commit with the
formatted code.

We also use [flake8](http://flake8.pycqa.org/en/latest/) to check the quality of the code and quickly catch
common errors.
The [`Makefile`](Makefile) contains rules for running this check:

```bash
make check   # Runs flake8 and black (in check mode)
```

### Testing

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of the code.

All of our test code and data are stored in the `tests` subpackage.
We use the [pytest](https://pytest.org/) framework to run the test suite.

Run the tests and calculate test coverage using:

```bash
make test
```

### Code Review

After you've submitted a pull request, you should expect to hear at least a comment
within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write a good and detailed description of what the PR does.
* Write tests for the code you wrote/modified.
* Readable code is better than clever code (even with comments).
* Write documentation for your code (docstrings) and leave comments explaining the
  *reason* behind non-obvious things.
* Include an example of new features in the gallery or tutorials.
* Follow the [PEP8](http://pep8.org) style guide for code

Pull requests will automatically have tests run by TravisCI.
This includes running both the unit tests as well as code linters.
Github will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment in the PR or
[get in touch](#how-can-i-talk-to-you).

 ## Credit

 This contributor guide is adapted from the [Contributor guide in the Fatiando a Terra project](https://github.com/fatiando/verde/blob/master/CONTRIBUTING.md).
