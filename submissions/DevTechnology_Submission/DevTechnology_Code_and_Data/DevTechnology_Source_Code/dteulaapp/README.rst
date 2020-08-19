******************************************
DevTechnology App for EULA Clause Analysis
******************************************

This single-page app is the demonstration of Dev Technology's work in the 2020
GSA AI/ML EULA Challenge.  It allows a user to enter EULA clause text and obtain
a judgment of its acceptability with respect to Federal regulations.  It also
offers examples of known acceptable and unacceptable clauses that closely match
the input clause(s).

The app includes both a single-page index.html and a RESTful, JSON-based API for
passing information to and from the page.

The app code also has a batch mode that can be run at the command line to
process all .pdf, .docx or .txt files found in a folder and produce JSON output.

The app is implemented in `Python Flask <https://flask.palletsprojects.com/en/1.1.x/>`_
and is intended to be a small-scale demonstration, not a fully scaled web application.

===========
How to Use
===========

It is very likely that this package will be in a directory with an absolute path of
over 160 characters.  Python can get unhappy about this.  We recommend that to use a
virtual environment for this package, that the virtual environment be created somewhere
else in a shorter pathed directory, such as /home/ubuntu/venvs/mlenv.

To install dependencies using pip and requirements.txt, use the following command:

    pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

This app assumes that the larger structure of the git repository is available to it,
particularly the trained model for classification.  Code modification will be
required to run this app in another context.

To run the app locally:

    python -m dteulaapp --operation app

The app will be accessible at http://127.0.0.1:5000

To run in batch mode:

    python -m dteulaapp --operation batch --input /home/ubuntu/data --output /home/ubuntu/output.json

Each .pdf, .docx or .txt file in the input folder will be processed for EULA clauses.
The output will contain the original txt of each clause, which file it came from, what
our acceptability prediction is, the closest known acceptable clause and the closest
known unacceptable clause.

To run tests, best to make sure coverage and unittest are installed, then from the
parent directory, run

    coverage run --source dteulaapp -m unittest tests.test_basic

`Dev Technology Group EULA Checker <https://eulacheck.devlab-dtg.com>`_
