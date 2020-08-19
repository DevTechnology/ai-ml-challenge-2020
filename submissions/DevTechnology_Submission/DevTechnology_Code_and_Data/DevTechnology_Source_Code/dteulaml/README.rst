*******************************************
Dev Technology EULA Clause Machine Learning
*******************************************

Python code for analyzing and building models of EULA clauses to assist the GSA in
determining if they are acceptable under Federal regulations.

++++++++++++++++++++
Technical Approach
++++++++++++++++++++

We implement several functions to characterize the GSA-provided clause dataset,
build and evaluate models and produce the desired validation data file.  Many
of these functions assume the structure of the git repository, as we do not
intend to use this module generally, and the structure of the repository was
given to us by the GSA.

The core classification model uses the RoBERTa deep learning language model
via the SimpleTransformers library.  Label weighting is used to compensate
for the unbalanced nature of the dataset (19/81 unacceptable/accpetable).
After hyperparamaterization exploration, we finalized on a 5-epoch training
run with unacceptable labels weighted 2 times acceptable.  We used a text
window of 256 tokens with a 10% overlap between windows (stride = 0.9).
This resulted in a model that has a 0.91 F1 Score and a 0.034 Brier loss.

Clause acceptability is a 'zero tolerance' classification problem - it only takes one
unacceptable statement within a clause to render it unacceptable.  No amount of
additional acceptable material can change this.  Using this form of encoding to
classify potentially very long texts requires passing a fixed-size window over the
text and developing a classification for that window.  The traditional method for
whole text classification with windows is to take the mode of the window labels as
the classification of the entire text.  Because of the zero-tolerance nature of
clause unacceptability, it would seem that any single window that has an unacceptable
label should cause the entire clause to be unacceptable.  In practice, when we
implemented this strategy, we did not see an improvement in classification.

Further details on metrics and results are available in our Submission to the GSA

++++++++++++++++++++
Mechanics
++++++++++++++++++++

It is very likely that this package will be in a directory with an absolute path of
over 160 characters.  Python can get unhappy about this.  We recommend that to use a
virtual environment for this package, that the virtual environment be created somewhere
else in a shorter pathed directory, such as /home/ubuntu/venvs/mlenv.

To install dependencies using pip and requirements.txt, use the following command:

    pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

Running the module at the command line requires between 1 and 3 arguments:
--operation is always required.  The next word is the name of something to do
--input is sometimes required when it is not obvious where the data shoud come from
--ouput is sometimes required when it is not obvious where the data should go

To analyze the text found in GSA-provided training clauses in the resources/sample_clauses.csv file:

    python -m  dteulaml --operation textanalyze --output /home/ubuntu/data

This will cause a file called 'textanalysis.csv' to be written to the output folder.
It will also show a couple histograms.

To detect identical clauses with different labels:

    python -m dteulaml --operation conflicts --output /home/ubuntu/data

This will cause a file called 'conflicts.csv' to be written to the output folder.

To train a RoBERTa model extended with a layer for the clause labels:

    python -m dteulaml --operation trainroberta

The parameters of the training can be modified in the buildbertargs() function.
Output will be written to the outputs/ folder in the execution folder.  Additional
cache and runs folders may be created.

To predict acceptability for clauses in a CSV data file:

    python -m dteulaml --operation predictall --input /home/ubuntu/clauses.csv --output /home/ubuntu/results.csv

The input file must have a column called 'Clause Text'.  If it also has a column called
'Classification,' a precision / recall curve and metrics will be shown.  The results CSV
file will contain all the input columns plus prediction and probaility acceptable columns
as well as how many text windows the clause took and how many of those were unacceptable
or acceptable.

To explore different hyperparameters:

    python -m dteulaml --operation hyperparam

Metrics for each combination of parameters will be written to standard out.
To change the ranges or actual parameters used, modify the hyperargs() function.

To reproduce the final model that can be found in the Compiled Models folder:

    python -m finalmodel

To reproduce the validation output data file that can be found in our
submission folder:

    python -m validation

To build and run our adapted Bayesian classifier on the clauses:

    python -m dteulaml --operation abc --output /home/ubuntu/data

This will cause a file called 'abcout.csv' to be written to the output folder.
While we did not end up using this technique for our final submission, the code
is here for potential future extension.

To analyze the transferred Section 508 compliance model:

    python -m dteulaml --operation transfer

The plain text contents and labels of the pickle file will be written to standard out.

To run tests, best to make sure coverage and unittest are installed, then from the
parent directory, run

    coverage run --source dteulaml -m unittest tests.test_basic

`Dev Technology Group <https://www.devtechnology.com>`_
