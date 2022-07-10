# Predict Customer Churn
Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Try to identify credit card customers that are most likely to churn.

## Files and data description
`churn_library.py`: main script that runs the training & reporting

`models/`: models saved during the trainings

`images/eda`: Analysis plots on the data

`images/results`: Result data from the training

## Running Files
Some libraries might need to be installed in your environment in order to run the main script (`churn_library.py`). Refer to requirements_pyxxx.txt to learn about which libraries are needed.( Or call `pip install -r requirements_pyxxx.txt` to install the libraries)

Use autopep8 via command lines in case you made some changes and want to make sure they
follows pep8-guidelines:

`autopep8 --in-place --aggressive --aggressive churn_library.py`

To test the code you could run the test script, either:
`ipython churn_script_logging_and_tests.py` 

or using pytest:
`pytest churn_script_logging_and_tests.py`

