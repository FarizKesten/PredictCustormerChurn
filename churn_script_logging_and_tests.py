import os
import logging
from math import ceil
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = import_data("./data/bank_data.csv")
	try:
		perform_eda(df)
		logging.info("Testing perform_eda: SUCCESS")
	except KeyError as err:
		logging.error("Testing perform_eda:ERROR %s", err)

	files = ["churn_histogram.png", "customer_age_histogram.png",
             "marital_status.png", "total_trans_density.png"]
	for file in files:
		try:
			assert os.path.exists("images/eda/" + file) is True
			logging.info("Testing perform_eda: %s was found", file)
		except AssertionError as err:
			logging.error("Testing perform_eda: %s was not found", file)
			raise err



def test_encoder_helper():
	'''
	test encoder helper
	'''
	df = import_data("./data/bank_data.csv")
	cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

	try:
		encoded_data = encoder_helper(df, cat_columns, response=None)
		assert encoded_data is not None #new data is not None
		assert encoded_data.equals(df) is False # new data is different then old data
		assert encoded_data.columns.equals(df.columns) is True # columns stay the same
		logging.info("Testing encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoder_helper:FAILED %s", err)
		raise err

def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	df = import_data("./data/bank_data.csv")
	try:
		x_trn, x_tst, y_trn, y_tst = perform_feature_engineering(df, response='Churn')
		# all data should not be empty
		assert x_trn is not None
		assert x_tst is not None
		assert y_trn is not None
		assert y_tst is not None
		# test dataset ~30% whole data
		assert (x_tst.shape[0] == ceil(0.3*df.shape[0])) is True
		logging.info("Testing perform_feature_engineering: SUCCESS")
	except AssertionError as err:
		logging.error('Testing perform_feature_engineering: FAILED %s', err)
		raise err



def test_train_models():
	'''
	test train_models
	'''
	bank_df = import_data(r"./data/bank_data.csv")
	x_trn, x_tst, y_trn, y_tst = perform_feature_engineering(bank_df,
                                 response='Churn')

	try:
		train_models(x_trn, x_tst, y_trn, y_tst)
		assert(os.path.exists('./images/results/roc.png')) is True
		assert(os.path.exists('./models/rfc_model.pkl')) is True
		assert(os.path.exists('./models/logistic_model.pkl')) is True
		logging.info("Testing train_models: SUCCESS")
	except AssertionError as err:
		logging.error('file not found %s', err)
		raise err



if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_train_models()