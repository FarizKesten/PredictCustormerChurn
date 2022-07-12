'''
Test scripts to test the functionalities of churn-libraries
Author: Fariz Kesten
Date: 12.07.2022
'''
import logging
import logging.config
import os
from math import ceil
from churn_library import (encoder_helper, import_data, perform_eda,
                           perform_feature_engineering, train_models)

logging.basicConfig(
    filename='./logs/test_logging.log',
    encoding='utf-8',
    level=logging.DEBUG)
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})
logger = logging.getLogger("<churn_script_logging_and_tests>")


def helper_check_file(test_name, file_pth):
    '''
    helper function to test if a file exist
    '''
    try:
        assert os.path.exists(file_pth) is True
        logger.info("Testing %s: %s was found", test_name, os.path.basename(file_pth))
    except AssertionError as err:
        logger.error("Testing %s: %s was not found", test_name, os.path.basename(file_pth))
        raise err


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert all(x > 0 for x in data.shape) is True
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    data = import_data("./data/bank_data.csv")
    try:
        perform_eda(data)
        logger.info("Testing perform_eda: SUCCESS")
    except KeyError as err:
        logger.error("Testing perform_eda:ERROR %s", err)

    files = ["churn_histogram.png", "customer_age_histogram.png",
             "marital_status.png", "total_trans_density.png"]
    for file in files:
        helper_check_file("perfom_eda", "images/eda/" + file)


def test_encoder_helper():
    '''
    test encoder helper
    '''
    data = import_data("./data/bank_data.csv")
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_data = encoder_helper(data, cat_columns, response=None)
        assert encoded_data is not None  # new data is not None
        # new data is different then old data
        assert encoded_data.equals(data) is False
        assert encoded_data.columns.equals(
            data.columns) is True  # columns stay the same
        logger.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logger.error("Testing encoder_helper:FAILED %s", err)
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    data = import_data("./data/bank_data.csv")
    try:
        x_trn, x_tst, y_trn, y_tst = perform_feature_engineering(
            data, response='Churn')
        # all data should not be empty
        assert x_trn is not None
        assert x_tst is not None
        assert y_trn is not None
        assert y_tst is not None
        # test dataset ~30% whole data
        assert (x_tst.shape[0] == ceil(0.3 * data.shape[0])) is True
        logger.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logger.error('Testing perform_feature_engineering: FAILED %s', err)
        raise err


def test_train_models():
    '''
    test train_models
    '''
    bank_data = import_data(r"./data/bank_data.csv")
    x_trn, x_tst, y_trn, y_tst = perform_feature_engineering(bank_data,
                                                             response='Churn')

    train_models(x_trn, x_tst, y_trn, y_tst)
    helper_check_file("train_models", "./images/results/roc.png")
    helper_check_file("train_models", "./models/rfc_model.pkl")
    helper_check_file("train_models", "./models/logistic_model.pkl")
    logger.info("Testing train_models: SUCCESS")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
