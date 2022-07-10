'''
Demo of predicting customer churn
'''
import logging
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/logging.log',
    encoding='utf-8',
    level=logging.DEBUG)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''
    try:
        data = pd.read_csv(pth)
        data['Churn'] = data['Attrition_Flag'].apply(lambda val: 0
                                                     if val == "Existing Customer" else 1)
        return data
    except FileNotFoundError:
        logging.error('%s cannot be read', pth)


def perform_eda(data):
    '''
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''

    save_plot(data['Churn'].hist(),
              "images/eda/churn_histogram.png")
    save_plot(data['Customer_Age'].hist(),
              "images/eda/customer_age_histogram.png")
    save_plot(data.Marital_Status.value_counts('normalize').plot(kind='bar'),
              "images/eda/marital_status.png")
    save_plot(sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True),
              "images/eda/total_trans_density.png")
    save_plot(
        sns.heatmap(
            data.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2),
        "images/eda/correlation_heatmap.png")

    return data


def save_plot(axes, pth, width=20, height=10):
    """save plot to a dedicated path

    Args:
        axes(matplotlib.AxesSubplot)): Axis of a plot
        path (string): Path of the image to be saved
        width (int, optional): figure width. Defaults to 20.
        height (int, optional): figure height. Defaults to 10.
    """
    axes.figure.set_figheight(height)
    axes.figure.set_figwidth(width)
    try:
        axes.figure.savefig(pth)
    except PermissionError:
        logging.error('cannot save figure at %s', pth)

    axes.cla()  # clear buffer after saving


def encoder_helper(data, category_lst, response=""):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    encoder_df = data.copy(deep=True)
    for category in category_lst:
        col_lst = []
        mean_groups = data.groupby(category).mean()['Churn']

        for val in data[category]:
            col_lst.append(mean_groups.loc[val])

        encoder_df[category + '_' +
                   response if response else category] = col_lst

    return encoder_df


def perform_feature_engineering(data, response):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Feature DataFrame
    data_y = data['Churn']
    data_x = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    encoded_data = encoder_helper(data, cat_columns, response)
    data_x[keep_cols] = encoded_data[keep_cols]

    return train_test_split(data_x, data_y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    def plot_and_save_report(name, file_name, y_train, y_train_preds,
                             y_test, y_test_preds):
        '''
        save & plot report
        '''
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(name), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(name), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(file_name)
        plt.cla()

    plot_and_save_report(
        'Random Forest Train',
        './images/results/random_forest_results.png',
        y_train,
        y_train_preds_lr,
        y_test,
        y_test_preds_lr)
    plot_and_save_report(
        'Logistic Regression Train',
        './images/results/logistic_results.png',
        y_train,
        y_train_preds_rf,
        y_test,
        y_test_preds_rf)


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar")
    plt.savefig(os.path.join(output_pth, "summary_plot.png"))
    plt.cla()

    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [x_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))

    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(os.path.join(output_pth, "feature_importance.png"))
    plt.cla()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Do a grid-search to find the best hyperparameters for rfc
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # compute and save ROC-curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                   x_test, y_test,
                   ax=axis, alpha=0.8)
    plot_roc_curve(lrc, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc.png')
    plt.cla()

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)
    feature_importance_plot(cv_rfc, x_test, './images/results/')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    bank_df = import_data(r"./data/bank_data.csv")
    perform_eda(bank_df)

    x_trn, x_tst, y_trn, y_tst = perform_feature_engineering(
        bank_df, response='Churn')
    train_models(x_trn, x_tst, y_trn, y_tst)
