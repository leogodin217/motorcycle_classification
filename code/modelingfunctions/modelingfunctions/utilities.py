import pandas as pd
import matplotlib.pyplot as plt


def plot_results(results):
    '''
    Plots the results from modeling.train_model
    results: Dict returned from mdoeling.train_model. Must contain the results
             of training and the name of the model
    '''
    data = pd.DataFrame(results['results'])
    data = data[['epoch', 'train_loss', 'val_loss', 'test_accuracy']]
    data = data.set_index('epoch')
    data.plot(title=results['name'])


def plot_accuracy(results):
    '''
    Plots the accuracy, topk3 accuracy and topk5 accuracy results from
    modeling.train_model
    results: Dict returned from mdoeling.train_model. Must contain the results
             of training and the name of the model
    '''
    data = pd.DataFrame(results['results'])
    data = data[['epoch', 'test_accuracy', 'test_accuracy_topk']]
    data = data.set_index('epoch')
    data.plot(title=results['name'])
