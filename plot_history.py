from pandas import read_pickle
from matplotlib import pyplot as plt

def plot_type(df, acc=True):
    if acc:
        return df[['acc', 'val_acc']].plot()
    else:
        return df[['loss', 'val_loss']].plot()

def plot_history(df, save=False):
    plot_type(df)
    if save:
        plt.savefig('hist_acc.png')
    else:
        plt.show()

    plot_type(df, False)
    if save:
        plt.savefig('hist_loss.png')
    else:
        plt.show()




if __name__ == '__main__':

    path = '/Users/alejandro.robles/PycharmProjects/Thesis/files/files/history2.pkl'
    df = read_pickle(path)
    plot_history(df, save=True)
