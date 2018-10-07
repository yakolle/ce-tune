import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 10), scoring=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, train_sizes=train_sizes,
                                                            scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 's-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def group_boxplot(df, x_col, y_col, data_dir):
    try:
        fig = plt.figure(figsize=(18, 9))
        sns.boxplot(x=x_col, y=y_col, data=df, palette="Set3", showmeans=True, meanline=True)
        fig.savefig(data_dir + "img\\" + x_col + '_target_box.png', bbox_inches='tight')
    except Exception as e:
        print('----【', x_col, '】----', e)


def box_plot(df, x_col, data_dir, suffix='all', ignore_values=None):
    try:
        fig = plt.figure(figsize=(18, 9))
        if ignore_values is None:
            sns.boxplot(x=x_col, data=df, palette="Set3", showmeans=True, meanline=True)
        else:
            col = df[x_col]
            col = col.loc[~col.isin(ignore_values)]
            sns.boxplot(x=col, palette="Set3", showmeans=True, meanline=True)
        fig.savefig(data_dir + "img\\" + x_col + '_box_' + suffix + '.png', bbox_inches='tight')
    except Exception as e:
        print('----【', x_col, '】----', e)


def count_bar(df, x_col, data_dir, suffix='all'):
    try:
        vals = df[x_col].unique()
        val_types = [type(v).__name__ for v in vals]
        if len(val_types) > 1:
            vals = sorted(vals, key=lambda v: str(v))
        else:
            vals = sorted(vals)

        fig = plt.figure(figsize=(18, 9))
        sns.countplot(x=x_col, data=df, order=vals)
        fig.savefig(data_dir + "img\\" + x_col + '_count_' + suffix + '.png', bbox_inches='tight')
    except Exception as e:
        print('----【', x_col, '】----', e)


def dist_plot(df, x_col, data_dir, suffix='all', ignore_values=None):
    try:
        x = df.loc[df[x_col].notnull(), x_col]
        if ignore_values is not None:
            x = x.loc[~x.isin(ignore_values)]

        fig = plt.figure(figsize=(18, 9))
        sns.distplot(x)
        fig.savefig(data_dir + "img\\" + x_col + '_dist_' + suffix + '.png', bbox_inches='tight')
    except Exception as e:
        print('----【', x_col, '】----', e)


def reg_plot(df, x_col, y_col, data_dir):
    try:
        fig = plt.figure(figsize=(18, 9))
        sns.regplot(x=x_col, y=y_col, data=df)
        fig.savefig(data_dir + "img\\" + x_col + '_reg.png', bbox_inches='tight')
    except Exception as e:
        print('----【', x_col, '】----', e)
