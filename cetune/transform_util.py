from scipy.special import boxcox1p
from sklearn.linear_model import LinearRegression

from cv_util import *


def calc_best_transform_linear(x, y, lmds, min_x=None, measure_func=metrics.mean_squared_error,
                               mean_std_coeff=(1.0, 1.0), detail=False):
    if min_x is None:
        min_x = min(x)
    x = (x - min_x).to_frame()
    cv_score_means = []
    cv_score_stds = []

    for lmd in lmds:
        cv_scores = bootstrap_k_fold_cv_train(LinearRegression(), boxcox1p(x, lmd), y, measure_func=measure_func)
        cv_score_mean = np.mean(cv_scores)
        cv_score_std = np.std(cv_scores)
        cv_score_means.append(cv_score_mean)
        cv_score_stds.append(cv_score_std)

        if detail:
            print('----------------boxcox(', lmd, ')[mean=', cv_score_mean, ', std=', cv_score_std,
                  '], diff[mean=', cv_score_mean - cv_score_means[0], ', std=', cv_score_std - cv_score_stds[0],
                  ']---------------')

    best_trans_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                             max_optimization=False)
    lmd = lmds[best_trans_index]

    print('best transformation is boxcox(' + str(lmd) + ')')
    print()

    return lmd, min_x


def probe_best_transform_linear(x, y, lmds=None, min_x=None, measure_func=metrics.mean_squared_error, cv_repeat_times=1,
                                mean_std_coeff=(1.0, 1.0), max_optimization=True, score_min_gain=1e-4, random_state=0,
                                detail=False):
    if lmds is None:
        lmds = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, .038, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    if min_x is None:
        min_x = min(x)
    x = (x - min_x).to_frame()

    large_num = 1e10
    bad_score = -large_num if max_optimization else large_num
    last_best_score = bad_score
    score_cache = {}

    while True:
        cv_score_means = []
        cv_score_stds = []
        for lmd in lmds:
            if lmd not in score_cache:
                try:
                    cv_scores = bootstrap_k_fold_cv_train(LinearRegression(), boxcox1p(x, lmd), y,
                                                          measure_func=measure_func, repeat_times=cv_repeat_times,
                                                          random_state=random_state)
                    cv_score_mean = np.mean(cv_scores)
                    cv_score_std = np.std(cv_scores)
                    score_cache[lmd] = cv_score_mean, cv_score_std
                except Exception as e:
                    cv_score_mean = bad_score
                    cv_score_std = large_num

                    print(e)
            else:
                cv_score_mean, cv_score_std = score_cache[lmd]

            cv_score_means.append(cv_score_mean)
            cv_score_stds.append(cv_score_std)

            if detail:
                print("----------------lmd=", lmd, ", mean=", cv_score_mean, ", std=", cv_score_std, "---------------")

        lmd_size = len(lmds)
        if max_optimization:
            cur_best_mean_index = max(range(lmd_size), key=lambda i: cv_score_means[i])
        else:
            cur_best_mean_index = min(range(lmd_size), key=lambda i: cv_score_means[i])
        cur_best_std_index = min(range(lmd_size), key=lambda i: cv_score_stds[i])

        if np.abs(cv_score_means[cur_best_mean_index] - last_best_score) < score_min_gain:
            cur_best_lmd_index = calc_best_score_index(cv_score_means, cv_score_stds, mean_std_coeff=mean_std_coeff,
                                                       max_optimization=max_optimization)
            cur_best_lmd = lmds[cur_best_lmd_index]
            print('--best lmd=', cur_best_lmd, ', mean=', cv_score_means[cur_best_lmd_index], ', std=',
                  cv_score_stds[cur_best_lmd_index])

            return cur_best_lmd, cv_score_means[cur_best_lmd_index], cv_score_stds[cur_best_lmd_index]

        last_best_score = cv_score_means[cur_best_mean_index]

        l = min(cur_best_mean_index, cur_best_std_index) - 1
        r = max(cur_best_mean_index, cur_best_std_index) + 1
        if r >= lmd_size:
            r = lmd_size
            right_value = lmds[-1] * 1.5 + .1 if lmds[-1] >= 0 else lmds[-1] / 2
            lmds = np.insert(lmds, [lmd_size], [right_value])
        if l < 0:
            l = 0
            left_value = lmds[0] / 2 - .1 if lmds[0] >= 0 else lmds[0] * 1.5
            lmds = np.insert(lmds, [0], [left_value])
        step = (lmds[l + 1] - lmds[l]) / 2
        if step <= 1e-10:
            continue

        next_lmds = np.arange(lmds[l], lmds[r] + step, step)
        lmd_size = len(next_lmds)
        if lmd_size <= 5:
            step = step / 2
            next_lmds = np.arange(lmds[l], lmds[r] + step, step)
        elif lmd_size > 16:
            if cur_best_mean_index < cur_best_std_index:
                next_lmds = np.append(np.arange(lmds[l], lmds[l + 2], step), lmds[l + 2:r + 1])
            else:
                step = (lmds[r - 1] - lmds[r - 2]) / 4
                next_lmds = np.append(lmds[l:r - 2], np.arange(lmds[r - 2], lmds[r] + step, step))

        lmds = np.round(next_lmds, 10)
