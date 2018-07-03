import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

def load_keel(string, separator=","):
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception

    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:,:-1], dtype=float)
    d = {'positive': 1, 'negative': 0}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip() for v in s])

    return X, y

def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

def spams():
    parameters = {
        'approach':['brute','purified','random'],
        'fuser':['equal', 'theta'],
        'focus':[1,2,3,4],
        'a_steps':[1,2,3,4],
        'grain':[8,16,32]
    }
    """
    parameters = {
        'approach':['random'],
        'fuser':['equal'],
        'focus':[1],
        'a_steps':[1],
        'grain':[8]
    }
    """
    return parameters

def plot(mean_scores, std_scores, ds_group, ds_name, clfs, ee):
    fig, ax = plt.subplots(1, figsize = (6,3))
    figname = 'figures/%s%s.png' % (ds_group,ds_name)
    ax.bar(clfs.keys(), mean_scores, yerr=std_scores)
    ax.set_ylim([0, 1])
    plt.savefig(figname)
    plt.savefig("bar.png")
    plt.close(fig)

    #print(len(ee.ensemble_))
    v = np.ceil(len(ee.ensemble_)/4).astype(int)
    #print(v)

    fig, ax = plt.subplots(v,4, figsize = (8, 2*v))
    fignameb = 'figures/%s%se.png' % (ds_group,ds_name)
    for e in range(v*4):
        if e < len(ee.ensemble_):
            ex = ee.ensemble_[e]
            ax[e // 4,e % 4].imshow(ex.rgb())
            ax[e // 4,e % 4].set_title("%s - %.3f" % (
                ex.given_subspace, ex.theta_
            ), fontsize=8)
        ax[e // 4,e % 4].axis('off')
    plt.tight_layout()
    plt.savefig(fignameb)
    plt.savefig("foo.png")
    plt.close(fig)

    return (figname, fignameb)

def markdown(figname, fignameb, clfs, mean_scores, std_scores):
    print("\n|CLF|ACC|STD|")
    print("|---|---|---|")
    for i, clf in enumerate(clfs):
        print("| %s | %.3f | +-%.2f|" % (clf, mean_scores[i], std_scores[i]))

    print("\n![](%s)" % figname)
    print("\n![](%s)" % fignameb)

def texline(scores, clfs, ds_name, best_params):
    mean_scores = np.mean(scores, axis = 1)
    std_scores = np.std(scores, axis = 1)
    ee_scores = scores[0]

    best_idx = np.argmax(mean_scores)
    #print("MS: %s" % mean_scores)
    #print("CEE: %s" % ee_scores)
    #print("BST: %s" % scores[best_idx])
    #print("Best %i" % best_idx)

    a = [stats.wilcoxon(ee_scores, scores[i]).pvalue > 0.05
         for i in range(1,6)]
    is_relative = np.where(a)
    #print(is_relative)
    is_best = np.isin(best_idx-1, is_relative)
    #print(is_best)
    #if is_best:
    #    print("Best or relative")

    # Info for ee
    if is_best:
        c = "\\cellcolor{green!25} %.3f" % mean_scores[0]
    else:
        c = "\\cellcolor{blue!25} %.3f" % mean_scores[0]
    b = [c]

    # Info for rest
    for i in range(1,6):
        score = mean_scores[i]
        if a[i-1] == True:
            if is_best:
                c = "\\cellcolor{green!25} %.3f" % mean_scores[i]
            else:
                c = "\\cellcolor{blue!25} %.3f" % mean_scores[i]
        else:
            c = "%.3f" % mean_scores[i]
        b.append(c)


    #    c = ""
    #    print(i)
    #    print(mean_scores[i])
    #    print(a[i-1])
    #    print(is_best)

    #print(b)
    b = " & ".join(b)
    #print(b)


    return "\emph{%s} & %s & %s & %i & %.1f & %i & %s\\\\\n" % (
        ds_name.replace('_','-'), best_params['approach'], best_params['fuser'],
        best_params["grain"], best_params["focus"],
        best_params["a_steps"], b)
