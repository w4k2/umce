"""
Ensemble of imbalance ratio folded undersampling experiments.
"""
import csv
import helper as h
from scipy import stats
from tqdm import tqdm
import numpy as np
import method as m
from sklearn import svm, base, neighbors, metrics, naive_bayes, tree, neural_network
from imblearn import under_sampling, over_sampling, ensemble

# Prepare experiments
base_clfs = {
    #"SVC" : svm.SVC(probability=True, gamma='scale'),
    "GNB" : naive_bayes.GaussianNB(),
    "kNN" : neighbors.KNeighborsClassifier(),
    "DTC" : tree.DecisionTreeClassifier(),
    #"MLP" : neural_network.MLPClassifier(),
}
datasets = h.datasets_for_groups([
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2",
    #"imb_IRlowerThan9",
])
header = ['dataset',
          'reg','regd',
          'us','usd',
          'os','osd',
          'ereg','eregd',
          'ewei','eweid',
          'ecwei','ecweid',
          'enwei','enweid',
          'encwei','encweid',
          'eregr','eregrd',
          'eweir','eweird',
          'ecweir','ecweird',
          'enweir','enweird',
          'encweir','encweird',
            'eregos','eregosd',
            'eweios','eweiosd',
            'ecweios','ecweiosd',
            'enweios','enweiosd',
            'encweios','encweiosd',
            'eregros','eregrosd',
            'eweiros','eweirosd',
            'ecweiros','ecweirosd',
            'enweiros','enweirosd',
            'encweiros','encweirosd'
]
versions = 20

for bclf in tqdm(base_clfs):
    base_clf = base_clfs[bclf]
    csvfile = open('results/%s.csv' % bclf, 'w')

    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for dataset in tqdm(datasets):
        #print(dataset)
        # Load dataset
        X, y, X_, y_ = h.load_dataset(dataset)

        # Prepare place for results
        #r_bac = []
        #os_bac = []
        #us_bac = []
        bacs = [[] for i in range(versions+3)]

        # Iterate folds
        for fold in range(5):
            # Get division
            X_train, X_test = X_[fold]
            y_train, y_test = y_[fold]

            # Evaluating regular clf
            bacs[0].append(m.regular_bac(base_clf,
                                       X_train, y_train,
                                       X_test, y_test))

            # Evaluating over and undersampling
            tmp_bacs = [[],[]]
            for i in range(10):
                us_os_bac = m.us_os_bac(base_clf,
                                        X_train, y_train,
                                        X_test, y_test)
                tmp_bacs[0].append(us_os_bac[0])
                tmp_bacs[1].append(us_os_bac[1])

            tmp_bacs = np.array(tmp_bacs)
            tmp_bacs = np.mean(tmp_bacs,axis = 1)

            bacs[1].append(tmp_bacs[0])
            bacs[2].append(tmp_bacs[1])

            # Evaluating method
            ens = m.UndersampledEnsemble(base_clf, dataset[1])
            ens.fit(X_train, y_train)
            ens.test(X_test)

            # Establish predictions for parameters
            ens_predictions = []
            for os in (False, True):
                for reduced in (False, True):
                    for mode in ('regular',
                                 'weighted', 'cweighted',
                                 'nweighted', 'ncweighted'):
                        ens_predictions.append(ens.predict(
                            ens.decision_cube(reduced = reduced,
                                              mode = mode,
                                              os = os)
                        ))

            # Score
            ens_bacs = [
                metrics.balanced_accuracy_score(y_test, pred)
                for pred in ens_predictions
            ]

            for i, ens_bac in enumerate(ens_bacs):
                bacs[i+3].append(ens_bac)

        #print("REG (%.3f +- %.3f)" % (np.mean(r_bac), np.std(r_bac)))
        #print("US_ (%.3f +- %.3f)" % (np.mean(us_bac), np.std(us_bac)))
        #print("OS_ (%.3f +- %.3f)\n--- Ens" % (np.mean(os_bac), np.std(os_bac)))
        #for row in bacs:
        #    print("ENS (%.3f +- %.3f)" % (np.mean(row), np.std(row)))

        mean_bacs = [np.mean(i) for i in bacs]
        std_bacs = [np.std(i) for i in bacs]

        leader = np.argmax(mean_bacs)

        p_treshold = 0.01
        leaders = []
        for i, vec in enumerate(bacs):
            pvalue = stats.wilcoxon(bacs[leader],bacs[i]).pvalue
            if pvalue < p_treshold or np.isnan(pvalue):
                leaders.append(i)

        #print(leaders)

        # Establish result row
        foo = [dataset[1].replace("_", "-")]

        for i, bacv in enumerate(bacs):
            if i in leaders:
                #print("LEADER")
                foo += ["\\cellcolor{green!25}" + ("%.3f" % np.mean(bacv))[1:]]
            else:
                #print("Moron")
                foo += [("%.3f" % np.mean(bacv))[1:]]

            foo += [("%.3f" % np.std(bacv))[1:]]

        #print(foo)

        """
        foo = [dataset[1].replace("_", "-")] + [("%.3f" % i)[1:] for i in
                             [np.mean(bacs[0]),np.std(bacs[0]),
                             np.mean(bacs[1]),np.std(bacs[1]),
                             np.mean(bacs[2]),np.std(bacs[2]),
                             np.mean(bacs[3]),np.std(bacs[3]),
                             np.mean(bacs[4]),np.std(bacs[4]),
                             np.mean(bacs[5]),np.std(bacs[5]),
                             np.mean(bacs[6]),np.std(bacs[6]),
                             np.mean(bacs[7]),np.std(bacs[7]),
                             np.mean(bacs[8]),np.std(bacs[8]),
                             np.mean(bacs[9]),np.std(bacs[9]),
                             np.mean(bacs[10]),np.std(bacs[10]),
                             np.mean(bacs[11]),np.std(bacs[11]),
                             np.mean(bacs[12]),np.std(bacs[12]),
                             np.mean(bacs[13]),np.std(bacs[13]),
                             np.mean(bacs[14]),np.std(bacs[14]),
                             np.mean(bacs[15]),np.std(bacs[15]),
                             np.mean(bacs[16]),np.std(bacs[16]),
                             np.mean(bacs[17]),np.std(bacs[17]),
                             np.mean(bacs[18]),np.std(bacs[18]),
                             np.mean(bacs[19]),np.std(bacs[19]),
                             np.mean(bacs[20]),np.std(bacs[20]),
                             np.mean(bacs[21]),np.std(bacs[21]),
                             np.mean(bacs[22]),np.std(bacs[22])]]
        """
        #print(foo)
        #exit()
        #print(len(header))
        #print(len(foo))
        #print(header)
        #print(foo)
        writer.writerow(foo)
        #exit()
    csvfile.close()
