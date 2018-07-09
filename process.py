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
    "GNB" : naive_bayes.GaussianNB(),
    #"kNN" : neighbors.KNeighborsClassifier(),
    "DTC" : tree.DecisionTreeClassifier(),
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

summary = np.zeros((len(base_clfs), 23))

for cid, bclf in enumerate(base_clfs):
    print(cid, bclf)
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
        summary[cid,leaders] += 1

        # Establish result row
        foo = [dataset[1].replace("_", "-")]

        for i, bacv in enumerate(bacs):
            c = ""
            if i in leaders:
                c = "\\cellcolor{green!25}"
            else:
                c = ""

            if np.mean(bacv) >= 1:
                foo += ["%s1" % (c)]
            else:
                foo += ["%s%s" % (c, ("%.3f" % np.mean(bacv))[1:])]
            foo += [("%.3f" % np.std(bacv))[1:]]


        """
        for i, bacv in enumerate(bacs):
            if i in leaders:
                #print("LEADER")
                foo += ["\\cellcolor{green!25}" + ("%.3f" % np.mean(bacv))[1:]]
            else:
                #print("Moron")
                foo += [("%.3f" % np.mean(bacv))[1:]]

            foo += [("%.3f" % np.std(bacv))[1:]]
        """

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


full = np.array([summary[:,0]]).T
us   = np.array([summary[:,1]]).T
os   = np.array([summary[:,2]]).T

without_os  = summary[:,3:13]
with_os     = summary[:,13:]

without_pru = summary[:,[ 3, 4, 5, 6, 7,13,14,15,16,17]]
with_pru    = summary[:,[ 8, 9,10,11,12,18,19,20,21,22]]

reg         = summary[:,[ 3, 8,13,18]]
wei         = summary[:,[ 4, 9,14,19]]
con         = summary[:,[ 5,10,15,20]]
nor         = summary[:,[ 6,11,16,21]]
nci         = summary[:,[ 7,12,17,22]]

full = np.sum(full,axis=1)
us   = np.sum(us,axis=1)
os   = np.sum(os,axis=1)

without_os  = np.max(without_os,axis=1)# / 10
with_os     = np.max(with_os,axis=1)# / 10

without_pru = np.max(without_pru,axis=1)# / 10
with_pru    = np.max(with_pru,axis=1)# / 10

reg         = np.max(reg,axis=1)# / 4
wei         = np.max(wei,axis=1)# / 4
con         = np.max(con,axis=1)# / 4
nor         = np.max(nor,axis=1)# / 4
nci         = np.max(nci,axis=1)# / 4

end_summary = np.column_stack((full,us,os,
                               without_os,with_os,
                               without_pru,with_pru,
                               reg,wei,con,nor,nci))

clfs = ['GNB','DTC']
header = ['clf',
          'full','us','os',
          'withoutos','withos',
          'withoutpru', 'withpru',
          'reg','wei','con','nor','nci']

print(end_summary)

print(header)
csvfile = open('results/summary.csv', 'w')

writer = csv.writer(csvfile, delimiter=',')
writer.writerow(header)

for i, row in enumerate(end_summary):
    print([clfs[i]] + list(row.astype(int)))
    writer.writerow([clfs[i]] + list(row))
csvfile.close()
