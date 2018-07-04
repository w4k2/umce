import helper as h
import numpy as np
import method as m
import csv
from sklearn import svm, base, neighbors, metrics, naive_bayes, tree, neural_network
from imblearn import under_sampling, over_sampling, ensemble

# Prepare experiments
base_clfs = {
    #"SVC" : svm.SVC(probability=True, gamma='scale'),
    "GNB" : naive_bayes.GaussianNB(),
    #"kNN" : neighbors.KNeighborsClassifier(),
    #"DTC" : tree.DecisionTreeClassifier(),
    #"MLP" : neural_network.MLPClassifier(),
}
datasets = h.datasets_for_groups([
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2",
    "imb_IRlowerThan9",
])

for bclf in base_clfs:
    print(bclf)
    base_clf = base_clfs[bclf]
    with open('results/%s.csv' % bclf, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dataset','reg','regd','us','usd','os','osd',
                             'ens','ensd','wens','wensd','cwens','cwensd'])
        for dataset in datasets:
            print(dataset)
            X, y, X_, y_ = h.load_dataset(dataset)
            r_bac = []

            e_bac = []
            f_bac = []
            g_bac = []

            re_bac = []
            rf_bac = []
            rg_bac = []

            os_bac = []
            us_bac = []
            for fold in range(5):
                X_train, X_test = X_[fold]
                y_train, y_test = y_[fold]

                # Evaluating regular clf
                regular_clf = base.clone(base_clf)
                regular_clf.fit(X_train, y_train)
                regular_pred = regular_clf.predict(X_test)
                regular_bac = metrics.balanced_accuracy_score(y_test,regular_pred)
                #print("Regular bac: %.3f" % regular_bac)
                r_bac.append(regular_bac)

                # Evaluating over and undersampling
                us = under_sampling.RandomUnderSampler()
                os = over_sampling.RandomOverSampler()
                X_us, y_us = us.fit_sample(X_train, y_train)
                X_os, y_os = os.fit_sample(X_train, y_train)
                us_clf = base.clone(base_clf)
                os_clf = base.clone(base_clf)
                us_clf.fit(X_us, y_us)
                os_clf.fit(X_os, y_os)
                us_pred = us_clf.predict(X_test)
                os_pred = os_clf.predict(X_test)
                us_bac.append(metrics.balanced_accuracy_score(y_test,us_pred))
                os_bac.append(metrics.balanced_accuracy_score(y_test,os_pred))

                # Evaluating method
                ens = m.UndersampledEnsemble(base_clf, dataset[1])
                ens.fit(X_train, y_train)
                ens.test(X_test)

                prediction = ens.predict(X_test)
                prediction_w = ens.predict_bac_weighted(X_test)
                prediction_c = ens.predict_c_weighted(X_test)

                ens.reduce()

                rprediction = ens.predict(X_test)
                rprediction_w = ens.predict_bac_weighted(X_test)
                rprediction_c = ens.predict_c_weighted(X_test)

                ensemble_bac = metrics.balanced_accuracy_score(y_test,prediction)
                fnsemble_bac = metrics.balanced_accuracy_score(y_test,prediction_w)
                gnsemble_bac = metrics.balanced_accuracy_score(y_test,prediction_c)

                rensemble_bac = metrics.balanced_accuracy_score(y_test,rprediction)
                rfnsemble_bac = metrics.balanced_accuracy_score(y_test,rprediction_w)
                rgnsemble_bac = metrics.balanced_accuracy_score(y_test,rprediction_c)

                e_bac.append(ensemble_bac)
                f_bac.append(fnsemble_bac)
                g_bac.append(gnsemble_bac)

                re_bac.append(rensemble_bac)
                rf_bac.append(rfnsemble_bac)
                rg_bac.append(rgnsemble_bac)

            print("REG (%.3f +- %.3f)" % (np.mean(r_bac), np.std(r_bac)))
            print("US_ (%.3f +- %.3f)" % (np.mean(us_bac), np.std(us_bac)))
            print("OS_ (%.3f +- %.3f)\n--- Regular" % (np.mean(os_bac), np.std(os_bac)))
            print("ENS (%.3f +- %.3f)" % (np.mean(e_bac), np.std(e_bac)))
            print("FNS (%.3f +- %.3f)" % (np.mean(f_bac), np.std(f_bac)))
            print("GNS (%.3f +- %.3f)\n--- Reduced" % (np.mean(g_bac), np.std(g_bac)))
            print("ENS (%.3f +- %.3f)" % (np.mean(re_bac), np.std(re_bac)))
            print("FNS (%.3f +- %.3f)" % (np.mean(rf_bac), np.std(rf_bac)))
            print("GNS (%.3f +- %.3f)\n" % (np.mean(rg_bac), np.std(rg_bac)))

            writer.writerow([dataset[1]] + ["%.3f" % i for i in
                                 [np.mean(r_bac),np.std(r_bac),
                                 np.mean(us_bac),np.std(us_bac),
                                 np.mean(os_bac),np.std(os_bac),
                                 np.mean(e_bac),np.std(e_bac),
                                 np.mean(f_bac),np.std(f_bac),
                                 np.mean(g_bac),np.std(g_bac)]])
