import os
import helper as h
import exposing
import numpy as np
from sklearn import svm, model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Prepare experiments
ds_dir = "datasets"
base_clf = svm.SVC
ds_groups = [
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2",
    "imb_IRlowerThan9",
    "imb_multiclass" ]
table_file = open("tables/results.tex", "w")

# Iterating groups
for group_idx, ds_group in enumerate(ds_groups):
    group_path = "%s/%s" % (ds_dir, ds_group)
    print("## Group %s" % ds_group)

    # Iterating datasets in group
    ds_list = sorted(os.listdir(group_path))
    for ds_idx, ds_name in enumerate(ds_list):
        if ds_name[0] == '.' or ds_name[0] == '_':
            continue
        h.notify(ds_name, "%i/%i (%i/%i)" % (
            ds_idx + 1,len(ds_list),
            group_idx + 1,len(ds_groups)
        ))
        exit()

        print("\n### %s dataset" % ds_name)
        scores = np.zeros((len(clfs), 5))

        # Load dataset
        X, y = h.load_keel("%s/%s/%s.dat" % (
            group_path, ds_name, ds_name
        ))

        # GridGridSearchCV
        print("\nBest parameters")
        ee = exposing.EE()
        gs = GridSearchCV(ee, h.spams())
        gs.fit(X, y)
        best_params = gs.best_params_
        print(">\t%s" % best_params)

        for i in range(1,6):
            X_train, y_train = h.load_keel("%s/%s/%s-5-fold/%s-5-%itra.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            ))
            X_test, y_test = h.load_keel("%s/%s/%s-5-fold/%s-5-%itst.dat" % (
                group_path, ds_name, ds_name, ds_name, i
            ))

            for j, clf_name in enumerate(clfs):
                if clf_name == 'EEC':
                    clf = exposing.EE(approach=best_params['approach'],
                                      fuser=best_params['fuser'],
                                      grain=best_params['grain'],
                                      focus = best_params['focus'],
                                      a_steps = best_params['a_steps'])
                    ee = clf
                else:
                    clf = clfs[clf_name]()
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores[j,i-1] = score

        mean_scores = np.mean(scores, axis = 1)
        std_scores = np.std(scores, axis = 1)

        figname, fignameb = h.plot(mean_scores, std_scores, ds_group,
                                   ds_name, clfs, ee)

        h.markdown(figname,fignameb,clfs, mean_scores, std_scores)

        tl = h.texline(scores, clfs, ds_name, best_params)
        print(tl)
        table_file.write(tl)
table_file.close()
