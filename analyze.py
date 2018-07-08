"""
Comparison of processed datasets.
"""

import csv
import helper as h
import numpy as np

# Numer porządkowy
# Nazwa zbioru
# Liczba atrybutów
# Liczba wzorców
# W tym większościowych
# W tym mniejszościowych
# IR
header = [
    'idx', 'dbname', 'features', 'samples', 'majority', 'minority', 'ir'
]
datasets = h.datasets_for_groups([
    "imb_IRhigherThan9p1",
    "imb_IRhigherThan9p2",
    #"imb_IRlowerThan9",
])

csvfile = open('results/datasets.csv', 'w')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(header)

for i, dataset in enumerate(datasets):
    dbname = dataset[1].replace("_", "-")
    #print(dbname)

    # Load and analyze dataset
    X, y, _, _ = h.load_dataset(dataset)
    classes, c_counts = np.unique(y, return_counts=True)
    majority_c = 0 if c_counts[0] > c_counts[1] else 1
    minority_c = 1 - majority_c

    n_samples = len(y)
    n_features = X.shape[1]
    n_majority = np.sum(y==majority_c)
    n_minority = np.sum(y==minority_c)
    IR = n_majority / n_minority

    #print("%i samples (%i maj / %i min)" % (
    #    n_samples, n_majority, n_minority
    #))
    #print("IR = %.2f" % IR)

    row = [i+1, dbname, n_features, n_samples, n_majority, n_minority, "%.2f" % IR]
    print(row)
    writer.writerow(row)

csvfile.close()
