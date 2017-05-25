import numpy as np 
import pandas

file1 = 'result_1.csv'
file0 = 'test_1_results.csv'

results_a = pandas.read_csv(file1)
results_b = pandas.read_csv(file0)

a = np.array(results_a['Survived'])
b = np.array(results_b['Survived'])

bo = np.equal(a,b)

print(np.mean(bo))