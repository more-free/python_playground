__author__ = 'morefree'

import scipy as sp
import matplotlib.pyplot as plt

# square error on f(x) vs true values y
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


data_path = '/Users/morefree/Documents/books/BuildingMachineLearningSystemsWithPython-master/ch01/data/web_traffic.tsv'
data = sp.genfromtxt(data_path, delimiter='\t')

hours = data[:,0]
hits = data[:,1]

hours = hours[~sp.isnan(hits)]
hits = hits[~sp.isnan(hits)]

#plt.scatter(hours, hits)
#plt.title('web trafic over the last month')
#plt.xlabel('time')
#plt.ylabel('hits per hour')
#plt.grid()
#plt.show()

fp1, residuals, rank, sv, rcond = sp.polyfit(hours, hits, 1, full=True)
print fp1, residuals, type(fp1)
f1 = sp.poly1d(fp1)
print type(f1)
print error(f1, hours, hits)


class callablefp():
    p1, p2 = 10, 5
    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.p1 * self.x + self.p2

a = callablefp(10)
print a