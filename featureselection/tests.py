from django.test import TestCase
from featureselection.backend import load_csv
import featureselection.algorithm as algorithm
import datetime
# Create your tests here.

data, labels = load_csv('C:\\Users\\Administrator\\Desktop\\featureSelect\\dataSets\\wine\\wine.csv')
f = algorithm.reliefF(data, labels)
print(f)
