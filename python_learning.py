__author__ = 'morefree'
__metaclass__ = type

# learning code from Peter's course on Udacity, Google Python lessons, Python books
class A:
    def __init__(self):
        self.numA = 100
        print "init A"

class B:
    def __init__(self):
        self.numB = 1000
        print "init B"

class C(A, B):
    def __init__(self):
        super(C, self).__init__()
        print "init C", self.numA, self.numB


class MyClass:
    @staticmethod
    def smeth():
        print "static method"

    @classmethod
    def cmeth(cls):
        print "this is cmeth", cls

m = MyClass()
MyClass.cmeth()
MyClass.smeth()
m.cmeth()
m.smeth()
m.fuck = "shit"
print m.fuck


class TestIter:
    def __iter__(self):
        return iterator()

class iterator:
    def __init__(self):
        self.val = 0

    def next(self):
        if self.val < 10:
            self.val += 1
            return self.val
        else:
            raise StopIteration


class A:
    def f(self):
        self.num = 10

def ll(s):
    print "fuck", s

A.f = ll

a1 = A()
a1.f()

import sys
for path in sys.path:
    if path.find('site-pac') >= 0:
        print path


if __name__ == '__main__':
    print "in main"

print globals()