'''
NumPy/Numerical Python: widely used open source library
universal standard for working with numerical data in Python
NumPy API used extensively in Pandas, SciPy, Matplotlib, scikit-learn etc

NumPy library contains multidimensional array and matrix data structures
It provides "ndarray", a homogeneous n-dimensional array object, with
methods to efficiently operate on it
NumPy can be used to perform various mathematical operations on arrays
'''

import numpy as np

a = np.arange(6)
a
b = np.arange(0, 6)
b
a.shape
a2 = a[np.newaxis, :]
a2
a2.shape

'''
difference between a python list and numpy:
numpy gives an enormous range of fast and efficient ways of creating arrays
and manipulating numerical data inside them
while pythn list can contain different data types within a single list, all of
the elements in a numpy array should be homogeneous
math operations that are meant to be performed on arrays would be extremely
inefficient if the arrays weren't homogeneous

numpy arrays are faster and more compact than python lists
an array consumes less memory and is convenient to use
'''

'''
numpy array: central data structure of the numpy library
an array is a grid of values and contains info about the raw data, how to
locate an element, and how to interpret an element.
it has a grid of elements that can be indexed in various ways
the elements are all of the same type, referred to as the array "dtype"

an array can be indexed by a tuple of nonnegative integers, by booleans,
by another array, or by integers

the "rank" of the array is the number of dimensions

the "shape" of the array is a tuple of integers giving the size of the array
in each dimension
'''

# initialize array from numpy arrays, using nested lists for two- or higher-
# dimensional data
a = np.array([1, 2, 3, 4, 5, 6])
a

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a

# elements can be accessed by indexing
a[0]
a[0][2]

'''
ndarray: stands for N-dimensional array, ie an array with any number of
dimensions (1-D, 2-D etc)
NumPy ndarray class is used to represent both matrices and vectors
"vector" is an array with a single dimension (ie no difference between row and
column vectors)
"matrix" refers to an array with two dimensions
"tensor" is the term for 3-D or higher dimensional arrays

array is usually a fixed size container of items of the same type and size
the no. of dimensions and items in an array is defined by its shape

the shape of an array is a tuple of non-negative integers that specify
the sizes of each dimension

in numpy, the dimensions are called axes, which means a 2D array looks like:
[[1, 1, 1],
 [2, 3, 3]]
in this example, the first axis has a length of 2, and the second axis of 3
'''

'''
creating a basic array
'''
# function to create numpy arrays
np.array()
# pass a list to it

a = np.array([1, 2, 3])
a

# you can also create an array filled with 0s
np.zeros(2)
# or an array filled with 1s, attn default is floats
np.ones(2)
# or even an empty array: attn the initial content is random and depends on
# the state of memory, empty() is faster to use than zeros() - but do
# remember to fill in all the elements later!
np.empty(2)

# you can create an array with a range of elements
np.arange(4)
# or an array of evenly spaced integers, but here you gotta specify the
# first number, last number and step size
np.arange(2, 9, 2)
# linspace() creates array with values that are spaced linearly in a specified
# interval
np.linspace(0, 10, num=5)
np.linspace(0, 10, num=25)

# specifying dtype
a = np.ones(2, dtype=np.int64)
a

'''
adding, removing and sorting elements
'''
# np.sort() : returns a sorted copy of the array
arr = np.array([3, 5, 6, 1, 8, 9, 2])
np.sort(arr)

arr2 = np.array([[5, 7, 6], [1, 2, 3]])
arr2
np.sort(arr2, axis=0) # axis=0 means sorting happens 'down'-wards
np.sort(arr2, axis=1) # whereas axis=1 happens 'right'-wards

# there are several variations of sort - and i will go into detail with some of them
'''np.argsort()'''
# np.argsort(): returns indices that would sort an array
# for a 1-D array:
x = np.array([3, 1, 2])
np.argsort(x)
# for a 2-D array:
x = np.array([[5, 7, 6], [1, 2, 3]])
x
# there are two ways of sorting for this, depending on which axis
# to sort along the first axis (down)
ind = np.argsort(x, axis=0)
ind
tnd = np.argsort(x, axis=1)
tnd

# np.take_along_axis() is similar to the above, with slightly different syntax
np.take_along_axis(x, ind, axis=0) # this is supposedly same as np.sort(x, axis=0), so why bother doing this?
# similarly below same as np.sort(x, axis=1) - but again whyyyy?
np.take_along_axis(x, tnd, axis=1)
# a couple more complicated things with argsort() but will skip for now
'''
np.lexsort(): sorts using keys
np.searchsorted(): find indices where elements should be inserted to maintain order
np.partition(): returns a partitioned copy of an array - or a former colony
'''

'''
np.concatenate()
'''

a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 9])
np.concatenate((a, b))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
np.concatenate((x, y), axis=0)
# np.concatenate((x, y), axis=1): does not work as sizes are different
y2 = np.array([[7, 8], [4, 5]])
np.concatenate((x, y2), axis=0)
np.concatenate((x, y2), axis=1)
# noice!

'''
shape and size of an array
ndarray.ndim: gives no. of axes, or dimensions, of an array

ndarray.size: gives total no. of elements of the array (this is product of
the elements of an array)

ndarray.shape: indicates no. of elements stored along each dimension of the array,
as a tuple of integers
'''

q = np.array([[[0, 1, 2, 3],[4, 5, 6, 7]],[[0, 1, 2, 3],[4, 5, 6, 7]],[[0, 1, 2, 3],[4, 5, 6, 7]]])

q

q.ndim

q.size

q.shape

'''
reshaping an array: arr.reshape() gives a new shape to an array without
changing the data - as long as the new array you want to produce has the
same no. of elements as the original array.
'''

a = np.arange(6)
a

a.reshape(3, 2)

a.reshape(2, 3)

# another way is numpy.reshape(a, newshape=(1, 6), order='C')
# the order= C or F arguments refer to a C-like or Fortran-like index order

'''
convert a 1D array into a 2D array

np.newaxis: will increase the dimensions of the array by 1 dimension when used once
ie 1D will become 2D, 2D will become 3D etc

np.expand_dims: inserts a new axis at a specified position
'''

a = np.array([1, 2, 3, 4, 5, 6])
a.shape

a2 = a[np.newaxis, :]
a2.shape
a2

# you can explicitly convert to 1D array with either a row vector or a column
# vector using np.newaxis.

row_vector = a[np.newaxis, :]
row_vector.shape
row_vector

column_vector = a[:, np.newaxis]
column_vector.shape
column_vector

# to add axis at index position 1
b = np.expand_dims(a, axis=1)
b.shape

# or add axis at position 0
c = np.expand_dims(a, axis=0)
c.shape


'''
Indexing and Slicing: done in the same ways as python lists
'''

data = np.array([1, 2, 3])

data[1]
data[0:2]
data[:2]
data[1:]
data[-2:]

# selecting values from array with conditions

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# values below 7
print(a[a < 7])

five_up = (a >= 5)
print(a[five_up])

divisible_by_2 = a[a%2==0]
print(divisible_by_2)

# select elemtsn that satisfy two conditions using & and | operators

c = a[(a > 2) & (a < 11)]
print(c)

five_up = (a > 5) | (a == 5)
print(five_up)
# this gives booleans

# can also use np.nonzero() to select elements from an array
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

b = np.nonzero(a < 5)
b
# this returns a tuple of arrays, one for each dimension
# first array represents the row indices where the values are found
# second array represents the column indices where the values are found

# to generate a list of coordiantes where the elements exist, you can zip the
# arrays, iterate over the list of coordinates and print them

list_of_coordinates = list(zip(b[0], b[1]))
for coord in list_of_coordinates:
    print(coord)

# can also use np.nonzero to print elements that are less than 5
print(a[b])
# interesting

# if the element you are looking for does not exist in the array,
# then the returned array of indices will be empty

not_there = np.nonzero(a == 42)
print(not_there)


'''
Create an array from existing data
'''
# by slicing
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

arr1 = a[3:8]
arr1

# using vertical or horizontal stack options when you have two arrays

a1 = np.array([[1, 1], [2, 2]])
a2 = np.array([[3, 3], [4, 4]])

a1

a2# vertical stack
np.vstack((a1, a2))

# horizontal stack
np.hstack((a1, a2))

# hsplit() to split an array

x = np.arange(1, 25).reshape(2, 12)
x
# to split this into three equally shaped arrays:
np.hsplit(x, 3)

# to split the column after the 3rd and 4th column,
np.hsplit(x, (3, 4))

np.vsplit(x, 2)

# view method creates a shallow copy of the the original array
# views are an important num py concept
# numpy functions as well as operations like indexing and slicing return views (whenever possible)
# this saves memory and is faster
# but imp to note: modifying data in a view will modify the original array

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# we create an array b1 by slicing a and modifying the frst element of b1 -
# this will modify a too!

b1 = a[0, :]
b1
b1[0] = 99
b1
a

'''
basic array operations
'''
data = np.array([1, 2])
data
ones = np.ones(2)
ones
data + ones
data - ones
data * data
data / data

a = np.array([1, 2, 3, 4])
a.sum()

b = np.array([[1, 1], [2, 2]])
b
# to sum the rows, or go downwards, axis=0
b.sum(axis=0)
# to sum the columns, or go sideways, axis=1
b.sum(axis=1)

'''
broadcasting: carry out an operation between an array and a single number
'''
data = np.array([1.0, 2.0])
data * 1.6

'''
other useful array operations
'''
data.max()
data.min()
data.sum()
# in multidimensional arrays, you can use axis argument along with sum/min/max

'''
creating matrices: pass a python list of lists
'''
data = np.array([[1, 2], [3, 4]])
data
# indexing and Slicing
data[0, 1]
data[1:3]
data[0:2, 0]
data.max()
data.min()
data.sum()
data.max(axis=0)
data.max(axis=1)

data = np.array([[1, 2], [3, 4]])
ones = np.array([[1, 1], [1, 1]])
data + ones

# you can also do these arithmetic operations on matrices of different sizes
# only if one matrix has only one column or one row (ie numpy will use broadcast rules)
data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
data
ones_row
data + ones_row

'''
random number generator: imp to randomly initialize weights in an artifical
neural network, split data into random sets, or randomly shuffle dataset etc
'''
rng = np.random.default_rng(0)
# to the generator class, you need to pass how many elements you need
rng.random(3)
rng.random((3, 2))

# to generate 2 x 4 array of random integers between 0 and 4
rng.integers(5, size=(2, 4))

'''
to get unique items and counts
'''
a = np.array([11, 11, 12, 14, 13, 11, 14, 19, 20])
unique_values = np.unique(a)
unique_values
# to get indices of unique values also:
unique_values, indices_list = np.unique(a, return_index=True)
unique_values
indices_list
# return_counts argument will give frequency count of unique values
unique_values, occurance_count = np.unique(a, return_counts=True)
unique_values
occurance_count

# this also works for 2D arrays

a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
unique_values = np.unique(a_2d)
unique_values

# to get unique rows or columsn use axis arg: 0 for rows, 1 for colums
unique_rows = np.unique(a_2d, axis=0)
unique_rows
unique_columns = np.unique(a_2d, axis=1)
unique_columns

unique_columns, indices, occurance_count = np.unique(a_2d, axis=1, return_counts=True, return_index=True)
unique_columns
indices
occurance_count


'''
transposing and reshaping a matrix
'''

data
data.reshape(2, 3)
data.reshape(3, 2)

# instead of doing this each time, .transpose changes it
data.transpose()

'''
reverse an array
'''

# np.flip() flips or reverses the contents of an array along an axis
# reversing a 1D array
arr = np.array([1, 2, 3, 4])
np.flip(arr)
# reversing a 2D array
a_2d
np.flip(a_2d)
# this reversed the contents in all rows and all columns

# to reverse only rows
np.flip(a_2d, axis=0)
#  and to reverse only columns
np.flip(a_2d, axis=1)

# you can reverse contents of only one column or row
# here reversing contents of row at index position 1
a_2d[1] = np.flip(a_2d[1])
a_2d

# to reverse column at index position 1, ie second column
a_2d[:, 1] = np.flip(a_2d[:, 1])
a_2d

'''
reshaping and flattening multidimensional arrays
'''

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
x.flatten()

x.ravel()

# difference is that ravel is a numpy view, so any changes made in it will
# also change the parent array


# ok rest was minor things - accessing help etc etc 
