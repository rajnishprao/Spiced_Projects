'''
Silly Bayes Predicts
'''

from sklearn.naive_bayes import MultinomialNB

def silly_bayes(x, y):
    model = MultinomialNB()
    model.fit(x, y)
    print('Silly Bayes is trained')
    return model 
