import csv #EXCELL
import pymorphy2 #БИБЛИОТЕКА МОРФОЛОГИЧЕСКОГО АНАЛИЗАТОРА РУССКОГО ЯЗЫКА 
import re #


# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port

import pyodbc
cnxn = pyodbc.connect(r'Driver=SQL Server;Server=YourPC_NAME\SQLEXPRESS;Database=app;Trusted_Connection=yes;')
cursor = cnxn.cursor()

morph = pymorphy2.MorphAnalyzer(lang='ru') #выбор языка


answer_id=[] 
answer = dict()

cursor.execute('SELECT id, answer FROM chats_answer;')
records = cursor.fetchall()
for row in records:
 answer[row[0]]=row[1]

questions=[] 

cursor.execute('SELECT question, answer_id FROM chats_question;')
records = cursor.fetchall()
transform=0

for row in records:
 if row[0]>"":
  if row[1]>0:
   phrases=row[0]
   words=phrases.split(' ')
   phrase=""
   for word in words:
    word = morph.parse(word)[0].normal_form  
    phrase = phrase + word + " "
   if (len(phrase)>0):
    questions.append(phrase.strip())
    answer_id.append(row[1])
    transform=transform+1

#print (questions)
#print (answer)
#print (answer_id)

cursor.close()
#conn.close()

import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

vectorizer_q = TfidfVectorizer()
vectorizer_q.fit(questions)
matrix_big_q = vectorizer_q.transform(questions)
print ("Размер матрицы: ")
print (matrix_big_q.shape)

# Трансформируем матрицу вопросов в меньший размер для уменьшения объема данных
# Трансформировать будем в 200 мерное пространство, если вопросов больше 200
# Размерность подбирается индивидуально в зависимости от базы вопросов, которая может содержать 1 млн. или 1к вопросов и 1
# Без трансформации большая матрицу будет приводить к потерям памяти и снижению производительности

if transform>200:
 transform=200
print(transform)
svd_q = TruncatedSVD(n_components=transform)
svd_q.fit(matrix_big_q)
matrix_small_q = svd_q.transform(matrix_big_q)
print ("Коэффициент уменьшения матрицы: ")
print ( svd_q.explained_variance_ratio_.sum())



import numpy as np

from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator

def softmax(x):
  #создание вероятностного распределения
  proba = np.exp(-x)
  return proba / sum(proba)

class NeighborSampler(BaseEstimator):
  def __init__(self, k=5, temperature=10.0):
    self.k=k
    self.temperature = temperature
  def fit(self, X, y):
    self.tree_ = BallTree(X)
    self.y_ = np.array(y)
  def predict(self, X, random_state=None):
    distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
    result = []
    for distance, index in zip(distances, indices):
      result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
    return self.y_[result]

from sklearn.pipeline import make_pipeline

ns_q = NeighborSampler()
ns_q.fit(matrix_small_q, answer_id) 
pipe_q = make_pipeline(vectorizer_q, svd_q, ns_q)


import re
import telebot
from telebot import types
telebot.apihelper.ENABLE_MIDDLEWARE = True
bot = telebot.TeleBot("_____________")
# добавте ваш ключ

@bot.message_handler(commands=['start'])
def start_message(message):
	bot.send_message(message.from_user.id, " Здравствуйте, давайте поговорим.\n Пишите ваш вопрос, слова exit или выход для выхода")

@bot.message_handler(func=lambda message: True) 
def get_text_messages(message):
	request=message.text
	words= re.split('\W',request)
	phrase=""
	for word in words:
		word = morph.parse(word)[0].normal_form  
		phrase = phrase + word + " "
	reply_id    = int(pipe_q.predict([phrase.strip()]))
	bot.send_message(message.from_user.id, answer[reply_id])
	print("Запрос:", request, " \n\tНормализованный: ", phrase, " \n\t\tОтвет :", answer[reply_id])
  


bot.infinity_polling(none_stop=True, interval=1)



request=""
while request not in ['exit', 'выход']:
 request=input()
 words= re.split('\W',request)
 phrase=""
 for word in words:
  word = morph.parse(word)[0].normal_form  
  phrase = phrase + word + " "
 reply_id    = int(pipe_q.predict([phrase.strip()]))
 print (answer[reply_id])

  