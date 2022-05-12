from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, GRU, LSTM
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_words = 10000
max_message_len = 100

train = pd.read_csv('Data/train.csv', sep=';', header=None)
train = train.drop(train.columns[2], axis=1)
train = train.rename(columns={0: 'Class'})
train = train.rename(columns={1: 'Message'})

messages = train['Message']

y_train = train['Class'] - 1

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(messages)

sequences = tokenizer.texts_to_sequences(messages)

x_train = pad_sequences(sequences, maxlen=max_message_len)

model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_message_len))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
model.summary()

model_save_path = 'best_model_cnn.h5'
checkpoint_callback = ModelCheckpoint(model_save_path,
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[checkpoint_callback])

plt.plot(history.history['accuracy'],
        label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
        label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

model.load_weights(model_save_path)

test = pd.read_csv('Data/test.csv',
                  header=None,
                  sep=';',
                  names=['Class', 'Message'])
test_sequences = tokenizer.texts_to_sequences(test['Message'])

x_test = pad_sequences(test_sequences, maxlen=max_message_len)

y_test = test['Class'] - 1

model.evaluate(x_test, y_test, verbose=1)

text = 'Может вообще не водить?Мы с мужем любили свои садики, я до сих пор вспоминаю воспитательницу и как летом на месяц выезжали с садиком на дачу (как у школьников лагерь, раньше в садиках были ""дачи"", пока в саду ремонт делали. Сейчас такого нет)Сын с полутора лет пошел в сад сразу на весь день. Не было тогда понятий адаптации, никто не водил на полдня.    Мы и не знали, что так можно.В общем, я к чему. Только начав читать ХМ я вообще узнала, что в садике может быть плохо и что можно водить на час, на два, на полдня...   Для меня это до сих пор в мозгах плохо укладывается, если честно. Мне это странно.Как по мне, я б отдала на весь день и баста. Конечно забирала не последним. Например, после полдника, но не раньше, чтоб ребёнок точно знал.Мне кажется, что уж или водить на весь день, или не водить совсем, чего ребенка мучить.Но это сугубо ИМХО, как говорится, потому что я совершенно не знакома с новой системой и подходом к проблеме детского сада И водил у нас только папа. С папой легче расставаться без слёз, потому что папе на работу. Надо - значит надо.'

sequences = tokenizer.texts_to_sequences([text])

data = pad_sequences(sequences, maxlen=max_message_len)

result = model.predict(data)

if result < 0.5:
    print('Присутствие стрессового состояния')
else:
    print('Отсутствие стрессового состояния')