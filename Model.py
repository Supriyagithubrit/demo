#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import argmax
import pandas as pd 
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

import pickle
from pickle import load,dump
import keras
import tensorflow 
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import concatenate


# ## Extract the feature Vector from all images

# In[11]:


model=InceptionV3(weights = 'imagenet')
new_model= Model(model.input,model.layers[-2].output)
new_model.summary()


# In[12]:


plot_model(new_model, to_file='model_new_CNN.png', show_shapes=True)


# ## Preprocessing for images

# In[13]:


img_path=('C:/Flickr8k/Flickr8k_dataset/Images/')
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def preprocess_image(img_path):
    img = tensorflow.keras.preprocessing.image.load_img(img_path,grayscale='grayscale',target_size=(299,299)) 
    img = tensorflow.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    return img


# In[14]:


path = 'C:/Flickr8k/Flickr8k_dataset/Images'
listdir=os.listdir(path)
def images_name(path):
    img_name = set([path+image for image in listdir])
    return img_name


# In[15]:


# Function to encode given image into a vector of size (2048, )
def encode_image(image):
    image = preprocess_image(image)
    feature_vector = new_model.predict(image)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1]) # reshape from (1, 2048) to (2048, )
    return feature_vector


# In[16]:


from time import time
def encode_image_feature(image_file_name, model):
    start_time = time()
    encoding_features = {}
    for idx,img in enumerate(image_file_name):
        encoding_features[img] = encode_image(img, new_model)
        if( (idx+1)%500 == 0):
            print('images encoded ',idx+1)        
    print("** Time taken for encoding images {} seconds **".format(time()-start_time))
    return encoding_features


# In[17]:


encode=encode_image('C:\Flickr8k\Flickr8k_dataset\kid.jpg')
print(encode)
encode.shape


# ### To train the image_id 

# In[3]:


def img_id_train(train_images):
    
        file=open(train_images)
        data= file.readlines()
        
        train_img_id = [] 
        
        for img_id in data:
            train_img_id.append(img_id.split('.')[0])  #separating with '.' to extract image id
       
        return train_img_id


# ### To train the caption

# In[4]:


filename=(r'C:\Flickr8k\Flickr8k_dataset\train_images.txt') 

# Loading the file containg all the descriptions into memory
def load_descriptions(filename):
    file = open(filename)
    text = file.readlines()
    return text

# load descriptions 
doc= load_descriptions(filename)

def load_captions(doc,img_name):
    
    doc = load_descriptions(filename) #calling the load_descriptions function
    
    train_captions = {}    
    
    for line in doc:
        
        tokens = line.split()# split() returns lists of all the words in the string
        image_id, image_caption = tokens[0], tokens[2:-1]
        image_id =image_id.split('.')[0] # separating with '.' to extract image id 
        modified_caption = '<Begin> ' + ' '.join(image_caption) + '<End>'

       
        if(image_id not in train_captions):
            train_captions[image_id] = []
            
            train_captions[image_id].append(modified_caption)
    
    
    return train_captions


# In[5]:


TRAIN_IMAGE_TEXT = 'C:/Flickr8k/Flickr8k_dataset/train_images.txt'
VALID_IMAGE_TEXT = 'C:/Flickr8k/Flickr8k_dataset/val_images.txt'
TEST_IMAGE_TEXT = 'C:/Flickr8k/Flickr8k_dataset/testImages.txt'

train_img_name = img_id_train(TRAIN_IMAGE_TEXT)
valid_img_name = img_id_train(VALID_IMAGE_TEXT)
test_img_name  = img_id_train(TEST_IMAGE_TEXT)

train_captions = load_captions('clean_descriptions.txt', train_img_name)
valid_captions = load_captions('clean_descriptions.txt', valid_img_name)
test_captions = load_captions('clean_descriptions.txt', test_img_name)


# ## split captionCreate vocabulary set (each vocabulary is unique)

# In[6]:


from collections import Counter
all_train_captions = []
for captions in train_captions.values():
    for caption in captions:
        all_train_captions.append(caption)
        
all_valid_captions = []
for captions in valid_captions.values():
    for caption in captions:
        all_valid_captions.append(caption)
        
all_test_captions = []
for captions in test_captions.values():
    for caption in captions:
        all_test_captions.append(caption)

corpus = []
for caption in all_train_captions:
    for token in caption.split():
        corpus.append(token)

#count the repeat-word(vocab)       
descriptions = Counter(corpus)
print(len(descriptions))
word_count= descriptions.most_common(50)
print(word_count)

vocab = []
for token,count in descriptions.items():
    if(count>=1):
        vocab.append(token)
        


# In[7]:


data= word_count
df=pd.DataFrame(data,columns=['word','count'])

import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(20,6))
sns.barplot(x='word',y='count',data=df[:50])
plt.xticks(rotation='vertical')
plt.title("plot of count vs word")


# ### Read the pre-trained Embedding weights and create the Embedding matrix

# In[17]:


def max_len_caption(all_train_captions):   
    max_len = 0
    for caption in all_train_captions:
        max_len = max(max_len,len(caption.split()))
    print('Maximum length of caption= ',max_len)
    return max_len


# In[18]:


word_to_index = {}
index_to_word = {}
    
for idx,token in enumerate(vocab):
    word_to_index[token] = idx+1
    index_to_word[idx+1] = token

vocab_size = len(index_to_word) + 1

max_length_caption = max_len_caption(all_train_captions)


# In[19]:


embeddings_index = {}

#Glove-Global-vectors-for-word-representation or "GloVe" is an unsupervised learning algorithm for obtaining vector representations for words
file = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt', encoding="utf-8")

for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
    
file.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

embedding_matrix = np.zeros((vocab_size, embedding_dim))


# In[20]:


for word, i in word_to_index.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
        
print("Embedding_matrix.shape = ",embedding_matrix.shape)


# In[21]:


word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
print(len(word_to_idx))


# In[22]:


word_to_idx


# In[4]:


# data generator, intended to be used in a call to model.fit_generator()

def data_generator(descriptions, photos, word_to_idx, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            temp='../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'
            
            photo = photos[temp+key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [word_to_idx[word] for word in desc.split(' ') if word in word_to_idx]
                
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                   
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                  
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0


# In[8]:


def show_history_metrics(history, metrics_name=None):
    if metrics_name==None:
         print("No performance metrics specified")
    else:
        plt.figure(figsize=(30, 5))
        plt.subplot(121)
        plt.plot(history.history[metrics_name])
        plt.plot(history.history['val_'+metrics_name])
        plt.title('Model '+metrics_name)
        plt.ylabel(metrics_name)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        plt.show()


# In[9]:


def greedySearch(photo, model, max_length_caption, wordtoix, ixtoword):
    in_text = 'startseq'
    for i in range(max_length_caption):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length_caption)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[10]:


def define_model(max_length, vocab_size, input_shape=(2048,)):
    inputs1 = Input(shape=input_shape) 
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model


# # Create image_captioning model

# In[ ]:


model_ic = define_model(max_length_caption, vocab_size)
model_ic.summary()


# In[ ]:


print(model_ic.layers[2].input)
print(model_ic.layers[2].output)
model_ic.layers[2].set_weights([embedding_matrix])
model_ic.layers[2].trainable = False


# In[ ]:


LOSS = 'categorical_crossentropy'
OPTIM = 'adam'
METRICS = [K.metrics.CosineSimilarity(),
           K.metrics.Precision()]


# In[ ]:


model_ic.compile(loss=LOSS, optimizer=OPTIM, metrics=METRICS)


# In[11]:


def Predict_test_caption(test_image_id, model, show_predict=True, CNN_units_num=2048):
    print("image_id:"+test_image_id)
    test_image_filename = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'+test_image_id+'.jpg'
    image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
    pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
    true_captions = preprocessed_map[test_image_id]
    if show_predict:
        x=plt.imread(test_image_filename)
        plt.imshow(x)
        plt.show()
        print("Predict:\n"+pred_caption)
        print("True:")
        print(*preprocessed_map[test_image_id],sep='\n')
    return pred_caption, true_captions


# In[12]:


def compute_BLEU(pred_caption, true_captions, show_bleu=True): 
    bleu = [0.0, 0.0, 0.0, 0.0]
    references = [true_captions[0].split(),true_captions[1].split(),true_captions[2].split(),true_captions[3].split(),true_captions[4].split()]
    hypothesis = pred_caption.split()
    smooth = SmoothingFunction()
    bleu[0] = sentence_bleu(references, hypothesis, weights=(1.0, 0, 0, 0), smoothing_function=smooth.method1)
    bleu[1] = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    bleu[2] = sentence_bleu(references, hypothesis, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth.method1)
    bleu[3] = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
    if show_bleu:
        print('BLEU-1: %f' % bleu[0])
        print('BLEU-2: %f' % bleu[1])
        print('BLEU-3: %f' % bleu[2])
        print('BLEU-4: %f' % bleu[3])
        
    return bleu


# In[13]:


def evaluate_BLEU(encoding_features, images_name, model, show_results=True, CNN_units_num=2048):
    mean_bleu = np.zeros(4)
    for test_id in iter(images_name):
        test_image_filename = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'+test_id+'.jpg'
        image = encoding_test[test_image_filename].reshape((1,CNN_units_num))
        pred_caption = greedySearch(image, model, max_length_caption, word_to_index, index_to_word)
        true_captions = preprocessed_map[test_id]
         
        bleu = compute_BLEU(pred_caption, true_captions, show_bleu=False)
        bleu_temp = np.array(bleu)
        mean_bleu = mean_bleu + bleu_temp
    
    mean_bleu = mean_bleu/len(images_name)
    if show_results:
        print('MEAN_BLEU-1: %f' % mean_bleu[0])
        print('MEAN_BLEU-2: %f' % mean_bleu[1])
        print('MEAN_BLEU-3: %f' % mean_bleu[2])
        print('MEAN_BLEU-4: %f' % mean_bleu[3])
    return mean_bleu    


# In[14]:


path = '../input/flicker8k-dataset/Flickr8k_Dataset/Flicker8k_Dataset/'
all_images_name = images_name(path)
train_image_full_name = [path+img+'.jpg' for img in train_img_name]
valid_image_full_name = [path+img+'.jpg' for img in valid_img_name]
test_image_full_name = [path+img+'.jpg' for img in test_img_name]

encoding_train = encode_image_feature(train_image_full_name, model_new)
encoding_valid = encode_image_feature(valid_image_full_name, model_new)
encoding_test = encode_image_feature(test_image_full_name, model_new)


# In[ ]:





# In[ ]:


tensorflow.keras.preprocessing.sequence(sequences,
                        maxlen=None,dtype='int32',padding='pre',truncating='pre',value=0.0)


# In[ ]:





# In[ ]:


def encode_image(img):
    img= preprocess_image(img)
    predict =new_model.predict(img)
    predict=predict.reshape(1,predict.shape[1])
    return predict


# In[ ]:


encode=encode_image('C:\Flickr8k\Flickr8k_dataset\LittleGirl.jpg')
encode


# In[ ]:


encode.shape


# In[ ]:


word_to_idx=pickle.load(open('word_to_idx.pkl','rb'))
print(len(word_to_idx))


# In[ ]:


word_to_idx


# In[ ]:


idx_to_word=pickle.load(open('idx_to_word.pkl','rb'))
print(len(idx_to_word))


# In[ ]:


idx_to_word


# In[ ]:


def predict_caption(photo):
  
    in_text='<star>'
    
    max_len=35
    
    
    for i in range(max_len):
        sequence= [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence= tf.keras.preprocessing.sequence.pad_sequences([sequence],maxlen= max_len,padding='post')
        
        pred=model.predict([photo,sequence])
        pred=pred.argmax()
        word=idx_to_word[pred]
        in_text+=''+word
        
        if word=='<end>':
            break
            
       
    caption=in_text.split()
    caption=caption[1:-1]
    caption=''.join(caption)       
    return caption


# In[ ]:


encode=encode_image("C:\Flickr8k\Flickr8k_dataset\LittleGirl.jpg")
print(encode)


# # Training Testing Data

# In[ ]:


train_images=open(r'C:\Flickr8k\Flickr8k_dataset\train_images.txt') 
train=train_images.read()
images_id = [i.split(".")[0] for i in train.split('\n')]
images_id


# In[ ]:


#load the testing data 
test_images= open(r"C:\Flickr8k\Flickr8k_dataset\testImages.txt")
test = test_images.read().split('\n')
for i in test:
    image_id=[i.split('.')[0]]
    print(image_id)


# In[ ]:





# In[ ]:


#Validation data
val_images=open(r"C:\Flickr8k\Flickr8k_dataset\val_images.txt")
val= val_images.read().split('\n')
for i in val:
    


# In[ ]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')

def load_descriptions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# load descriptions 
text= load_descriptions(filename)

captions=doc.split('\n')

# creating a "descriptions" dictionary  where key is 'img_id' and value is list of captions corresponding to that image_file.
def image_to_caption(text):
    descriptions = {}

    for img_to_cap in captions:
        captions= img_to_cap.split("\t")
        image_id=captions[0].split(".")[0] # separating with '.' to extract image id
    
    if descriptions.get(image_id) == None:
        descriptions[image_id] = [captions]
    else:
        descriptions[image_id].append(captions)
        
        return descriptions


# In[ ]:


filename = open(r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
caption=filename.read().split('\n')
caption
def cleaning_text(caption):
    preprocessed_caption={}
    #create a translation table,Remove all punctuation from each token.  
    table = str.maketrans('','',string.punctuation)
    #here key is 'img_id' and value is list of captions corresponding to that caption_file.

    for key in caption.keys():
        for idx in range(len(caption[key])):
            token=caption[key][idx].split()
            #converts to lowercase
            token= [word.lower() for word in token]
            #remove punctuation from each token
            token = [word.translate(table) for word in token]
            #remove hanging 's and a 
            token = [word for word in token if(len(word)>1)]
            #remove tokens with numbers in them
            token= [word for word in token if(word.isalpha())]
            #convert back to string
            caption[key][idx] = ' '.join(token)
        
            return caption
        


# In[ ]:


with open("C:\Flickr8k\Flickr8k_dataset\storage\encoded_test_images.pkl", "rb") as encoded_test_pickle:
    encoding_test = pickle.load(encoded_test_pickle)


# In[ ]:





# from time import time
# start = time()
# 
# encoding_test = {}
# 
# for i, img in enumerate():
# 
#     img = "C:\Flickr8k\Flickr8k_dataset\Images{}.jpg".format(test[i])
#     encoding_test[img[len(images):]] = encode_image(img)
#     
#     if i%100==0:
#         print("Encoding image- "+ str(i))
#     
# print("Time taken in seconds =", time()-start)

# 
# for i in range(20):
#     rn =  np.random.randint(0, 1000)
#     img_name = list(encoding_test.keys())[rn]
#     photo = encoding_test[img_name].reshape((1,2048))
# 
#     i = pltimg.imread(images+img_name)
#     plt.imshow(i)
#     plt.axis("off")
#     plt.show()
# 
#     caption = predict_caption(photo)
#     print(caption)

# In[ ]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')

def load_descriptions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

doc=load_descriptions(filename)
caption=doc.split('\n')

# creating a "descriptions" dictionary  where key is 'img_id' and value is list of captions corresponding to that image_file.
def image_to_caption(text):
    descriptions = {}
    # load descriptions 
    text= load_descriptions(filename)
    des=text.split('\n')[:-1]

    for img_to_cap in des:
        captions= img_to_cap.split("\t")
        image_id=captions[0].split(".")[0] # separating with '.' to extract image id
    
    if descriptions.get(image_id) == None:
        descriptions[image_id] = []
    else:
        descriptions[image_id].append(captions)
        
        return descriptions


# In[ ]:



filename = ('C:\Flickr8k\Flickr8k_dataset\captions.txt')

def load_descriptions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
doc=load_descriptions(filename)
caption=doc.split('\n')
import string
for word in caption:
    #create a translation table,Remove all punctuation from each token.
    table = str.maketrans('', '', string.punctuation)
    #remove punctuation from each token
    token = word.translate(table)
    print(token)
    


# In[ ]:


for word in caption:
    lowercase=word.lower()
    print(lowercase)


# In[ ]:


def cleaning_text(caption):
    preprocessed_caption=[]
    #create a translation table,Remove all punctuation from each token.  
    table = str.maketrans('','',string.punctuation)
    #here key is 'img_id' and value is list of captions corresponding to that caption_file.

    for key in caption.keys():
        for idx in range(len(caption[key])):
            token=caption[key][idx].split()
            #converts to lowercase
            token= [word.lower() for word in token]
            #remove punctuation from each token
            token = [word.translate(table) for word in token]
            #remove hanging 's and a 
            token = [word for word in token if(len(word)>1) if(word.isalpha())]
            #convert back to string
            caption[key][idx] = ' '.join(token)
            
            return caption


# In[ ]:


filename = (r'C:\Flickr8k\Flickr8k_dataset\captions.txt')
text=load_descriptions(filename)
caption=image_to_caption(text)
caption['1000268201_693b08cb0e']


# In[ ]:





# In[ ]:





# In[ ]:


def preprocess(map_img_to_captions):
    preprocessed_captions = []
    for key in map_img_to_captions.keys():
        for idx in range(len(map_img_to_captions[key])):
            tokens = map_img_to_captions[key][idx].split()
            tokens = [token.lower() for token in tokens if len(token)>1 if token.isalpha()]
            map_img_to_captions[key][idx] = ' '.join(tokens)
            
    return map_img_to_captions


# In[ ]:


def create_vocabulary(preprocessed_map):
    vocabulary = set()
    for img_captions in preprocessed_map.values(): # list of 5 captions for each image
        for caption in img_captions:
            for token in caption.split():
                vocabulary.add(token)    
    return vocabulary


# In[ ]:


text= load_descriptions(filename)
map_img_to_captions = image_to_caption(text)
map_img_to_captions['1000268201_693b08cb0e']


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(20,6))
sns.barplot(x="word",y="count",data=)
plt.xticks(rotation='90')
plt.title("Plot of count vs words")


# In[ ]:


all_vocab=[]
for key in clean_descriptions.keys():
     for desc in clean_descriptions[key]:
          # Iterating through each line of the descriptions
          for i in desc.split():
            all_vocab.append(i)
            print('All vocab size: %d' % len(all_vocab))
            print(all_vocab[1:15])  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




