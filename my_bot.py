import os
import telebot
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.keras.models import load_model


with open("token.txt") as f:
  TOKEN = f.read().strip()
bot = telebot.TeleBot(TOKEN, parse_mode=None)

# Load the pre-trained audio classification model
model = tf.keras.models.load_model('weights_audio_classification.h5')

# Define the audio labels
labels = ['parisa','davood','javad','khadijeh','kiana','mona','matin','mohammad_parvari','mohammad','azra','nima','omid','abdollah','shima','maryam','sajedeh','parsa','amirhossein','melika','mohadeseh','nahid','tara']

@bot.message_handler(commands=['start'])

def send_welcome(message):
  bot.reply_to(message, "Hello " + str(message.chat.first_name) + ", welcome!"+" Send me a voice message to get started.")

@bot.message_handler(content_types=['voice'])

def handle_voice(message):
  file_id = message.voice.file_id

  file_info = bot.get_file(file_id)
  file_path = file_info.file_path
  downloaded_file = bot.download_file(file_path)

  with open(file_path, 'wb') as new_file:
    new_file.write(downloaded_file)

  wav, _ = librosa.load(file_path, sr=None, mono=True, duration=30, offset=0.0)

  length = 64000
  resized_wavefrom = librosa.util.fix_length(wav, size=length)

  input_data = np.expand_dims(resized_wavefrom, axis=-1)
  input_data = np.expand_dims(resized_wavefrom, axis=0)

  pred = model.predict(input_data)
  label = np.argmax(pred)
  predicted_label = labels[label]

  bot.reply_to(message, f'The voice belongs to {predicted_label}')
  print(f'The voice belongs to {predicted_label}')


bot.polling()