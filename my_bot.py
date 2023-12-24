
import os
import telebot
import librosa
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.keras.models import load_model

TOKEN = None

with open("token.txt") as f:
    TOKEN = f.read().strip()
bot = telebot.TeleBot(TOKEN, parse_mode=None)

# Load the pre-trained audio classification model
model = tf.keras.models.load_model('model\weights_audio_classification.h5')

# Define the audio labels
labels = ['parisa','davood','javad','khadijeh','kiana','mona','matin','mohammad_parvari','mohammad','azra','nima','omid','abdollah','shima','maryam','sajedeh','parsa','amirhossein','melika','mohadeseh','nahid','tara']



@bot.message_handler(commands=['start'])

def send_welcome(message):
    bot.reply_to(message, "Hello " + str(message.chat.first_name)+ ", welcome!"+" Send me a voice message to get started.")


#  Need to download a audiofile from telegram bot and save in project folder.
#  Get the audio file from the message
@bot.message_handler(content_types=['audio'])

def handle_audio(message):
    # Get the file ID of the voice message
    file_id = message.audio.file_id
    
    # Download the voice message file from Telegram servers
    file_info =bot.get_file(file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)

   # Save the audio file to disk
    file_name = f"{file_id}.{file_info.file_path.split('.')[-1]}"
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)


    # Convert the audio file to WAV format
    sound = AudioSegment.from_file(file_name)
    wav_file_name = f"{file_id}.wav"
    sound.export(wav_file_name, format="wav")

    
    # Perform audio classification on the WAV file
    y, sr = librosa.load(wav_file_name, mono=True, duration=30, offset=0.0, res_type='kaiser_fast')
    if len(y) < 64000:
        y = np.pad(y, (0, 64000 - len(y)), 'constant')
    else:
        y = y[:64000]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    mfccs_scaled = mfccs_scaled.reshape(1,-1)
    prediction = model.predict(mfccs_scaled)
    label = labels[np.argmax(prediction)]

 

    # Send a reply message to the user with the predicted label
    bot.reply_to(message,  f'The voice belongs to {label}.')
    

    os.remove(file_name)
    os.remove(wav_file_name)

    
# Start the bot
bot.polling()
