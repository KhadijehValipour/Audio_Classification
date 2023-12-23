
import os
import telebot
import librosa
import numpy as np
from tensorflow.keras.models import load_model


TOKEN = None

with open("token.txt") as f:
    TOKEN = f.read().strip()
bot = telebot.TeleBot(TOKEN, parse_mode=None) # You can set parse_mode by default. HTML or MARKDOWN

@bot.message_handler(commands=['start'])

def send_welcome(message):
    bot.reply_to(message, "Hello" + str(message.chat.first_name)+ ", welcome"+" Send me a voice message to get started.")


#  Need to download a audiofile from telegram bot and save in project folder.
#  Get the audio file from the message
@bot.message_handler(content_types=['voice'])

def handle_voice_message(message):
    # Get the file ID of the voice message
    file_id = message.voice.file_id
    
    # Download the voice message file from Telegram servers
    file_info =bot.get_file(file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)

    # Save the downloaded file to disk as a .ogg file
    file_name = 'audio.ogg'
    with open(file_name, 'wb') as f:
        f.write(downloaded_file)
    
    # Convert the .ogg file to a .wav file
    y, sr = librosa.load(file_name, sr=None)
    y_16k = librosa.resample(y, sr, res_type='kaiser_best', scale=True)

    output_sequence_length = 64000
    if len(y_16k) < output_sequence_length:
        y_16k = np.pad(y_16k, (0, output_sequence_length - len(y_16k)), mode='constant')
    else:
        y_16k = y_16k[:output_sequence_length]
    librosa.output.write_wav('audio.wav', y_16k, sr=16000)


    # Load the pre-trained model
    model = load_model('model\weights_audio_classification.h5')

    # Get the list of labels
    labels = os.listdir('dataset')

    # Make a prediction using the pre-trained model
    input_data, _ = librosa.load('audio.wav', sr=None)
    input_data = np.expand_dims(input_data, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)
    pred = model.predict(input_data)
    label_index = np.argmax(pred)
    label = labels[label_index]

    # Send a reply message to the user with the predicted label
    bot.reply_to(message,  f'The voice belongs to {label}.')

# Start the bot
bot.polling()
