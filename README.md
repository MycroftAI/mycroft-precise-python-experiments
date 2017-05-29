# Mycroft Precise #

Mycroft Precise is a wake word listener using the latest machine learning and audio processing technology available.

## How it works ##

First, using [librosa][librosa], it extracts `20` [mfcc features][mfcc] from the input audio. Next, it creates an [LSTM network][lstm] using [tflearn][tflearn]. This model is saved to the `model/` folder and can be loaded to test for the wake word against new audio.

## Current State ##

Currently, we are in the process of collecting and tagging data to train the network and assess how well it generalizes.

## File Descriptions ##

 - `train_keyword.py`: Reads wav files in the `data` folder and creates/trains the model. Every `20` epochs you will see the script saves the model to the `model` folder. Hit `Ctrl+C` to stop training. The next time the script is run it will continue training where it left off.
 - `test_keyword.py`: Uses the audio in `data/test` and the model saved in `model/` to run some statistics on the accuracy of the model.
 - `collect_data.py`: Allows quickly recording wav files.
 - `mycroft_keyword.py`: Used internally to architect the network and to perform common tasks like loading training data.
 
## Setup ##

 - Install `python3`. (Ubuntu: `sudo apt-get install python3-pip`)
 - Install dependencies, ideally in a virtualenv: (`sudo` if installing system wide) `pip3 install -r requirements.txt`

That's it! Now putt some data in the data folder, run the script, and you should be good to go!

[tflearn]:http://tflearn.org/
[librosa]:https://github.com/librosa/librosa
[mfcc]:https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
[lstm]:https://en.wikipedia.org/wiki/Long_short-term_memory

