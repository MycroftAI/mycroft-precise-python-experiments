#!/usr/bin/env python3

from sys import byteorder
from array import array
from struct import pack
from os.path import isfile
import curses

import pyaudio
import wave

from sys import stdin
from select import select
import tty
from termios import tcsetattr, tcgetattr, TCSADRAIN

def key_pressed():
	return select([stdin], [], [], 0) == ([stdin], [], [])

orig_settings = None
def termios_wrapper(main):
	global orig_settings
	orig_settings = tcgetattr(stdin)
	try:
		hide_input()
		main()
	finally:
		tcsetattr(stdin, TCSADRAIN, orig_settings)

def show_input():
	tcsetattr(stdin, TCSADRAIN, orig_settings)

def hide_input():
	tty.setcbreak(stdin.fileno())

CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

RECORD_KEY = ' '
EXIT_KEY_CODE = 27

def record_until(p, should_return):
	stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
		input=True, frames_per_buffer=CHUNK_SIZE)

	frames = []
	while not should_return():
		frames.append(stream.read(CHUNK_SIZE))

	stream.stop_stream()
	stream.close()

	return b''.join(frames), p.get_sample_size(FORMAT)

def save_audio(name, data, width):
	wf = wave.open(name, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(width)
	wf.setframerate(RATE)
	wf.writeframes(data)
	wf.close()

def next_name(name):
	name += '.wav'
	pos, num_digits = None, None
	try:
		pos = name.index('#')
		num_digits = name.count('#')
	except ValueError:
		print("Name must contain at least one # to indicate where to put the number.")
		raise

	def get_name(i):
		nonlocal name, pos
		return name[:pos] + str(i).zfill(num_digits) + name[pos + num_digits:]

	i = 0
	while True:
		if not isfile(get_name(i)):
			break
		i += 1

	return get_name(i)

def wait_to_continue():
	while True:
		c = stdin.read(1)
		if c == RECORD_KEY:
			return True
		elif ord(c) == EXIT_KEY_CODE:
			return False

def main():
	def should_return():
		return key_pressed() and stdin.read(1) == RECORD_KEY

	show_input()
	audio_name = input("Audio name (Ex. recording-##):")
	hide_input()

	p = pyaudio.PyAudio()

	while True:
		print('Press space to record (esc to exit)...')

		if not wait_to_continue():
			break

		print('Recording...')
		d, w = record_until(p, should_return)
		name = next_name(audio_name)
		save_audio(name, d, w)
		print('Saved as ' + name)

	p.terminate()

if __name__ == '__main__':
	termios_wrapper(main)

	
