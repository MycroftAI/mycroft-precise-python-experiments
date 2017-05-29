#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import
import mycroft_keyword as kw

epochs_per_step = 20

inputs, outputs = kw.load_training_data(kw.load_length())
net = kw.create_net(*inputs[0].shape, *outputs[0].shape)
kw.fix_version_errors()
model = kw.create_model(net)
kw.try_load_into_model(model)

while True:
	kw.train_model(model, inputs, outputs, 20)
	print("Saving...")
	kw.save_model(model, len(inputs[0][0]))

