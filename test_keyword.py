#!/usr/bin/env python3

from __future__ import division, print_function, absolute_import
import mycroft_keyword as kw

ins, outs = kw.load_test_data(kw.load_length())
net = kw.create_net(*ins[0].shape, *outs[0].shape)
kw.fix_version_errors()
model = kw.create_model(net)
if not kw.try_load_into_model(model):
	raise RuntimeError

_outs = model.predict(ins)

print (outs)
print (_outs)

num_correct, num_incorrect = 0, 0
num_false_pos, num_false_neg = 0, 0
for i in range(len(outs)):
	if (_outs[i][0] > _outs[i][1]) == (outs[i][0] > outs[i][1]):
		num_correct += 1
	else:
		if _outs[i][0] > _outs[i][1]:
			num_false_neg += 1
		else:
			num_false_pos += 1
		num_incorrect += 1
total = num_incorrect + num_correct

print(str(num_correct) + " out of " + str(total))
print(str(100.0 * num_correct / total) + " %")
print
print(str(100.0 * num_false_pos / total) + " % false positives")
print(str(100.0 * num_false_neg / total) + " % false negatives")

