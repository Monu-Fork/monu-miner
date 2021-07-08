# https://github.com/monu-fork
#   functional example miner
#   using Tensorflow Keras

import binascii
import struct
import requests
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from time import time_ns


rpc_url = 'http://127.0.0.1:36061/json_rpc'
wallet_address = ''
final_weights = []

training_iterations = 33333
batches = 96


# convert hexadecimal characters to normalised embeddings
def toEmbed(ic):
    c = ic.upper()
    if(c == '0'):   return -0.8828125
    elif(c == '1'): return -0.765625
    elif(c == '2'): return -0.6484375
    elif(c == '3'): return -0.53125
    elif(c == '4'): return -0.4140625
    elif(c == '5'): return -0.296875
    elif(c == '6'): return -0.1796875
    elif(c == '7'): return -0.0625
    elif(c == '8'): return  0.0546875
    elif(c == '9'): return  0.171875
    elif(c == 'A'): return  0.2890625
    elif(c == 'B'): return  0.40625
    elif(c == 'C'): return  0.5234375
    elif(c == 'D'): return  0.640625
    elif(c == 'E'): return  0.7578125
    elif(c == 'F'): return  0.875

def pack_weights(blob, weights):
    b = binascii.unhexlify(blob)
    bin = struct.pack('43B', *bytearray(b[:43]))
    bin += struct.pack('33280B', weights)
    bin += struct.pack('{}B'.format(len(b)-33323), *bytearray(b[33323:]))
    return bin

payload = {
    'jsonrpc':'2.0',
    'id':'0',
    'method':'get_block_template',
    'params': {'wallet_address': wallet_address}
}

print('~~ Fetching block template')
req     = requests.post(rpc_url, json=payload)
result  = req.json().get('result')
btb     = result.get('blocktemplate_blob')
diff    = result.get('difficulty')
height  = result.get('height')-1
max128  = pow(2, 128)-1
ndiff   = (height / max128) * diff # this is the difficulty we need
print('~~ Target difficulty: {}'.format(ndiff))

# use get_block_headers_range to get all blocks headers
# by height so that we can extract the hashes and train
# the neural network with them.
# https://www.getmonero.org/resources/developer-guides/daemon-rpc.html#get_block_headers_range
payload = {
    'jsonrpc':'2.0',
    'id':'0',
    'method':'get_block_headers_range',
    'params':{"start_height":height-ndiff, "end_height":height}
}
req     = requests.post(rpc_url, json=payload)
result  = req.json().get('result')


# convert result to x and y fit arrays
x_train = []
y_train = []
# TODO: Finish this segement


# construct neural network
model = Sequential()
model.add(Dense(64, activation='tanh', input_dim=64))
model.add(Dense(64, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')

# train network
st = time_ns()
model.fit(x_train, y_train, epochs=training_iterations, batch_size=batches)
timetaken = (time_ns()-st)/1e+9
print("~~ Time Taken:", "{:.2f}".format(timetaken), "seconds")

#output weights in the correct format for Monu block header
for layer in model.layers:
    total_layer_weights = layer.get_weights()[0].flatten().shape[0]
    total_layer_units = layer.units
    layer_weights_per_unit = total_layer_weights / total_layer_units
    wc = 0
    bc = 0
    if layer.get_weights() != []:
        for weight in layer.get_weights()[0].flatten():
            wc += 1
            final_weights.append(bytearray(struct.pack("f", weight)))
            if wc == layer_weights_per_unit:
                final_weights.append(bytearray(struct.pack("f", layer.get_weights()[1].flatten()[bc])))
                wc = 0
                bc += 1

# submit back the trained weights
btb = binascii.hexlify(pack_weights(btb, final_weights))
payload = {
    'jsonrpc':'2.0',
    'id':'0',
    'method':'submit_block',
    'params': [ btb ]
}
print('~~ Submitting block')
print(payload)
req     = requests.post(rpc_url, json=payload)
result  = req.json()
print('~~ Response')
print(result)

# sudo apt install nvidia-cuda-toolkit
# sudo apt install nvidia-driver-465
# sudo apt install python3
# sudo apt install python3-pip
# sudo pip3 install --upgrade pip
# pip3 install tensorflow-gpu
# sudo ln /usr/lib/x86_64-linux-gnu/libcusolver.so.10.6.0.245 /usr/lib/x86_64-linux-gnu/libcusolver.so.11

# Then download and install the runtime and developer debs from here:
# https://developer.nvidia.com/rdp/cudnn-download

# nvidia-driver-465 has to be installed after nvidia-cuda-toolkit or the toolkit overwrites
# the nvidia-smi program. nvidia-settings can also be a helpful utility to install.

# .. although training on a GPU does not seem to yeild a significant performance gain.
