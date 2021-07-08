# based on monero-powpy/solo-bock.py
# adapted for Monu
#   https://github.com/monu-fork
import binascii
import struct
import requests
import json

rpc_url = 'http://127.0.0.1:36061/json_rpc'
wallet_address = ''
final_weights = []

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
height  = result.get('height')
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
    'params':{"start_height":height, "end_height":height-ndiff}
}
req     = requests.post(rpc_url, json=payload)
result  = req.json().get('result')

# train neural network here.
#

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
