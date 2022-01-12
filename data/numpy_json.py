# Links:
# https://pynative.com/python-serialize-numpy-ndarray-into-json/
# Other links:
# https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def encode_array(np_array):
    encodedNumpyData = json.dumps(np_array, cls=NumpyArrayEncoder)
    return encodedNumpyData

def decode_array(encoded_array):
    decodedArrays = json.loads(encoded_array)
    return decodedArrays

# numpyArrayOne = numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])

# # Serialization
# numpyData = {"array": numpyArrayOne}
# encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
# print("Printing JSON serialized NumPy array")
# print(encodedNumpyData)

# # Deserialization
# print("Decode JSON serialized NumPy array")
# decodedArrays = json.loads(encodedNumpyData)

# finalNumpyArray = numpy.asarray(decodedArrays["array"])
# print("NumPy Array")
# print(finalNumpyArray)