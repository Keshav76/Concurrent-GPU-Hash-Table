import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "Books_5_part0.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sayedmahmoud/amazanreviewscor5",
  file_path
)
print("First 5 records:", df.head())
df = df[['asin', 'reviewText']]
df.head()
import hashlib

def hash_string_32bit(s):
    s = str(s)
    md5_hash = hashlib.md5(s.encode('utf-8')).digest()
    return int.from_bytes(md5_hash[:4], 'little', signed=False)

print(hash_string_32bit("hello world"))

df['asin'] = df['asin'].apply(hash_string_32bit)
df['reviewText'] = df['reviewText'].apply(hash_string_32bit)

import pandas as pd

with open("keys.txt", "w") as f:
    f.write(" ".join(map(str, df['asin'].values)))

with open("values.txt", "w") as f:
    f.write(" ".join(map(str, df['reviewText'].values)))


import struct

def txt_to_bin(txt_file, bin_file):
    with open(txt_file, 'r') as f_in:
        data = f_in.read().split()
    
    # Assuming the data is a series of unsigned integers
    # You can adjust the format if it's a different data type
    with open(bin_file, 'wb') as f_out:
        for value in data:
            # Convert each value to an unsigned int (little-endian format)
            f_out.write(struct.pack('I', int(value)))  # 'I' is for unsigned int


# Usage
import struct

def txt_to_bin(txt_file, bin_file):
    with open(txt_file, 'r') as f_in:
        data = f_in.read().split()
    
    # Assuming the data is a series of unsigned integers
    # You can adjust the format if it's a different data type
    with open(bin_file, 'wb') as f_out:
        for value in data:
            # Convert each value to an unsigned int (little-endian format)
            f_out.write(struct.pack('I', int(value)))  # 'I' is for unsigned int

# Usage
txt_to_bin('values.txt', 'values.bin')
txt_to_bin('keys.txt', 'keys.bin')
