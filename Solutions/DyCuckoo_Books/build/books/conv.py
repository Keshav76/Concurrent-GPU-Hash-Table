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
txt_to_bin('keys.txt', 'keys.bin')
txt_to_bin('values.txt', 'values.bin')
