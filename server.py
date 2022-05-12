from operator import xor
import numpy as np
import random
import cv2
import xlsxwriter
import os

# Bit counters for individual scramblers
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

key = get_random_bytes(16)
iv = get_random_bytes(16)
counter_dvb = [0, 0]
counter_v34 = [0, 0]
counter_x16 = [0, 0]
data_counter = [0, 0]

counter_dvb_diffrent_bits = []
counter_v34_diffrent_bits = []
counter_x16_diffrent_bits = []
counter_data_diffrent_bits = []

counter_dvb_longest_sequence = [0, 0]
counter_v34_longest_sequence = [0, 0]
counter_x16_longest_sequence = [0, 0]
counter_data_longest_sequence = [0, 0]

switch_intensity = 1

# Clock method for additive scrambler, XOR feedback for framebits and input signal
def sync_clock(frame, data, bit):
    if bit[1] != -1:  # Checking whether we use both bits required for some scramblers
        temp = xor(frame[bit[0] - 1], frame[bit[1] - 1])  # XOR for bit[0] and bit[1]
    else:  # If there is only 1 bit, value is assigned to this bit
        temp = frame[bit[0] - 1]
    frame.pop()  # Remove the last bit from the frame
    frame.insert(0, temp)  # Add XOR result at the beginning
    xor_value = xor(temp, data)  # Feedback of input syganle and XOR values of frame bits
    return xor_value  # Return result of the coded signal


# Clock method for multiplicative scrambler, XOR feedback for frame bits and input signal
def async_clock(frame, data, bit):
    if bit[1] != -1:  # Checking whether we use both bits required for some scramblers
        temp = xor(frame[bit[0] - 1], frame[bit[1] - 1])  # XOR for bit[0] and bit[1]
    else:  # If there is only 1 bit, value is assigned to this bit
        temp = frame[bit[0] - 1]
    frame.pop()  # Remove the last bit from the frame
    xor_value = xor(temp, data)  # Feedback of input syganle and XOR values of frame bits
    frame.insert(0, xor_value)  # Add XOR value at the beginning
    return xor_value  # Return result of the coded signal


# Clock method for multiplicative desramblers
def reverse_async_clock(frame, data, bit):
    if bit[1] != -1:  # Checking whether we use both bits required for some scramblers
        temp = xor(frame[bit[0] - 1], frame[bit[1] - 1])  # XOR for bit[0] and bit[1]
    else:  # If there is only 1 bit, value is assigned to this bit
        temp = frame[bit[0] - 1]
    frame.pop()  # Remove the last bit from the frame
    frame.insert(0, data)  # Adding a signalbit at the beginning of the frame
    xor_value = xor(data, temp)  # XOR for the input synganle bit and the previous XOR
    return xor_value  # Return result of the coded signal


# DVB additive scrambler
def scram_DVB(bits):
    data_length = len(bits)  # Length of input signal
    frame_DVB = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]  # Synchronization frame for scrambler
    scram_bit = [len(frame_DVB),
                len(frame_DVB) - 1]  # Bits for feedback – for DVB this is the last and penultimate bit
    output_signal = []  # Array for output data
    for i in range(0, data_length):  # Iteration over the entire input array
        clock_result = sync_clock(frame_DVB, bits[i], scram_bit)  # Perform clock operations for additive scrambler
        output_signal.append(clock_result)  # Add results to the output array
    return output_signal


# V34 multiplicative scrambler 
def scram_V34(bits):
    data_length = len(bits)
    frame_V34 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # Synchronization frame for scrambler
    scram_bits = [18, 23]  # Bits used in feedback - for V34 bit 18 and 23
    output_signal = []  # Array for output data
    for i in range(0, data_length):  # Iteration over the entire input array
        clock_result = async_clock(frame_V34, bits[i], scram_bits)  # Perform clock operations for multiplicative scrambler
        output_signal.append(clock_result)  # Add results to the output array
    return output_signal


# V34 multiplicative desrambler
def descram_V34(bits):
    data_length = len(bits)
    frame_V34 = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # Synchronization frame for scrambler
    scram_bits = [18, 23]  # Bits used in feedback - for V34 bit 18 and 23
    output_signal = []  # Array for output data
    for i in range(0, data_length):
        clock_result = reverse_async_clock(frame_V34, bits[i], scram_bits)  # Decoding operation
        output_signal.append(clock_result)  # Add results to the output array
    return output_signal


#  x^16+1 additive scrambler
def scram_X16(bits):
    data_length = len(bits)
    frame_X16 = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # Synchronization frame for scrambler
    scram_bit = [16, -1]  # Bits used in the feedback - for x16 bit 16, -1 if second bit is missing
    output_signal = []  # Array for output data
    for i in range(0, data_length):
        clock_result = sync_clock(frame_X16, bits[i], scram_bit)  # Perform clock operations for additive scrambler
        output_signal.append(clock_result)  # Add results to the output array
    return output_signal


# Counting the number of bits
def sum_of_bits(bits, counter):
    for i in range(0, len(bits)):
        if bits[i] == 0:
            counter[0] += 1
        else:
            counter[1] += 1


def split(word):
    return [char for char in word]


def encryption(data, array):
    cipher = AES.new(key, AES.MODE_CBC, IV = iv)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    b = split(ct_bytes)
    results = list(map(int, b))
    bits = []
    image_to_bits(results, bits)
    bits_to_bytes(bits, array)
    if not(os.path.exists('Output/AES')): # Check whether the specified path exists or not
        os.makedirs('Output/AES')
    cv2.imwrite('Output/AES/AES_encryption.jpg', array)  # Save image after encryption

    cipher2 = AES.new(key, AES.MODE_CBC, IV = iv)
    pt = unpad(cipher2.decrypt(ct_bytes), AES.block_size).decode("utf-8")
    desired_array = [int(numeric_string) for numeric_string in pt]
    bits_to_bytes(desired_array, array)
    cv2.imwrite('Output/AES/AES_descryption.jpg', array)  # Save image after decryption


# Convert image bytes to bits
def image_to_bits(data, bits):  # data -> bytes, bits -> Target bit storage
    for i in range(0, len(data)):
        # Convert bytes to bits
        current_byte = format(data[i], '08b')

        bit_array = []
        for j in range(0, len(current_byte)):
            bit_array.append(int(current_byte[j]))
        # Add bits of the investigated byte to the bit array
        for k in range(0, len(bit_array)):
            bits.append(bit_array[k])


# Swap bits after too long a sequence of a single bit
def switch_bits(bits, switch_chance, counter, do_count):
    max_zeros_amount = 0
    max_ones_amount = 0
    amount = 0  # Number of the same bits in sequence
    chance = 0  # Chance to skip a bit
    is_zero_now = True  # Used to count the number of the same bits in sequence
    index = 5  # Loop index

    while index < len(bits):  # For each bit, index -> bit
        if (bits[index] == 0 and is_zero_now) or (bits[index] == 1 and not is_zero_now):  # Counting
            amount += 1
            chance = chance + (0.0075 * switch_chance * (amount / 2.0))
            if do_count:
                if is_zero_now and max_zeros_amount < amount:
                    max_zeros_amount = amount
                elif (not is_zero_now) and max_ones_amount < amount:
                    max_ones_amount = amount
        else:
            is_zero_now = not is_zero_now
            amount = 0
            chance = 0

        # Skips one bit when the correct number is drawn
        rand = random.uniform(1.0, 100.0)
        if rand <= chance:
            if bits[index] == 0:
                bits.pop(index)
                bits.insert(index, 1)
            else:
                bits.pop(index)
                bits.insert(index, 0)
        index += 1
    if do_count:
        counter[0] = max_zeros_amount
        counter[1] = max_ones_amount


def bits_to_bytes(bits, array):  # Convert back to bytes
    bit_holder = []
    bit_counter = 0
    data = []
    for i in range(0, len(bits)):
        bit_holder.append(bits[i])
        bit_counter += 1
        if bit_counter == 8:
            string = ''
            for j in range(0, len(bit_holder)):
                string += str(bit_holder[j])
            data.append(int(string, 2))
            bit_counter = 0
            bit_holder = []

    byte_counter = 0
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            for k in range(0, len(array[i][j])):
                array[i][j][k] = data[byte_counter]
                byte_counter += 1


def tests_DVB(bits, array):
    # Required for converting bits to bytes and saving
    scrambled_dvb_array = array.copy()
    descrambled_dvb_array = array.copy()

    scrambled_dvb_bits = scram_DVB(bits.copy())  # Scrambling
    sum_of_bits(scrambled_dvb_bits, counter_dvb)  # Counting the number of bits
    bits_to_bytes(scrambled_dvb_bits, scrambled_dvb_array)  # Convert bits to bytes
    if not(os.path.exists('Output/DVB')): # Check whether the specified path exists or not
        os.makedirs('Output/DVB')
    cv2.imwrite('Output/DVB/DVB_scrambled.jpg', scrambled_dvb_array)  # Saving the scrambled image

    # A copy of the scrambed image needed to restore the original version after each loop
    scrambled_dvb_bits_copy = scrambled_dvb_bits.copy()

    for i in range(1, 101):  # Loop in range 1-100
        if i == switch_intensity:  # Save image with sample (No. 50 – middle)
            switch_bits(scrambled_dvb_bits, i, counter_dvb_longest_sequence, True)  # Bits switching
            descrambled_dvb_bits = scram_DVB(scrambled_dvb_bits)  # Descrambling
            bits_to_bytes(descrambled_dvb_bits, descrambled_dvb_array)  # Convert bits to bytes
            cv2.imwrite('Output/DVB/DVB_descrambled.jpg', descrambled_dvb_array)  # Saving the descrambled image
        else:
            switch_bits(scrambled_dvb_bits, i, counter_dvb_longest_sequence, False)  # Bits switching
            descrambled_dvb_bits = scram_DVB(scrambled_dvb_bits)  # Descrambling

        count_switched_bits(descrambled_dvb_bits, counter_dvb_diffrent_bits)
        scrambled_dvb_bits = scrambled_dvb_bits_copy.copy()  # Restore a scrambled image with modified bits to the original image


def tests_V34(bits, array):
    # Required for converting bits to bytes and saving
    scrambled_v34_array = array.copy()
    descrambled_v34_array = array.copy()

    scrambled_v34_bits = scram_V34(bits.copy())  # Scrambling
    sum_of_bits(scrambled_v34_bits, counter_v34)  # Counting the number of bits
    bits_to_bytes(scrambled_v34_bits, scrambled_v34_array)  # Convert bits to bytes
    if not(os.path.exists('Output/V34')): # Check whether the specified path exists or not
        os.makedirs('Output/V34')
    cv2.imwrite('Output/V34/V34_scrambled.jpg', scrambled_v34_array)  # Saving the scrambled image

    # A copy of the scrambed image needed to restore the original version after each loop
    scrambled_v34_bits_copy = scrambled_v34_bits.copy()

    for i in range(1, 101):  # Loop in range 1-100
        if i == switch_intensity:  # Save image with sample (No. 50 – middle)
            switch_bits(scrambled_v34_bits, i, counter_v34_longest_sequence, True)  # Bits switching
            descrambled_v34_bits = descram_V34(scrambled_v34_bits)  # Descrambling
            bits_to_bytes(descrambled_v34_bits, descrambled_v34_array)  # Convert bits to bytes
            cv2.imwrite('Output/V34/V34_descrambled.jpg', descrambled_v34_array)  # Saving the descrambled image
        else:
            switch_bits(scrambled_v34_bits, i, counter_v34_longest_sequence, False)  # Bits switching
            descrambled_v34_bits = descram_V34(scrambled_v34_bits)  # Descrambling
        count_switched_bits(descrambled_v34_bits, counter_v34_diffrent_bits)
        scrambled_v34_bits = scrambled_v34_bits_copy.copy()  # Restore a scrambled image with modified bits to the original image


def tests_X16(bits, array):
    # Required for converting bits to bytes and saving
    scrambled_x16_array = array.copy()
    descrambled_x16_array = array.copy()

    scrambled_x16_bits = scram_X16(bits.copy())  # Scrambling
    sum_of_bits(scrambled_x16_bits, counter_x16)  # Counting the number of bits
    bits_to_bytes(scrambled_x16_bits, scrambled_x16_array)  # Convert bits to bytes
    if not(os.path.exists('Output/X16')): # Check whether the specified path exists or not
        os.makedirs('Output/X16')
    cv2.imwrite('Output/X16/X16_scrambled.jpg', scrambled_x16_array)  # Saving the scrambled image
    
    # A copy of the scrambed image needed to restore the original version after each loop
    scrambled_x16_bits_copy = scrambled_x16_bits.copy()

    for i in range(1, 101):  # Loop in range 1-100
        if i == switch_intensity:  # Save image with sample (No. 50 – middle)
            switch_bits(scrambled_x16_bits, i, counter_x16_longest_sequence, True)  # Bits switching
            descrambled_x16_bits = scram_X16(scrambled_x16_bits)  # Descrambling
            bits_to_bytes(descrambled_x16_bits, descrambled_x16_array)  # Convert bits to bytes
            cv2.imwrite('Output/X16/X16_descrambled.jpg', descrambled_x16_array)  # Saving the descrambled image
        else:
            switch_bits(scrambled_x16_bits, i, counter_x16_longest_sequence, False)  # Bits switching
            descrambled_x16_bits = scram_X16(scrambled_x16_bits)  # Descrambling
        count_switched_bits(descrambled_x16_bits, counter_x16_diffrent_bits)
        scrambled_x16_bits = scrambled_x16_bits_copy.copy()  # Restore a scrambled image with modified bits to the original image


def tests_start(bits, array):
    # Counting bits
    sum_of_bits(bits, data_counter)

    for i in range(1, 101):  # Loop in range 1-100
        # Copy the image and replace the bits in the copy
        image_switched_bits = bits.copy()
        if i == switch_intensity:
            switch_bits(image_switched_bits, i, counter_data_longest_sequence, True)
            image_switched_array = array.copy()
            bits_to_bytes(image_switched_bits, image_switched_array)
            cv2.imwrite('Output/Statistics/switched_bits.jpg', image_switched_array)
        else:
            switch_bits(image_switched_bits, i, counter_data_longest_sequence, False)
        count_switched_bits(image_switched_bits, counter_data_diffrent_bits)


def count_switched_bits(bits, counter):
    count = 0  # Counter of the number of changed bits
    for i in range(0, len(image_data_bits)):  # Loop over the entire data length
        if image_data_bits[i] is not bits[i]:  # If the bits are not equal, add 1 to the counter
            count += 1
    counter.append(count)  # Adding a counter to the list


def write_stats_to_xlsx(): # Save statiscics to .xlsx file
    workbook = xlsxwriter.Workbook('Output/Statistics/stats.xlsx')
    worksheet = workbook.add_worksheet()
    table_style = 'Table Style Light 11'

    worksheet.add_table('B3:F103', {'header_row': True, 'style': table_style,
                                    'autofilter': False, 'first_column': True,
                                    'columns': [{'header': 'Switch Intensity'},
                                                {'header': 'START'},
                                                {'header': 'DVB'},
                                                {'header': 'V34'},
                                                {'header': 'X16'}]})

    worksheet.write(1, 1, 'Amount of bits switched')

    for i in range(0, 100):
        worksheet.write(i + 3, 1, i + 1)
        worksheet.write(i + 3, 2, counter_data_diffrent_bits[i])
        worksheet.write(i + 3, 3, counter_dvb_diffrent_bits[i])
        worksheet.write(i + 3, 4, counter_v34_diffrent_bits[i])
        worksheet.write(i + 3, 5, counter_x16_diffrent_bits[i])

    worksheet.add_table('H3:L5', {'header_row': True, 'style': table_style,
                                    'autofilter': False, 'first_column': True,
                                    'columns': [{'header': 'Longest bit sequence'},
                                                {'header': 'Start'},
                                                {'header': 'DVB'},
                                                {'header': 'V34'},
                                                {'header': 'X16'}]})

    worksheet.write(3, 7, '0')
    worksheet.write(4, 7, '1')

    worksheet.write(3, 8, counter_data_longest_sequence[0])
    worksheet.write(4, 8, counter_data_longest_sequence[1])

    worksheet.write(3, 9, counter_dvb_longest_sequence[0])
    worksheet.write(4, 9, counter_dvb_longest_sequence[1])

    worksheet.write(3, 10, counter_v34_longest_sequence[0])
    worksheet.write(4, 10, counter_v34_longest_sequence[1])

    worksheet.write(3, 11, counter_x16_longest_sequence[0])
    worksheet.write(4, 11, counter_x16_longest_sequence[1])

    worksheet.add_table('H7:L9', {'header_row': True, 'style': table_style,
                                  'autofilter': False, 'first_column': True,
                                  'columns': [{'header': 'Amount of bits'},
                                              {'header': 'Start'},
                                              {'header': 'DVB'},
                                              {'header': 'V34'},
                                              {'header': 'X16'}]})

    worksheet.write(7, 7, '0')
    worksheet.write(8, 7, '1')

    worksheet.write(7, 8, data_counter[0])
    worksheet.write(8, 8, data_counter[1])

    worksheet.write(7, 9, counter_dvb[0])
    worksheet.write(8, 9, counter_dvb[1])

    worksheet.write(7, 10, counter_v34[0])
    worksheet.write(8, 10, counter_v34[1])

    worksheet.write(7, 11, counter_x16[0])
    worksheet.write(8, 11, counter_x16[1])

    workbook.close()


def print_stats():
    print(f"================== START IMAGE ==================")
    print((f"| Amount of Bits: [0:{data_counter[0]}], [1:{data_counter[1]}]").ljust(48)+"|")
    print((f"| Longest sequence of bits: [0:{counter_data_longest_sequence[0]}], [1:{counter_data_longest_sequence[1]}]").ljust(48)+"|")
    print((f"| Amount of bits switched: {counter_data_diffrent_bits[switch_intensity - 1]}").ljust(48)+"|")

    print(f"====================== DVB ======================")
    print((f"| Amount of Bits: [0:{counter_dvb[0]}], [1:{counter_dvb[1]}]").ljust(48)+"|")
    print((f"| Longest sequence of bits: [0:{counter_dvb_longest_sequence[0]}], [1:{counter_dvb_longest_sequence[1]}]").ljust(48)+"|")
    print((f"| Amount of bits switched: {counter_dvb_diffrent_bits[switch_intensity - 1]}").ljust(48)+"|")

    print(f"====================== V34 ======================")
    print((f"| Amount of Bits: [0:{counter_v34[0]}], [1:{counter_v34[1]}]").ljust(48)+"|")
    print((f"| Longest sequence of bits: [0:{counter_v34_longest_sequence[0]}], [1:{counter_v34_longest_sequence[1]}]").ljust(48)+"|")
    print((f"| Amount of bits switched: {counter_v34_diffrent_bits[switch_intensity - 1]}").ljust(48)+"|")

    print(f"====================== X16 ======================")
    print((f"| Amount of Bits: [0:{counter_x16[0]}], [1:{counter_x16[1]}]").ljust(48)+"|")
    print((f"| Longest sequence of bits: [0:{counter_x16_longest_sequence[0]}], [1:{counter_x16_longest_sequence[1]}]").ljust(48)+"|")
    print((f"| Amount of bits switched: {counter_x16_diffrent_bits[switch_intensity - 1]}]").ljust(48)+"|")
    print(f"=================================================")
    print(f"\nThe number of exchanged bits displayed at a conversion intensity of {switch_intensity}!")


def del_output():
    os.remove("Output/Statistics/start_image.jpg")
    os.remove("Output/Statistics/switched_bits.jpg")
    os.remove("Output/Statistics/stats.xlsx")
    os.remove("Output/AES/AES_descryption.jpg")
    os.remove("Output/AES/AES_encryption.jpg")
    os.remove("Output/DVB/DVB_scrambled.jpg")
    os.remove("Output/DVB/DVB_descrambled.jpg")
    os.remove("Output/V34/V34_scrambled.jpg")
    os.remove("Output/V34/V34_descrambled.jpg")
    os.remove("Output/X16/X16_scrambled.jpg")
    os.remove("Output/X16/X16_descrambled.jpg")
    os.rmdir('Output/Statistics')
    os.rmdir('Output/AES')
    os.rmdir('Output/DVB')
    os.rmdir('Output/V34')
    os.rmdir('Output/X16')
    os.rmdir('Output')


if (__name__) == '__main__':
    # Loading the file to be sent
    image_name = input("Type file name: ")
    image = cv2.imread(image_name)
    image_array = np.array(image)
    if not(os.path.exists('Output/Statistics')): # Check whether the specified path exists or not
        os.makedirs('Output/Statistics')
    cv2.imwrite('Output/Statistics/start_image.jpg', image_array)
    image_data = []
    for x in range(0, len(image_array)):
        for y in range(0, len(image_array[x])):
            for z in range(0, len(image_array[x][y])):
                image_data.append(image_array[x][y][z])
    # Convert image to bits
    image_data_bits = []  # Bits of the starting image
    image_to_bits(image_data, image_data_bits)

    bytes_as_String = ''.join(str(x) for x in image_data_bits)

    switch_intensity = int(input("\nType switching intensity [1-100] (only for in-console stats and file save): \n"))
    if switch_intensity < 1 or switch_intensity > 100:
        print(f"Value is not between 1-100! Changing it to 50.\n")
        switch_intensity = 50

    tests_start(image_data_bits.copy(), image_array.copy())

    tests_DVB(image_data_bits.copy(), image_array.copy())

    tests_V34(image_data_bits.copy(), image_array.copy())

    tests_X16(image_data_bits.copy(), image_array.copy())

    encryption(bytes_as_String.encode('UTF-8'), image_array.copy())

    # Printing statistics
    print_stats()

    # Saving statistics to .xlsx file
    write_stats_to_xlsx()

    # Deleting the output files
    if input("\nDo you want to delete output files? [y/n]: ") == 'y':
        del_output()
