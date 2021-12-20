ROUNDS = 10
import numpy as np
from Sbox import sbox
import sys
import time


def tobits(string):
    """
        Convert a string to bits in big-endian format
    """
    result = []
    for c in string:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def tabbits_to_string(bits_list):
    """
        tablica bitów na stringi #Convert array/list of bits into the string
    """
    string = ""
    for i in range(0, len(bits_list)):

        string += str(bits_list[i])
    return string


def bitstring_to_bytes(s):
    """
        Convert bits string into the bytes
    """
    v = int(s, 2)
    b = bytearray()
    while v:
        b.append(v & 0xff)
        v >>= 8
    return bytes(b[::-1])


def M_BLOCKS(plaintext):
    """
        Create blocks(not padded) to padd them latter
    """
    buffor = plaintext
    blocks = []
    while True:
        if len(buffor) > 64:
            get_64chars = buffor[0:64]
            blocks.append(get_64chars)
            buffor = buffor[len(get_64chars):]
        else:
            blocks.append(buffor)
            break
    return blocks


def pad(string):
    """
        Padding function (from the documentation)
    """
    full = 512
    for i in range(len(string)):
        if len(tobits(string[i])) < 512:
            str_bits = tabbits_to_string(tobits(string[i]))
            N = len(str_bits)
            w = (-N - 65) % full
            r_64b = str('{:064b}'.format(int((N + w + 65) / full)))
            str_bits += str(1) + (str(0) * w) + r_64b
            padded_string = bitstring_to_bytes(bin(int(str_bits, 2))[2:])
            string[i] = padded_string
    return string


def everynth(list, k, n):
    """
        Take every 8th element from the k element
    """
    return list[k::n]


def init_state(text):
    """
        Initialization of the  64 bytes state, according to the documentation.
    """
    if type(text[0]) == type(str()):  #If are they strings coonvert them to the bytes
        text[0] = str.encode(text[0])
    else:
        pass
    state = [[b'0'] * 8 for _ in range(8)]
    for i in range(0, 8):
        buffor = everynth(text[0], i, 8)
        for j in range(0, 8):
            state[i][j] = buffor[j]
    return state


def init_all_states(padded_blocks):
    """
        Initialize all blocks to throw them one by one into the compression function
    """
    inicializated_messages = []

    for i in range(0, len(padded_blocks)):
        inicializated_messages.append(np.array(init_state([padded_blocks[i]])))
    return inicializated_messages


# ------------------------------------------------------------
# Add Round constant P
# ------------------------------------------------------------

def add_round_constant_P(state, i):
    zeros = np.array([0x00] * 64)

    for column in range(8):
        zeros[column * 8] = (column * 16) ^ i

    zeros = np.reshape(zeros, (8, 8)).T
    return np.bitwise_xor(zeros, state)


# ------------------------------------------------------------
# Add Round constant Q
# ------------------------------------------------------------

def add_round_constant_Q(state, i):
    ff = np.array([0xff] * 64)
    rows = 8
    cols = 8

    for column in range(cols):
        ff[column * 8 + 8 - 1] = (0xff - column * 16) ^ i

    ff = np.reshape(ff, (8, 8)).T
    return np.bitwise_xor(ff, state)


# ------------------------------------------------------------
# Subbytes
# ------------------------------------------------------------
def SubBytes(state, sbox):
    state = np.reshape(state, (1, 64))
    state = np.reshape(state, (8, 8))
    transform = [[sbox[byte] for byte in word] for word in state]
    state = np.reshape(transform, (8, 8))
    return state


# ------------------------------------------------------------
# ShiftBytes
# ------------------------------------------------------------
def getrow(state, row):
    return state[row]


def ShiftBytes(state, distinct):
    for i in range(0, 8):
        getrow(state, i)
        x = np.roll(getrow(state, i), -distinct[i])
        state[i] = x
    return state


# ------------------------------------------------------------
# Reduction over GF(2^8)
# ------------------------------------------------------------
def mul1(b):
    return np.uint8(b)


def mul2(b):
    if b >> 7:
        return np.uint8((b << 1) ^ 0x1b)
    else:
        return np.uint8(b << 1)


def mul3(b):
    return np.uint8(mul2(b) ^ mul1(b))


def mul4(b):
    return np.uint8(mul2(mul2(b)))


def mul5(b):
    return np.uint8(mul4(b) ^ mul1(b))


def mul6(b):
    return np.uint8(mul4(b) ^ mul2(b))


def mul7(b):
    return np.uint8(mul4(b) ^ mul2(b) ^ mul1(b))


# ------------------------------------------------------------
# MixBytes
# ------------------------------------------------------------
def mix_bytes(state):
    cols = 8
    rows = 8
    temp = [0x00] * rows


    for i in range(0, cols):
        for j in range(0, rows):
            temp[j] = (np.uint8(mul2(state[(j + 0) % rows][i])) ^
                       np.uint8(mul2(state[(j + 1) % rows][i])) ^
                       np.uint8(mul3(state[(j + 2) % rows][i])) ^
                       np.uint8(mul4(state[(j + 3) % rows][i])) ^
                       np.uint8(mul5(state[(j + 4) % rows][i])) ^
                       np.uint8(mul3(state[(j + 5) % rows][i])) ^
                       np.uint8(mul5(state[(j + 6) % rows][i])) ^
                       np.uint8(mul7(state[(j + 7) % rows][i]))
                       ) & 0xFF

        for j in range(0, rows):
            state[j][i] = temp[j]
    return state


# ------------------------------------------------------------
# Permutation P
# ------------------------------------------------------------

def permutationP(state):
    for i in range(ROUNDS):
        state = add_round_constant_P(state, i)
        state = SubBytes(state, sbox)
        state = ShiftBytes(state, (0, 1, 2, 3, 4, 5, 6, 7))
        state = mix_bytes(state)

    return state


# ------------------------------------------------------------
# Permutation Q
# ------------------------------------------------------------

def permutationQ(state):
    for i in range(ROUNDS):
        state = add_round_constant_Q(state, i)
        state = SubBytes(state, sbox)
        state = ShiftBytes(state, (1, 3, 5, 7, 0, 2, 4, 6))
        state = mix_bytes(state)
    return state


# ------------------------------------------------------------
# IV STATE GENERATOR
# ------------------------------------------------------------
def iv():  # generacja stanu IV
    """
        Generate IV - state
    """
    size = 256
    state = [0] * 64
    p = len(state) - 1
    while size > 0:
        state[p] = size % 256
        size >>= 8
        p -= 1
        assert p != 0
    state = init_state([state])
    return np.array(state)


# ------------------------------------------------------------
# Compresion
# ------------------------------------------------------------

def compresion(h, m):
    pside = permutationP(np.bitwise_xor(h, m))
    qside = permutationQ(m)
    pqxor = np.bitwise_xor(pside, qside)
    return np.bitwise_xor(pqxor, h)


# ------------------------------------------------------------
# Trunc
# ------------------------------------------------------------

def trunc(state):
    res = np.bitwise_xor(permutationP(state), state)
    r1 = res[:, 4].copy()
    r2 = res[:, 5].copy()
    r3 = res[:, 6].copy()
    r4 = res[:, 7].copy()
    res = np.concatenate([r1, r2, r3, r4])

    string = ""
    for i in range(0, 32):
        string += format(res[i], '02x')

    return string


# ------------------------------------------------------------
# GROESTL-256
# ------------------------------------------------------------

def groestl(text):
    text = pad(M_BLOCKS(text))
    h = iv()
    states = init_all_states(text)

    for i in range(len(states)):
        h = compresion(h, states[i])

    return trunc(h)


def calc_bit_lenght(score):
    """
        An example hex value 0x1234 is 4 * 4 bits = 16
    """
    wb = score * 4
    return wb



def comapre_test():
    text = ""

    result = groestl(text)
    print("The first test vector:", text, ",(empty) from wikipedia\n", result)
    print("Długość bitowa wynosi = ", len(calc_bit_lenght(result)), "\n")

    text = "The quick brown fox jumps over the lazy dog"
    result = groestl(text)
    print("The second test vector:", text, ", from wikipedia\n", groestl(text))
    print("The bit length is = ", len(calc_bit_lenght(result)), "\n")

    text = "abc"
    result = groestl(text)
    print("The third test vector:", text, ", from the authors\n", groestl(text))
    print("The bit length is = ", len(calc_bit_lenght(result)), "\n")

    text = "Bartek"
    result = groestl(text)
    print("The fourth test vector:", text, ", as my name\n", groestl(text))
    print("The bit length is = ", len(calc_bit_lenght(result)), "\n")



if __name__ == '__main__':
    print("***************")
    print("**GROESTL-256**")
    print("***************")
    print("Filename:", sys.argv[1])


    # File handling
    try:
        filee = open(sys.argv[1], 'rb')
    except FileNotFoundError:
        print(f"File  {sys.argv[1]} not found ... End of program")
        sys.exit(1)
    except OSError:
        print(f" An IOError error occurred while opening the file:{sys.argv[1]}")
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error while opening the file: {sys.argv[1]} ", repr(err))
        sys.exit(1)
    else:
        with filee as file:
            bytecode = file.read()

    print("The file length is = ", round(len(bytecode)), "bytes")
    hash = groestl(bytecode.decode('ISO-8859-1'))
    print("A new file will be created, please enter its name to which the shortcut will be saved:")
    time.sleep(1)
    res = input()
    newfile = open(res, "w")
    newfile.write(hash)
    file.close()
    newfile.close()

