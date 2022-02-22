from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import algorithms, Cipher, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
_CHUNK_SIZE = 1024
__PBKDF2_ITER = 100000
__NONCE_SIZE = 16
__KEY_LENGTH = 32
__BACKEND = default_backend()
import struct


def encode(input_stream, output_stream, encode_key):
    """Encode a stream using a byte object key as the encoding 'password'.

    Derives the 256 bit key using PBKDF2 password stretching and encodes using
    AES in CTR mode 1024 bytes at a time. Includes the raw initialization value
    -- 'nonce' -- as the first item in the encoded ciphertext.

    NOTE: this function does not close the input or output stream.
    The stream is duck-typed, as long as the object can be read N bits at a
    time via read(N) and written to using write(BYTES_OBJECT).

    Args:
        input_stream (open readable binary io stream): Input stream, typically
            an open file. Data read from this stream is encoded and written to
            the output_stream.
        output_stream (open writable binary io stream): Writable output stream
            where the encoded data of the input_file is written to.
        encode_key (bytes): byte text representing the encoding password.
            b"yourkey"

    Returns:
        Nothing.

    Raises:
        TypeError: if the encode_key is not a byte object

    """
    if not isinstance(encode_key, bytes):
        try:
            encode_key = str.encode(encode_key)
        except Exception:
            raise TypeError('encode_key must be passed as a byte object')

    kdf = PBKDF2HMAC(algorithm=(hashes.SHA256()), length=__KEY_LENGTH, salt=b'\xcb<\xdc.\x85\x86\x89\xda\x90\x85c\x05j\x00\xdb\xed\t\xac\x07lxUC\xc4a\x0f\x06\xd3\x1bS\xa72',
      iterations=__PBKDF2_ITER,
      backend=__BACKEND)
    key = kdf.derive(encode_key)
    nonce = os.urandom(__NONCE_SIZE)
    output_stream.write(nonce)
    cipher = Cipher((algorithms.AES(key)), (modes.CTR(nonce)), backend=__BACKEND)
    encoder = cipher.encryptor()
    while 1:
        chunk = input_stream.read(_CHUNK_SIZE)
        output_stream.write(encoder.update(chunk))
        if len(chunk) < _CHUNK_SIZE:
            output_stream.write(encoder.finalize())
            break


def decode(input_stream, output_stream, encode_key):
    """Decode a stream using byte object key as the decoding 'password'.

    Derives the 256 bit key using PBKDF2 password stretching and decodes
    a stream encoded using AES in CTR mode by the above encode function.
    Processes 1024 bytes at a time and uses the 'nonce' value included at
    the beginning of the cipher text input stream.

    NOTE: This function does not delete the encoded cipher text.

    Args:
        input_stream (open readable binary io stream): Encoded input stream,
            typically an open file. Data read from this stream is decoded and
            written to the output_stream.
        output_stream (open writable binary io stream): Writable output stream
            where the decoded data of the input_file is written to.
        encode_key (bytes): byte text representing the encoding password.
            b"yourkey".

    Returns:
        Nothing.

    Raises:
        TypeError: if the encode_key is not a byte object
        ValueError: if a valid nonce value can't be read from the given file.

    """
    if not isinstance(encode_key, bytes):
        try:
            encode_key = str.encode(encode_key)
        except Exception:
            raise TypeError('encode_key must be passed as a byte object')

    kdf = PBKDF2HMAC(algorithm=(hashes.SHA256()), length=__KEY_LENGTH, salt=b'\xcb<\xdc.\x85\x86\x89\xda\x90\x85c\x05j\x00\xdb\xed\t\xac\x07lxUC\xc4a\x0f\x06\xd3\x1bS\xa72',
      iterations=__PBKDF2_ITER,
      backend=__BACKEND)
    key = kdf.derive(encode_key)
    cipher = Cipher((algorithms.AES(key)), (modes.CTR(input_stream.read(__NONCE_SIZE))),
      backend=__BACKEND)
    decoder = cipher.decryptor()
    while 1:
        chunk = input_stream.read(_CHUNK_SIZE)
        output_stream.write(decoder.update(chunk))
        if len(chunk) < _CHUNK_SIZE:
            output_stream.write(decoder.finalize())
            break


key = "your_api_key when use for train with tao"
input_etlt_model_onnx = "the exported model may be onnx_encoded_etlt"
with open("./out.onnx", 'wb') as onnxf:
    with open(input_etlt_model_onnx, 'rb') as f:
        input_blob_len= f.read(4)
        c = struct.unpack('<i', input_blob_len)
        print(c)
        blob_name = f.read(c[0])
        decode(f, onnxf, key.encode())

input_etlt_model_uff = "the exported model may be uff_encoded_etl"
with open("./out.uff", 'wb') as ufff:
    with open(input_etlt_model_uff, 'rb') as f:
        input_blob_len= f.read(4)
        c = struct.unpack('<i', input_blob_len)
        print(c)
        blob_name = f.read(c[0])
        decode(f, ufff, key.encode())
