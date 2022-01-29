"""
Generate random uuid, encode into base64 Hex.
Generate uuid using MD5 Hash and a keyword string, cannot be decoded
Provide random 22 characters of string
:author chuongvo
"""
import uuid
import secrets
import string
from Based64Convertion import encodedBased64


def generatedID():
    """
    Random UUID and encoding
    :return: the string in Hex
    """
    bid = encodedBased64(str(uuid.uuid4()))
    return bid


def generatedIDwithKey(stringkey):
    """
    Generation an UUID using MD5 hash and the pwd key, encoding
    :param stringkey: using the string key (pwd) to generate an uuid
    :return: an UUID string in base64 Hex
    """
    gid = uuid.uuid3(uuid.NAMESPACE_DNS, str(stringkey))
    ide = encodedBased64(str(gid))
    return ide


def randomString22():
    """
    Random a string of characters with length of 22, no encoding
    :return: a string with mixed characters
    """
    result = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(22))
    #encoded = encodedBased64(result)
    return '_' + result
