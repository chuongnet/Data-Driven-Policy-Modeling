"""
Encode and Decode UUID to base64 Hex string.
:author chuongvo
"""
import base64


def encodedBased64(stringText):
    """
    Encoding text with base64
    :param stringText:
    :return: the Hex base64
    """
    try:
        # convert text into byte-like for base64.b64encode
        b = base64.b64encode(stringText.encode('utf-8'))
        # convert back to string type
        bstr = str(b, 'utf-8')
        # using regular to remove = in text
        bstrip = bstr.strip(r'=+')
        return bstrip
    except:
        return "Encoding error."


def decodedBased64(stringText):
    """
    Decode the Hex base64 of the text
    :param stringText:
    :return: the decoded text
    """
    try:
        # convert to byte-like and decode
        d = base64.b64decode(stringText.encode('utf-8'))
        # convert back to a string
        dstr = str(d, 'utf-8')
        return dstr
    except:
        return "Decode error."


