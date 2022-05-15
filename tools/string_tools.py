import os
import string

from unicodedata import normalize

#taken from https://stackoverflow.com/questions/13939120/sanitizing-a-file-path-in-python

def os_path_separators():
    seps = []
    for sep in os.path.sep, os.path.altsep:
        if sep:
            seps.append(sep)
    return seps


def sanitise_filesystem_name(potential_file_path_name):
    # Sort out unicode characters
    valid_filename = normalize('NFKD', potential_file_path_name).encode('ascii', 'ignore').decode('ascii')
    # Replace path separators with underscores
    for sep in os_path_separators():
        valid_filename = valid_filename.replace(sep, '_')
    # Ensure only valid characters
    valid_chars = "-_.() {0}{1}".format(string.ascii_letters, string.digits)
    valid_filename = "".join(ch for ch in valid_filename if ch in valid_chars)
    # Ensure at least one letter or number to ignore names such as '..'
    valid_chars = "{0}{1}".format(string.ascii_letters, string.digits)
    test_filename = "".join(ch for ch in potential_file_path_name if ch in valid_chars)
    if len(test_filename) == 0:
        # Replace empty file name or file path part with the following
        valid_filename = "(Empty Name)"
    return valid_filename