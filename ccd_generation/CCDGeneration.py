"""
Read the completed well-defining of CCD in the JSON structure (final format)
CCD file generation in XML format
:author chuongvo
"""
from xml.etree import ElementTree as ET
from xmi import XMIResource
import json
from collections import defaultdict

class CCDGeneration:

    def __init__(self, filepath):
        self._filepath = filepath

    def write_file(self, filename):
        """
        write the output into the CCD file
        :param input: data stream to write
        :param filename: the name of the output file
        :return: successful or error message.
        """
        message = ''
        if filename is '':
            filename = 'autofile'
        else:
            try:
                file = str(filename).strip(' ') + '.ccd'
                input = self._ccd_generation()
                # print(input)
                with open(file, 'w+') as f:
                    f.write(input)
                    message = "write a file successful."
                    f.close()
            except Exception as e:
                message = e.args[0]
        return message

    def _check_dict(self, nesteddata, root):
        """
        Recursive function checking nested nodes
        :param nesteddata: the nested nodes
        :param entity: the entity name of the nested
        :param root: parent node for a nested node
        """
        for element in nesteddata:
            if isinstance(nesteddata[element], dict):
                # create a sub node
                node = ET.SubElement(root, element)
                self._check_dict(nesteddata[element], node)
            elif isinstance(nesteddata[element], list):
                # nested node in a list of values error
                sub = ''
                for i in range(len(nesteddata[element])):
                    if isinstance(nesteddata[element][i], dict):
                        raise Exception("Wrong format list of element's values. No nested in list")
                        break
                    else:
                        sub += str(nesteddata[element][i]) + ' '
                root.set(element, sub.strip())  # set element as an attribute and its values as a string
            else:
                root.set(element, str(nesteddata[element]))

    def _ccd_generation(self):
        root = None
        root_name = ''
        output = ''
        if self._filepath is None or self._filepath == '':
            raise Exception('No input file to generate.')
        else:
            with open(self._filepath, 'r') as file:
                try:
                    data = json.load(file, object_pairs_hook=self._load_duplication_key)
                    # decoder = json.JSONDecoder(object_pairs_hook=self._load_duplication_key)
                    # data = decoder.decode(file)
                except:
                    raise Exception("Json file is in wrong format.")
                print(data)
                xmires = XMIResource()
                # read level 0, root entity
                for entity in range(len(data) - 1):
                    if entity == 0:
                        for k, v in data[entity].items():
                            if k == 'CCD':
                                root_name = xmires.ccd_ns + ':' + k
                                root = ET.Element(root_name)
                                self._check_dict(v, root)
                            else:
                                raise Exception("Wrong format for CCD file, checking the root node of document.")
                    xmires.update_root_meta(root, root_name)
                    # read data of the root, nested entities (e.g. actors, objects)
                    lvl1data = data[entity + 1]
                    for element in lvl1data:
                        if isinstance(lvl1data[element], dict):
                            # create a sub node
                            node = ET.SubElement(root, element)
                            # read nested nodes from the root node e
                            self._check_dict(lvl1data[element], node)
                        elif isinstance(lvl1data[element], list):
                            # if nested node in the list of values
                            sub = ''
                            for i in range(len(lvl1data[element])):
                                if isinstance(lvl1data[element][i], dict):
                                    raise Exception("Wrong format list of element's values. No nested in list")
                                    break
                                else:
                                    sub += str(lvl1data[element][i]) + ' '
                            root.set(element, sub.strip())
                        else:
                            root.set(element, str(lvl1data[element])) # set the attribute name for the root node

                file.close()
                output = xmires.prettify(root)
        return output

    @staticmethod
    def _load_duplication_key(pairs):
        d = defaultdict
        for k, v in pairs:
            d[k]= v
        return d