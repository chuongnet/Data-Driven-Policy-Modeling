"""
    JSON extraction functions, reading nested json and parsing into list of data stream
    Exception: list of nested node [{entity:attribute}, {entity:attribute}]
    Supports customized json format
    :author chuongvo
"""
import json


class ExtractJson:

    def __init__(self, filepath):
        """initial instances of class"""
        self.path = filepath
        self.message = ''

    #@staticmethod
    def messageError(self, message):
        self.message = message

    #@staticmethod
    def checkDict(self, nestedNode, element, output):
        """
        check nested node if any,
        :param nestedNode: input nested json node
        :param element: entity element
        :param output: list of keyword strings, returns result in list
        """
        for ele in nestedNode:
            if isinstance(nestedNode[ele], dict):
                self.checkDict(nestedNode[ele], element + '.' + ele, output)
            elif isinstance(nestedNode[ele], list):
                self.checkList(nestedNode[ele], element + '.' + ele, output)
            elif isinstance(str(nestedNode[ele]), str): # convert to string if value is numeric type
                output.append(element + '.' + ele + ':' + str(nestedNode[ele])) # root.entity:value

    #@staticmethod
    def checkList(self, listedNode, element, output):
        """
        check for attribute has the list of values
        :exception return error when had list of nested node
        :param listedNode:
        :param element:
        :param output:
        """
        for i in range(len(listedNode)):
            if isinstance(listedNode[i], dict):
                self.messageError("Wrong json format.")
                break
            elif isinstance(listedNode[i], list):
                self.checkList(listedNode[i], element, output)
            else:
                output.append(element + ':' + str(listedNode[i]))

    def extractJson(self):
        """
        Main function for parsing JSON
        :return: the list of string in entity.childNode.attribute:value form
        """
        output = []
        try:
            with open(self.path) as f:
                data = json.load(f)
                for entity in data:
                    sub = []
                    if isinstance(data[entity], dict):
                        self.checkDict(data[entity], entity, sub)
                    elif isinstance(data[entity], list):
                        self.checkList(data[entity], entity, sub)
                    elif isinstance(str(data[entity]), str):
                        sub.append(entity + ':' + str(data[entity]))
                    output.append(sub)
            #self.message = "parsing json successful."
        except:
            self.messageError("Wrong JSON format.")
        return output
