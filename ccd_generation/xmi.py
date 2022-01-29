"""
XMI resources
Specify unique sources for meta-data or attributes
creation meta-data at root node: identify uri for xml namespace, model name (eg. org.ocopomo),
:author chuongvo
"""
from xml.etree import ElementTree as ET
# from xml.dom import minidom, Node
from io import BytesIO
from GenerationID import randomString22


# class _MetaResource:
#     """
#     create meta-data resource at the root node
#     provides customized root class namespace and uri
#     """
#     def __init__(self, options=None):
#         # get submitted dict namespace and uri as options
#         self._options = options or {}
#         self._ccd_ns = ''
#         self._ccd_uri = ''
#         self._anno_ns = ''
#         self._anno_uri = ''
#
#
#     def set_ccd_ns(self, ccd_ns):
#         self._ccd_ns = 'xmlns:' + ccd_ns
#
#     def set_ccd_uri(self, ccd_uri):
#         self._ccd_uri = ccd_uri
#
#     def set_anno_ns(self, anno_ns):
#         self._anno_ns = 'xmlns:' + anno_ns
#
#     def set_anno_uri(self, anno_uri):
#         self._anno_uri = anno_uri
#
#     # def get_anno_ns(self):
#     #     """ provides annotation namespace for subElement """
#     #     return self._anno_ns
#     #
#     # def get_ccd_ns(self):
#     #     return self._ccd_ns
#
#     def mapns(self):
#         """Map ns and uri as a dict"""
#         if self._ccd_ns == '' or self._ccd_uri == '' or self._anno_ns == '' or self._anno_uri == '':
#             return "Please provide namespace and its uri."
#         else:
#             self._options[self._xmi_ns] = self._xmi_uri
#             self._options[self._ccd_ns] = self._ccd_uri
#             self._options[self._anno_ns] = self._anno_uri
#             return self._options


class XMIResource:
    """
    provides xmi_ID key attribute, xmi_annotation type attribute, update_root_meta
    """

    def __init__(self, xmi_ver=None, ccd_ns='org.ocopomo.ccd', ccd_uri='http://www.ocopomo.org/ccd/1.0', \
                 anno_ns='org.ocopomo.annotation', anno_uri='http://www.ocopomo.org/annotation/1.0'):
        self._ccd_uri = ccd_uri
        self._anno_uri = anno_uri
        self._xmi_version = xmi_ver
        self.anno_ns = anno_ns
        self.ccd_ns = ccd_ns
        self._xmi_ns = 'xmlns:xmi'
        self._xmi_uri = 'http://www.omg.org/XMI'

    def _mapns(self):
        """ Map meta-data into dict"""
        options = {}
        if self.ccd_ns == '' or self.anno_ns == '' or self._ccd_uri == '' or self._anno_uri == '':
            return "Namespace schema is not empty."
        else:
            options[self._xmi_ns] = self._xmi_uri
            options['xmlns:' + self.ccd_ns] = self._ccd_uri
            options['xmlns:' + self.anno_ns] = self._anno_uri
            return options

    def xmi_id_attr(self):
        """ :return array of key, value pair """
        id = randomString22()
        return ['xmi:id', id]

    def xmi_type_anno_attr(self, type):
        """
        :type: indicates the type of annotation document (eg. TxtAnnotation/PdfAnnotation).
        :return  array of key, value pair
        """
        prefix = 'xmi:type'
        val = self.anno_ns + ':' + type
        return [prefix, val]

    def update_root_meta(self, node_root, root_name):
        """
        Update meta-data for node root
        :raise ValueError if meta-data is empty
        :param node_root: the root element of document
        :param root_name: the name of root element of xml document
        :return: fulfill details of root node.
        """
        # check a node is a xml node
        if isinstance(node_root, ET.Element):
            # check if is a root node
            root = ET.ElementTree(node_root).getroot()
            if str(root.tag) != root_name:
                raise ValueError("Not a root element.")
            else:
                if self._xmi_version is None:
                    node_root.set("xmi:version", '2.0')
                else:
                    node_root.set("xmi:version", self._xmi_version)
                # set up root meta-data info
                if isinstance(self._mapns(), str):
                    raise ValueError(self._mapns())
                else:
                    for k, v in self._mapns().items():
                        node_root.set(str(k), str(v))
                    # set up root serialized id
                    id = self.xmi_id_attr()
                    node_root.set(id[0], id[1])
        else:
            raise ValueError("Node is not a XML element.")

    def prettify(self, node):
        """ Print Pretty format XML  with encoding utf-8"""
        # rough_string = ET.ElementTree.tostring(node, 'utf-8')
        # reparsed = minidom.parseString(rough_string)
        # return reparsed.toprettyxml(indent=' ')

        et = ET.ElementTree(node)
        f = BytesIO()
        et.write(f, encoding='utf-8', xml_declaration=True)
        return str(f.getvalue(), 'utf-8')


