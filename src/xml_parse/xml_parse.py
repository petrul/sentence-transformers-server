# import xml.etree.ElementTree as ET
from lxml import etree

# xml_string = """
# <root>
#   <child1>value1</child1>
#   <child2>value2</child2>
# </root>
# """

filename = '/home/petru/work/scriptorium-masters/build/en/gibbon-decline_and_fall_1.xml'

tree = etree.parse(filename)
# root = tree.getroot()

namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

# tree = ET.parse()
xpath = etree.XPath("//tei:p", namespaces = namespaces)

# for child in root:
#   print(child.tag, child.text)

# paragraphs = xpath(root)
p_elements = tree.xpath("//tei:p", namespaces=namespaces)
paragraphs = (para.text for para in p_elements)

if __name__ == "__main__":

  def p(args):
    print(args)

  for i in range(len(paragraphs)):
    para = paragraphs[i]
    p(f"{i} =>")
    print(para.tag)
    print(para.text)
    print("=")

  p("asd")
  p(len(paragraphs))

  # print(root.xpath("//p"))