import xml.etree.ElementTree as ET
tree = ET.parse('map-grid.xml')
root = tree.getroot()
print "---Root----Root.attribute---"
print root.tag,"  ",root.attrib

dic={}

for child in root:
	dic[child.tag]=[]
	for gchild in child:
		x=gchild.tag
		dic[child.tag].append(gchild.attrib)
		# print gchild.tag," ",gchild.attrib

for key in dic:
	print "key = ", key
	for value in dic[key]:
		print value
	print "\n\n"

