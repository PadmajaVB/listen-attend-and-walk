import xml.etree.ElementTree as ET
import csv

tree = ET.parse('SingleSentence.xml')
root = tree.getroot()
# print "---Root----Root.attribute---"
# print root.tag,"  ",root.attrib

Grid = []
Jelly = []
L = []

count = 0

for child in root:
	temp = {}
	mapName = child.attrib['map']
	temp['id'] = child.attrib['id']
	temp['map'] = mapName
	# print "mapName = ", type(mapName)
	for gchild in child:
		if gchild.tag == "instruction":
			temp['filename'] = gchild.attrib['filename']
			modText = gchild.text
			modText = modText.strip('\n')
			modText = modText.strip('\t')
			modText = modText.replace('\n','')
			modText = modText.lstrip()
			modText = modText.rstrip()
			temp['instruction'] = modText
		elif gchild.tag == "path":
			temp['path'] = gchild.text
	# print "temp = ", temp
	if mapName == "Grid":
		Grid.append(temp)
	elif mapName == "Jelly":
		Jelly.append(temp)
	elif mapName == "L":
		L.append(temp)
	else:
		print "Nothing ----------"


# with open("GridSingleSent.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in Grid:
#         writer.writerow([val['instruction']]) 

# with open("JellySingleSent.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in Jelly:
#         writer.writerow([val['instruction']]) 

# with open("LSingleSent.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in L:
#         writer.writerow([val['instruction']]) 

for x in L:
	print x['instruction']
	# print type(x['instruction'])
# 	print
# print
# print "------------------------------------------------"

# for x in Jelly:
# 	print x
# 	print
# print
# print "------------------------------------------------"

# for x in L:
# 	print x
# 	print
# print
# print "------------------------------------------------"
