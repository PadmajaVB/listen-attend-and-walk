import xml.etree.ElementTree as ET
import csv

tree = ET.parse('Paragraph.xml')
root = tree.getroot()
# print "---Root----Root.attribute---"
# print root.tag,"  ",root.attrib
txtfile = "Grid.txt"

Grid = []
Jelly = []
L = []

for child in root:
	mapName = child.attrib['map']
	# print "mapName = ", type(mapName)
	for gchild in child:
		if gchild.tag == "instruction":
			text = gchild.text
			if mapName == "Grid":
				Grid.append(text)
			elif mapName == "Jelly":
				Jelly.append(text)
			elif mapName == "L":
				L.append(text)
			else:
				print "Nothing ----------"

# for x in range(0, len(Grid)):
# 	Grid[x] = Grid[x].replace('\n', '')

for x in range(0, len(Jelly)):
	Jelly[x] = Jelly[x].strip('\n')
	Jelly[x] = Jelly[x].strip('\t')
	Jelly[x] = Jelly[x].replace('\n', '')

for x in range(0, len(L)):
	L[x] = L[x].strip('\n')
	L[x] = L[x].strip('\t')
	L[x] = L[x].replace('\n', '')

# with open("Grid.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in Grid:
#         writer.writerow([val]) 

# with open("Jelly.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in Jelly:
#         writer.writerow([val]) 

# with open("L.txt", "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in L:
#         writer.writerow([val]) 


print "Grid = ", Grid
# print "\n\nJelly = ", Jelly
# print "\n\nL = s", L
