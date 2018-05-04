import xml.etree.ElementTree as ET
import csv
import cPickle as pickle

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
			modText = gchild.text
			modText = modText.strip('\n')
			modText = modText.strip('\t')
			modText = modText.replace('\n','')
			modText = modText.lstrip()
			modText = modText.rstrip()
			tempList = eval(modText)
			#print "list - ",tempList
			temp['path'] = tempList
			#code for generating actions
			if (tempList[0][2]==-1):
				temp_list=list(tempList[0])
				if(len(tempList)==1):
					temp_list[2]=0
				else:
					temp_list[2]=tempList[1][2]-90
				tempList[0] = tuple(temp_list)
			size=len(tempList)
			actions = []
			for i in range(size-1):
				#print i #0-13
				xdiff=tempList[i+1][0]-tempList[i][0]
				ydiff=tempList[i+1][1]-tempList[i][1]
				zdiff=tempList[i+1][2]-tempList[i][2]
				if((xdiff!=0) or (ydiff!=0)):
					actions.append([1,0,0,0])
				elif(zdiff!=0):
					if(zdiff==90 or zdiff==(-270)):
						actions.append([0,0,1,0])
					else:
						actions.append([0,1,0,0])
			actions.append([0,0,0,1])
	temp['actions']=actions
	#print "temp = ", temp
	count=count+1

	if mapName == "Grid":
		Grid.append(temp)
	elif mapName == "Jelly":
		Jelly.append(temp)
	elif mapName == "L":
		L.append(temp)
	else:
		print "Nothing ----------"

#print len(Grid) #874
#print len(Jelly) #1293
#print len(L) #1070
#print count #3237

dataset={}
dataset['Grid']=Grid
dataset['Jelly']=Jelly
dataset['L']=L
#print len(dataset) #3

pickle.dump( dataset, open( "data.pickle", "wb" ) )

#temp1 = pickle.load( open( "data.pickle", "rb" ) )
#print len(temp1)


