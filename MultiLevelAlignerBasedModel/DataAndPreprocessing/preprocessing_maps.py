import xml.etree.ElementTree as ET
tree = ET.parse('map-grid.xml')
root = tree.getroot()
print "---Root----Root.attribute---"
print root.tag,"  ",root.attrib
    
dic={}
walls={'butterfly':0, 'fish':1, 'tower':2}
floors={'concrete':3, 'yellow':4, 'flower':5, 'grass':6, 'brick':7, 'gravel':8, 'wood':9, 'blue':10}
objects={'hatrack':11, 'lamp':12, 'chair':13, 'sofa':14, 'barstool':15, 'easel':16}

for child in root:
	dic[child.tag]=[]
	for gchild in child:
		x=gchild.tag
		dic[child.tag].append(gchild.attrib)
		#print gchild.tag," ",gchild.attrib

for key in dic:
	print "key = ", key
	for value in dic[key]:
		print value
	print "\n\n"

print "------------------------------------------------------------"

def search_edge_dictionary(key1, key2, value1, value2):
    #print key1, key2, value1, value2
    for elem in dic['edges']:
        if ((elem[key1] == value1 and elem[key2] == value2) or (elem[key2] == value1 and elem[key1] == value2)):
            return elem
    
newNodeDic={}
for value in dic['nodes']:
    temp_str = value['x']+","+value['y']
    newNodeDic[temp_str]=value
#creates a mapping of coordinate-in-string-format to the node
        
#print dic['nodes']
#print dic['edges']
nodes_dic={}
nodes_list=[]
count270=0
count90=0
count180=0
count0=0

for value in dic['nodes']:
    temp={}
    #print value gives {'y': '5', 'x': '0', 'item': ''}
    temp['y']=eval(value['y'])
    temp['x']=eval(value['x'])
    #print temp gives {'y': 5, 'x': 0}
    temp['item']=value['item']
    objAtNode=temp['item']
    objvec=[0,0,0,0,0,0]
    if(objAtNode=='hatrack'):
        objvec[0]=1
    elif(objAtNode=='lamp'):
        objvec[1]=1
    elif(objAtNode=='chair'):
        objvec[2]=1
    elif(objAtNode=='sofa'):
        objvec[3]=1
    elif(objAtNode=='barstool'):
        objvec[4]=1
    elif(objAtNode=='easel'):
        objvec[5]=1
    temp['objvec']=objvec
    #----------------------for neighbors-------------------------
    temp['neighbors']={}
    cur_node=str(temp['x'])+","+str(temp['y'])
    #print cur_node
    try:
        next_node=str(temp['x']-1)+","+str(temp['y'])
        #print "next_node = ",next_node
        if (newNodeDic[next_node]):
            edge=search_edge_dictionary('node1','node2',next_node,cur_node)
            if (edge):
                temp['neighbors']['270']={}
                temp['neighbors']['270']['node']=newNodeDic[next_node]
                temp['neighbors']['270']['edge']={}
                temp['neighbors']['270']['edge']['wall']=edge['wall']
                temp['neighbors']['270']['edge']['floor']=edge['floor']
            count270=count270+1
    except KeyError:
        pass
    try:
        next_node=str(temp['x']+1)+","+str(temp['y'])
        #print "next_node = ",next_node
        if (newNodeDic[next_node]):
            edge=search_edge_dictionary('node1','node2',next_node,cur_node)
            if (edge):
                temp['neighbors']['90']={}
                temp['neighbors']['90']['node']=newNodeDic[next_node]
                temp['neighbors']['90']['edge']={}
                temp['neighbors']['90']['edge']['wall']=edge['wall']
                temp['neighbors']['90']['edge']['floor']=edge['floor']
            count90=count90+1
    except KeyError:
        pass
    try:
        next_node=str(temp['x'])+","+str(temp['y']+1)
        #print "next_node = ",next_node
        if (newNodeDic[next_node]):
            edge=search_edge_dictionary('node1','node2',next_node,cur_node)
            if (edge):
                temp['neighbors']['180']={}
                temp['neighbors']['180']['node']=newNodeDic[next_node]
                temp['neighbors']['180']['edge']={}
                temp['neighbors']['180']['edge']['wall']=edge['wall']
                temp['neighbors']['180']['edge']['floor']=edge['floor']
            count180=count180+1
    except KeyError:
        pass
    try:
        next_node=str(temp['x'])+","+str(temp['y']-1)
        #print "next_node = ",next_node
        if (newNodeDic[next_node]):
            edge=search_edge_dictionary('node1','node2',next_node,cur_node)
            if (edge):
                temp['neighbors']['0']={}
                temp['neighbors']['0']['node']=newNodeDic[next_node]
                temp['neighbors']['0']['edge']={}
                temp['neighbors']['0']['edge']['wall']=edge['wall']
                temp['neighbors']['0']['edge']['floor']=edge['floor']
            count0=count0+1
    except KeyError:
        pass
   
                       
    print "temp = ",temp
    nodes_list.append(temp)
#print "270COUNT = ",count270 #gives 20
#print "180COUNT = ",count180 #gives 23
#print "90COUNT = ",count90 #gives 20
#print "0COUNT  = ",count0 #gives 23

nodes_dic['nodes'] = nodes_list



