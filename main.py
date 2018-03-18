
# coding: utf-8

# # Домашнее задание по курсу Теория конечных графов и ее приложения

# ## Классы для работы с узлами и дорогами

# In[40]:


import svgwrite
import time
from lxml import etree
import pandas as pd
import numpy as np


# In[2]:


class Way:
    def __init__(self, id, way_type = 'none', nodes = []):
        self.id = id
        self.way_type = way_type
        self.nodes = nodes
class Node:
    def __init__(self, id, lat, lon):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.is_start_node = False
        self.is_end_node = False
        self.is_in_road = False
        self.is_crossroad = False
        self.is_in_highway = False
        self.number_in_dict = 0
    def isDeletable(self):
        return not(self.is_end_node or self.is_start_node or self.is_crossroad )


# ## Парсинг

# In[29]:


def parse_osm(filename = 'kal.osm'):

    print("Parsing .osm...")
    counted_highways = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
                   'motorway_link', 'trunk_link',  'tertiary_link',
                   'primary_link', 'secondary_link'} 
    
    number_of_ways = 0
    number_of_nodes = 0

    tree = etree.iterparse(filename, events=("start", "end"))
    for event, elem in tree:
        # parsing ways
        if elem.tag == 'way' and event == 'start':

            way = Way(elem.get('id'))

            children = elem.getchildren()
            nd = []
            is_highway = False

            for child in children:
                if child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in counted_highways :
                    way.way_type = child.get('v')
                    is_highway = True
            if is_highway:
                for child in children:
                    if child.tag == 'nd':
                        nd.append(child.get('ref'))
                way.nodes = nd
                ways.append(way)
                number_of_ways += 1

        # parsing nodes
        if elem.tag == 'node' and event == 'start':
            node = Node(elem.get('id'), elem.get('lat'), elem.get('lon'))
            nodes[elem.get('id')] = node
            number_of_nodes += 1
        elem.clear()            

    print('Node number:',number_of_nodes)
    print('Road number:',number_of_ways)
    print("Done")


# ## Удаление промежуточных узлов

# In[7]:


def delete_transitional_nodes():
    print('\nDeleting transitional nodes...')
    print('Nodes number before:',len(nodes))
    for way in ways:
        is_first_node = True
        nodenum = 0
        for node in way.nodes:
            nodes.get(node).is_in_highway = True
            nodenum += 1
            if is_first_node:
                nodes.get(node).is_start_node = True
                is_first_node = False
            elif nodenum == len(way.nodes):
                nodes.get(node).is_end_node = True
            else:
                if nodes.get(node).is_in_road:
                    nodes.get(node).is_crossroad = True
                else:
                    nodes.get(node).is_in_road = True

    to_pop_list = []
    for node in nodes:
        if not nodes.get(node).is_in_highway:
            to_pop_list.append(str(node))
    for i in range(0,len(to_pop_list)-1):
        nodes.pop(to_pop_list[i])

    for way in ways:
        list_to_remove = []
        for node in way.nodes:
            if nodes.get(node).isDeletable():
                nodes.pop(node)
                list_to_remove.append(node)
        for i in range(0,len(list_to_remove)):
            way.nodes.remove(list_to_remove[i])
    print('Nodes number after:',len(nodes))

    number = 0
    for node in nodes:
        number += 1
        nodes.get(node).number_in_dict = number
    print("Done")


# ## Отрисовка карты 

# In[13]:


def trans(a):
    return ((a * 10e6) )
def draw_map(svgurl = "pictures/kaliningrad_map.svg"): 
    print("\nDrawing map...")
    ratio = 1.3 # to avoid too much deformation
    ypicsize = 3000
    xpicsize = int(ypicsize / ratio)
    ypicsizepx = str(xpicsize) + 'px'
    xpicsizepx = str(ypicsize) + 'px'

    start_time = time.time()

    # TODO: to get from .osm
    minlat = 54.6266000
    maxlat = 54.7817000
    minlon = 20.2794000
    maxlon = 20.6632000

    scalelat = ((maxlat - minlat) * 10e6) / xpicsize
    scalelon = ((maxlon - minlon) * 10e6) / ypicsize

    svg_document = svgwrite.Drawing(filename = svgurl, size = (xpicsizepx, ypicsizepx))
    svg_document.add(svg_document.text("Kaliningrad roadmap",insert = (40, 40)))
    svg_document.save()

    black_roads = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
    green_roads = ['unclassified', 'residential']
    for elem in ways:

        color = 'blue' # small roads
        swidth = 1
        if elem.way_type in black_roads:
            color = 'black' # big roads
            swidth = 3
        if elem.way_type in green_roads:
            color = 'green' # other roads
            swidth = 2

        points = []
        for i in range(0, len(elem.nodes)):
            y = (trans(maxlat) - trans(float(nodes.get(elem.nodes[i]).lat))) / scalelat
            x = ypicsize - (trans(maxlon) - trans(float(nodes.get(elem.nodes[i]).lon))) / scalelon
            points.append((x,y))
            svg_document.add(svg_document.circle(center = (x,y), r = 2, stroke = "red")) # write nodes to svg
        svg_document.add(svg_document.polyline(points, stroke=color, stroke_width = swidth, fill='none')) # write roads to svg

    time_final = (time.time() - start_time)
    print("--- %s seconds ---" % time_final)
    svg_document.save()
    print("Done\n\n")


# ## Матрица смежности и список смежности в .csv

# In[9]:


def write_csv(matrix_path = 'csv/adjacency_matrix.csv', list_path = 'adjacency_list.csv'):
    print("\nWriting into csv...")
    node_count = len(nodes)
    adjacency_matrix = np.zeros([node_count, node_count], dtype=np.int8) #int\
    adjacency_list = {}

    for w in ways:
        for n in range(len(w.nodes) - 1):
            x = nodes.get(w.nodes[n])
            y = nodes.get(w.nodes[n+1])

            adjacency_matrix[x.number_in_dict,y.number_in_dict] = 1
            adjacency_matrix[y.number_in_dict,x.number_in_dict] = 1

            temp = adjacency_list.get(x.id,[])
            temp.append(y.id)
            adjacency_list.update({x.id:temp})
            temp = adjacency_list.get(y.id,[])
            temp.append(x.id)
            adjacency_list.update({y.id:temp})

    df_am = pd.DataFrame(adjacency_matrix, columns=nodes.keys())
    df_am.index = nodes.keys()

    df_al = pd.DataFrame.from_dict(adjacency_list, orient="index")

    df_am.to_csv(matrix_path)
    df_al.to_csv(list_path)
    print("Done\n\n")


# In[74]:


df_am.head(10)


# In[75]:


df_al.head(10)


# In[43]:


ways = []
nodes = {}

parse_osm()
delete_transitional_nodes()
draw_map()
write_csv()
print("All done")

