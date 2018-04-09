
# coding: utf-8

# # Домашнее задание по курсу Теория конечных графов и ее приложения

# ## Классы для работы с узлами и дорогами

# In[9]:


import svgwrite
import time
from lxml import etree
import sys
import pandas as pd
import numpy as np
import geopy.distance as geo


# In[56]:


class Way:
    def __init__(self, id, way_type = 'none', nodes = []):
        self.id = id
        self.way_type = way_type
        self.nodes = nodes
        self.name = 'untitled'
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
        self.is_in_hospital = False
        self.closest_node = 0
        self.number_in_dict = 0
        self.is_entrance = False
        self.name = 'untitled'
    def isDeletable(self):
        return not(self.is_end_node or self.is_start_node or self.is_crossroad)


# ## Парсинг

# In[55]:


def parse_osm(filename = 'kal.osm'):
    print("Parsing .osm...")
    
    nodes = {}
    ways = []
    bounds = []
    start_time = time.time()
    counted_highways = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'living_street',
                   'motorway_link', 'trunk_link',  'tertiary_link', 'service', 'residential',
                   'primary_link', 'secondary_link', 'road'} 
    
    number_of_ways = 0
    number_of_nodes = 0
    hospital_count = 0

    tree = etree.iterparse(filename, events=('start',"end",))
    for event, elem in tree:
        # parsing ways
        way = Way(0)
        if elem.tag == 'bounds':
            bounds.append(elem.get('minlat'))
            bounds.append(elem.get('minlon'))
            bounds.append(elem.get('maxlat'))
            bounds.append(elem.get('maxlon'))
        if elem.tag == 'way':
            way.id = elem.get('id')
            if event == 'end':
                children = elem.getchildren()
                nd = []
                is_highway = False
                is_hospital = False

                for child in elem.iter('nd', 'tag'):
                    if child.tag == 'tag' and child.get('k') == 'highway' and child.get('v') in counted_highways :
                        way.way_type = child.get('v')
                        is_highway = True
                    if child.tag == 'tag' and child.get('k') == 'amenity' and child.get('v') == "hospital":
                        hospital_count += 1
                        if hospital_count < 11:
                            way.way_type = 'hospital'
                            is_hospital = True
                    if child.tag == 'tag' and child.get('k') == 'name':
                        way.name = child.get('v')
                if is_highway or is_hospital:
                    for child in elem.iter('nd', 'tag'):
                        if child.tag == 'nd':
                            nd.append(child.get('ref'))
                    way.nodes = nd
                    ways.append(way)
                    number_of_ways += 1
                elem.clear()
        # parsing nodes
        if elem.tag == 'node':
            node = Node(elem.get('id'), elem.get('lat'), elem.get('lon'))
            if event == "end":
                children = elem.getchildren()
                for child in children:
                    if child.tag == 'tag' and child.get('k') == 'entrance':
                        node.is_entrance = True
                nodes[elem.get('id')] = node
                number_of_nodes += 1 
                elem.clear()

    print('Node number:',number_of_nodes)
    print('Road number:',number_of_ways)
    time_final = (time.time() - start_time)
    print("--- %s seconds ---" % time_final)
    print("Done")
    
    return nodes, ways, bounds


# ## Удаление промежуточных узлов

# In[12]:


def delete_transitional_nodes(nodes,ways):
    print('\nDeleting transitional nodes...')
    print('Nodes number before:',len(nodes))
    
    start_time = time.time()
    
    hospital_nodes = []
    
    for way in ways:
        if way.way_type == 'hospital':
            for node in way.nodes:
                nodes.get(node).is_in_hospital = True
                hospital_nodes.append(node)
                continue
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
        if not nodes.get(node).is_in_highway and not nodes.get(node).is_in_hospital:
            to_pop_list.append(str(node))
    for i in range(0,len(to_pop_list)-1):
        nodes.pop(to_pop_list[i])

    for way in ways:
        if way.way_type == 'hospital':
            continue
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
        
    time_final = (time.time() - start_time)
    print("--- %s seconds ---" % time_final)
    print("Done")
    
    return nodes,ways, hospital_nodes


# ## Отрисовка карты 

# In[170]:


def trans(a):
    return ((a * 10e3) )
def draw_map(nodes, ways, bounds, svgurl = "pictures/kaliningrad_map.svg", temp=[], shortest_way = []): 
    print("\nDrawing map...")
    
    ratio = 1.3 # to avoid too much deformation
    ypicsize = 10000
    xpicsize = int(ypicsize / ratio)
    ypicsizepx = str(xpicsize) + 'px'
    xpicsizepx = str(ypicsize) + 'px'

    start_time = time.time()

    # Bounds
    minlat = float(bounds[0]) #54.6266
    maxlat = float(bounds[2]) #54.7817
    minlon = float(bounds[1]) #20.2794
    maxlon = float(bounds[3]) #20.6632

    scalelat = trans(maxlat - minlat) / xpicsize
    scalelon = trans(maxlon - minlon) / ypicsize

    svg_document = svgwrite.Drawing(filename = svgurl, size = (xpicsizepx, ypicsizepx))

    yellow_roads = ['motorway','trunk','primary']
    black_roads = ['secondary', 'tertiary','unclassified', 'residential']
    hospitals = ['hospital']
    for elem in ways:
        color = 'blue' # small roads
        swidth = 1
        if elem.way_type in yellow_roads:
            color = 'yellow' # the biggest roads
            swidth = 6
        if elem.way_type in black_roads:
            color = 'black' # big roads
            swidth = 3
        if elem.way_type in hospitals:
            color = 'red' # hospitals
            swidth = 4

        points = []
        for i in range(0, len(elem.nodes)):
            y = (trans(maxlat) - trans(float(nodes.get(elem.nodes[i]).lat))) / scalelat
            x = ypicsize - (trans(maxlon) - trans(float(nodes.get(elem.nodes[i]).lon))) / scalelon
            points.append((x,y))
    
            point_color = 'grey'
            point_r = 2
            if nodes.get(elem.nodes[i]).is_entrance and nodes.get(elem.nodes[i]).is_in_hospital:
                point_color = 'green'
                point_r = 6
            svg_document.add(svg_document.circle(center = (x,y), r = point_r, stroke = point_color, fill=point_color )) # write nodes to svg
            
        svg_document.add(svg_document.polyline(points, stroke=color, stroke_width = swidth, fill='none')) # write roads to svg 
    points = []
    if len(temp) > 0:
        for way in temp:
            points.clear()
            for elem in way:
                y = (trans(maxlat) - trans(float(nodes.get(elem).lat))) / scalelat
                x = ypicsize - (trans(maxlon) - trans(float(nodes.get(elem).lon))) / scalelon
                points.append((x,y))
            svg_document.add(svg_document.polyline(points, stroke="fuchsia", stroke_width = 8, fill='none'))
    if len(shortest_way) > 0:
        points.clear()
        x = 0
        y = 0
        for elem in shortest_way:
            y = (trans(maxlat) - trans(float(nodes.get(elem).lat))) / scalelat
            x = ypicsize - (trans(maxlon) - trans(float(nodes.get(elem).lon))) / scalelon
            points.append((x,y))
        svg_document.add(svg_document.polyline(points, stroke="red", stroke_width = 9, fill='none'))
        svg_document.add(svg_document.circle(center = (x,y), r = 15, stroke = 'aqua', fill='aqua')) 
    time_final = (time.time() - start_time)
    print("--- %s seconds ---" % time_final)
    svg_document.save()
    print("Done")


# ## Матрица смежности и список смежности

# In[14]:


def form_adj(nodes, ways):
    print("\nForming adjacency matrix and list...")
    
    start_time = time.time()
    
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

    '''df_am.to_csv(matrix_path)
    df_al.to_csv(list_path)'''
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    
    return df_al, df_am
    #df_am.head(10)
    #df_al.head(10)


# ## Запись в .csv

# In[15]:


def write_into_csv(df_am, df_al, matrix_path = 'csv/adjacency_matrix.csv', list_path = 'csv/adjacency_list.csv'):
    print("\nWriting into csv...")
    
    start_time = time.time()
    
    df_am.to_csv(matrix_path)
    df_al.to_csv(list_path)
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")


# ## Поиск кратчайшей дороги до больницы

# In[147]:


def find_ways_to_hospital(nodes, adj_list, h_nodes, v_set, start):
    
    main_time = time.time()
    def get_key(item):
        return found_ways.get(item)[1]

    print("\nLooking for the shortest way...")
    dijkstra_shortest = []
    a_star_shortest = []
    levit_shortest = []
    found_ways = {}
    dijkstra_ways = {}
    a_star_ways = {}
    levit_ways = {}
    found_ways = {}
    
    # Dijkstra algorithm
    print("\nImplementing Dijkstra algorithm...")
    start_time = time.time()
    
    d, p = dijkstra(adj_list, v_set, start)
    for node in h_nodes:
        if nodes.get(node).closest_node != 0:
            target_node = nodes.get(node).closest_node
        else:
            target_node = node
        found_ways.update({node: (reconstruct_path(p, target_node), d.get(target_node))})
    final_time = (time.time() - start_time)
    print("--- %s seconds ---" % final_time)
    
    for way in ways:
        if way.way_type == 'hospital':
            min_dist = [way.nodes[0], found_ways.get(way.nodes[0])[1]]
            for node in way.nodes:
                if found_ways.get(node)[1] < min_dist[1]:
                    min_dist = [node, found_ways.get(node)[1]]
            temp = 0
            for node in way.nodes:
                temp += 1
                if min_dist[0] != node and temp != len(way.nodes):
                    found_ways.pop(node)
    
    dijkstra_ways = found_ways
    dijkstra_shortest = dijkstra_ways.get(min(dijkstra_ways, key = get_key))[0]
    
    found_ways.clear()
    
    # A Star algorithm
    print("\nImplementing A* algorithm...")
    start_time = time.time()
    for node in h_nodes:
        if nodes.get(node).closest_node != 0:
            target_node = nodes.get(node).closest_node
        else:
            target_node = node
        found_ways.update({node: a_star(adj_list, v_set, nodes, start, target_node)})
    final_time = (time.time() - start_time)
    print("--- %s seconds ---" % final_time)
    
    for way in ways:
        if way.way_type == 'hospital':
            min_dist = [way.nodes[0], found_ways.get(way.nodes[0])[1]]
            for node in way.nodes:
                if found_ways.get(node)[1] < min_dist[1]:
                    min_dist = [node, found_ways.get(node)[1]]
            temp = 0
            for node in way.nodes:
                temp += 1
                if min_dist[0] != node and temp != len(way.nodes):
                    found_ways.pop(node)
            
    a_star_ways = found_ways
    a_star_shortest = a_star_ways.get(min(a_star_ways, key = get_key))[0]
        
    
    # Finding ways
    '''for node in nodes:
        if nodes.get(node).is_in_hospital:
            if d.get(node) == sys.maxsize:
                found_ways.update({nodes.get(node).closest_node: d.get(nodes.get(node).closest_node)})
            else:
                found_ways.update({node: d.get(node)})

    shortest_way = min(found_ways, key = found_ways.get)

    # Finding shortest way
    node = shortest_way
    while node != start:
        closest_way.append(node)
        node = p.get(node)'''
    
    
    final_time = (time.time() - main_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    return dijkstra_shortest, levit_shortest, a_star_shortest, dijkstra_ways, levit_ways, a_star_ways


# ## DFS 

# In[17]:


def DFS(vertex_set, v):
    print("\nDFS...")
    start_time = time.time()
    discovered = []
    
    def DFS_rec(vertex_set, v):
        
        discovered.append(v)

        for w in vertex_set.get(v):
            if w[0] not in discovered:
                DFS_rec(vertex_set, w[0])
            #print(discovered)
    
    DFS_rec(vertex_set,v)
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    
    return discovered


# ## Нахождение подъездов к больницам

# In[174]:


def find_closest_node(nodes, discovered, coordinates):
    dist = [0, sys.maxsize]
    for node in nodes:
        if nodes.get(node).is_in_highway and not nodes.get(node).is_in_hospital and node in discovered:
                    coords_1 = coordinates
                    coords_2 = (nodes.get(node).lat, nodes.get(node).lon)
                    if dist[1] > geo.vincenty(coords_1, coords_2).km:
                        dist[0] = node
                        dist[1] = geo.vincenty(coords_1, coords_2).km
    return dist[0]

def set_closest_nodes(hosp_nodes, nodes, discovered):
    print('\nFinding reachable roads to hospital...')
    
    start_time = time.time()
    
    for node_1 in hosp_nodes:
        if nodes.get(node_1).is_in_hospital and node_1 not in discovered:
            dist = [0, sys.maxsize]
            for node_2 in nodes:
                if nodes.get(node_2).is_in_highway and not nodes.get(node_2).is_in_hospital:
                    coords_1 = (nodes.get(node_1).lat, nodes.get(node_1).lon)
                    coords_2 = (nodes.get(node_2).lat, nodes.get(node_2).lon)
                    if dist[1] > geo.vincenty(coords_1, coords_2).km:
                        dist[0] = node_2
                        dist[1] = geo.vincenty(coords_1, coords_2).km
            result = nodes.pop(node_1)
            result.closest_node = dist[0]
            nodes.update({node_1: result})
            
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    return nodes


# ## Алгоритм Дейкстры

# In[108]:


def getKey(item):
    return item[1]

def vertex_set(adj_list,nodes):
    row_num, col_num = adj_list.shape
    vertex_set = {}
    for i in range(0, row_num):
        adj_nodes = []
        for j in range(0, col_num):
            if adj_list[j][i] in nodes:
                coords_1 = (nodes.get(adj_list.index[i]).lat, nodes.get(adj_list.index[i]).lon)
                coords_2 = (nodes.get(adj_list[j][i]).lat, nodes.get(adj_list[j][i]).lon)
                adj_nodes.append((adj_list[j][i], geo.vincenty(coords_1, coords_2).km))
            else:
                break
        vertex_set.update({adj_list.index[i]: adj_nodes})
    return vertex_set

def dijkstra(adj_list, vertex_set, start):
    #print("\nImplementing Dijkstra algorithm...")
    
    row_num, col_num = adj_list.shape
    start_time = time.time()
    
    unvisited = []
    dist = {}
    dist_of_unvisited = {}
    prev = {}
    
    for i in range(0, row_num):
        unvisited.append(adj_list.index[i])
        dist.update({adj_list.index[i]: sys.maxsize})
        dist_of_unvisited.update({adj_list.index[i]: sys.maxsize})
    
    dist.update({start:0})  
    dist_of_unvisited.update({start:0})
    
    while len(unvisited) != 0:
        u = min(dist_of_unvisited, key = dist_of_unvisited.get)
        unvisited.remove(u)
        dist_of_unvisited.pop(u)
        
        for i in range(0,len(vertex_set.get(u))):
            v = vertex_set.get(u)[i][0]
            alt = dist.get(u) + vertex_set.get(u)[i][1]
            if alt < dist.get(v):
                dist.update({v:alt})
                dist_of_unvisited.update({v:alt})
                prev.update({v:u})
                
    final_time = (time.time() - start_time)
    
    #print("--- %s seconds ---" % final_time)
    #print("Done")
    
    return dist, prev


# ## Алгоритм Левита 

# In[20]:


def levit(adj_list, vertex_set, start = '534451360'):
    print("\nImplementing Levit algorithm...")
    start_time = time.time()
    
    row_num, col_num = adj_list.shape
    
    dist = {}
    prev = {}
    
    M0 = []
    M1 = []
    M2 = []
    
    for i in range(0, row_num):
        M2.append(adj_list.index[i])
        dist.update({adj_list.index[i]: sys.maxsize})
        
    M2.remove(start)
    M1.append(start)
    
    #while M0
    
    final_time = (time.time() - start_time)
    print("--- %s seconds ---" % final_time)
    print("Done")
    
    return dist, prev


# ## Алгоритм A*

# In[77]:


def heuristic_cost_estimate(a, b, metric = "eucl"):
    result = 0
    a = (float(a[0]), float(a[1]))
    b = (float(b[0]), float(b[1]))
    if metric == "cheb":
        result = -1
        for i in range(0,len(a)):
            if result < math.ceil(a[i] - b[i]):
                result = math.ceil(a[i] - b[i])
    elif metric == "eucl":
        for i in range(0,len(a)):
            result += (a[i] - b[i]) ** 2
        result = result ** (0.5)
    elif metric == "manh":
        for i in range(0,len(a)):
            result += math.ceil(a[i] - b[i]) 
    return result

def a_star(adj_list, vertex_set, nodes, start = '534451360', end = '2979387711', metric = 'eucl'):
    #print("\nImplementing A* algorithm...")
    start_time = time.time()
    
    row_num, col_num = adj_list.shape
    
    opened = []
    closed = []
    dist = {}
    fscore = {}
    #dist_of_unvisited = {}
    prev = {}
    
    for i in range(0, row_num):
        dist.update({adj_list.index[i]: sys.maxsize})
        #dist_of_unvisited.update({adj_list.index[i]: sys.maxsize})
    
    start_point = (nodes.get(start).lat, nodes.get(start).lon)
    end_point = (nodes.get(end).lat, nodes.get(end).lon)
    dist.update({start: 0})  
    fscore.update({start: heuristic_cost_estimate(start_point, end_point, metric)})
    opened = [start]
    
    while opened:
        current_node = min(opened, key = fscore.get)
        
        if current_node == end:
            final_time = (time.time() - start_time)
            #print("--- %s seconds ---" % final_time)
            #print("Done")
            return (reconstruct_path(prev, current_node), dist.get(end))
        
        opened.remove(current_node)
        closed.append(current_node)
        
        for neighbour in vertex_set.get(current_node):
            if neighbour[0] in closed:
                continue
            if neighbour[0] not in opened:
                opened.append(neighbour[0])
                
            tentative_dist = dist.get(current_node) + neighbour[1]
            if tentative_dist >= dist.get(neighbour[0]):
                continue
            
            prev.update({neighbour[0]: current_node})
            dist.update({neighbour[0]: tentative_dist})
            start_point = (nodes.get(neighbour[0]).lat, nodes.get(neighbour[0]).lon)
            end_point = (nodes.get(end).lat, nodes.get(end).lon)
            fscore.update({neighbour[0]: dist.get(neighbour[0]) + heuristic_cost_estimate(start_point, end_point, metric)})
    
    
    final_time = (time.time() - start_time)
    #print("--- %s seconds ---" % final_time)
    print("Failure")
    
    return -1

def reconstruct_path(prev, current_node):
    total_path = [current_node]
    while current_node in prev.keys():
        current_node = prev.get(current_node)
        total_path.append(current_node)
    return total_path


# ## MAIN

# In[69]:


if __name__ == "__main__":
    nodes, ways, bounds = parse_osm()
    nodes, ways, h_nodes = delete_transitional_nodes(nodes, ways)
    
    draw_map(nodes, ways, bounds)
    
    adj_list, adj_matrix = form_adj(nodes, ways)
    # calculating distances between nodes in km
    v_set = vertex_set(adj_list, nodes)
    reachable_nodes = DFS(v_set, '532159053')
    
    # for hospital nodes that are not connected to road
    # works too slow
    # TODO: write somewhere
    nodes = set_closest_nodes(h_nodes, nodes, reachable_nodes)
    
    #write_into_csv(adj_list, adj_matrix)
    print("\nAll done")


# In[177]:


print("Enter your coordinates")
lat = input("Latitude: ")
lon = input("Longitude: ")

start = find_closest_node(nodes,reachable_nodes,(lat,lon))

a,b,c, a_ways, b_ways, c_ways = find_ways_to_hospital(nodes, adj_list, h_nodes, v_set, start)

ways_to_draw = []
for way in a_ways:
    ways_to_draw.append(a_ways.get(way)[0])
    
draw_map(nodes, ways, bounds, 'pictures/test3.svg', ways_to_draw, a)


# In[314]:


adj_list.head(10)

