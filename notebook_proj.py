
# coding: utf-8

# # Домашнее задание по курсу Теория конечных графов и ее приложения

# ## Классы для работы с узлами и дорогами

# In[6]:


import svgwrite
import time
from lxml import etree
import sys
import pandas as pd
import numpy as np
import geopy.distance as geo
import math
import random as r
from heapq import heappush, heappop


# In[7]:


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
        self.number_in_dict = 0
        self.is_entrance = False
        self.name = 'untitled'
    def isDeletable(self):
        return not(self.is_end_node or self.is_start_node or self.is_crossroad)


# # Задание 1

# ## Парсинг

# In[8]:


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

# In[9]:


def delete_transitional_nodes(nodes,ways):
    print('\nDeleting transitional nodes...')
    print('Nodes number before:',len(nodes))
    
    start_time = time.time()
    
    hospital_coord = []
    
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
    
    return nodes,ways


# ## Отрисовка карты 

# In[63]:


def trans(a):
    return ((a * 10e3) )
def draw_map(nodes, ways, bounds, svgurl = "pictures/kaliningrad_map.svg", temp=[], shortest_way = [], isTSP = False): 
    print("\nDrawing map...")
    
    ratio = 1.3 # to avoid too much deformation
    ypicsize = 5000
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

    yellow_roads = {'motorway','trunk','primary'}
    black_roads = {'secondary', 'tertiary','unclassified', 'residential'}
    hospitals = {'hospital'}
    
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
            svg_document.add(svg_document.circle(center = (x,y), r = point_r, stroke = point_color, fill=point_color )) # write nodes to svg
            
        svg_document.add(svg_document.polyline(points, stroke=color, stroke_width = swidth, fill='none')) # write roads to svg 
    points = []
    if len(temp) > 0:
        count = 0
        number = (0,0)
        for way in temp:
            count = count + 1
            points.clear()
            isFirst = True
            goToLast = False
            way_len = len(way)
            iterat = 0;
            for elem in way:
                iterat = iterat + 1
                y = (trans(maxlat) - trans(float(nodes.get(elem).lat))) / scalelat
                x = ypicsize - (trans(maxlon) - trans(float(nodes.get(elem).lon))) / scalelon
                if isFirst:
                    isFirst = False
                    if number != (x,y):
                        number = (x,y)
                    else:
                        goToLast = True
                if goToLast and iterat == way_len:
                    number = (x,y)
                points.append((x,y))
            svg_document.add(svg_document.polyline(points, stroke="fuchsia", stroke_width = 8, fill='none'))
            if isTSP and count != 11:
                svg_document.add(svg_document.text(str(count), insert=number, font_size=100))
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
     
    print('\nSaving .svg...')
    svg_document.save()
    time_final = (time.time() - start_time)
    #print("--- %s seconds ---" % time_final)
    #print("Done")


# ## Матрица смежности и список смежности

# In[11]:


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
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    
    return adjacency_list, df_al, df_am
    #df_am.head(10)
    #df_al.head(10)


# In[12]:


#adj_list.head(10)


# In[13]:


#adj_matrix.head(10)


# ## Запись в .csv

# In[61]:


def write_into_csv(df_am, df_al, matrix_path = 'csv/adjacency_matrix.csv', list_path = 'csv/adjacency_list.csv'):
    print("\nWriting into csv...")
    
    start_time = time.time()
    
    df_am.to_csv(matrix_path)
    df_al.to_csv(list_path)
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    
def write_into_csv_shortest_paths(paths, url = 'csv/shortest_paths.csv'):
    print("\nWriting into csv...")
    
    start_time = time.time()
    
    matrix = []
    for path in paths:
        matrix.append(paths.get(path)[0])
        
    df_sp = pd.DataFrame(matrix)
    df_sp.to_csv(url)
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    
def write_into_csv_tsp_routes(routes, url = 'csv/tsp_routes.csv'):
    print("\nWriting into csv...")
    start_time = time.time()
    
    matrix = []
    for route in routes:
        matrix.append(route)
    df_sp = pd.DataFrame(matrix)
    df_sp.to_csv(url)
    
    final_time = (time.time() - start_time)
    print("--- %s seconds ---" % final_time)


# # Задание 2

# ## DFS 

# In[15]:


def DFS(adjacency_list, v):
    print("\nDFS...")
    start_time = time.time()
    discovered = set()
    
    def DFS_rec(adjacency_list, v):
        
        discovered.add(v)

        for w in adjacency_list.get(v):
            if w not in discovered:
                DFS_rec(adjacency_list, w)
    
    DFS_rec(adjacency_list,v)
    
    final_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % final_time)
    print("Done")
    
    return discovered


# ## Алгоритм Дейкстры

# In[16]:


def dijkstra(adjacency_list, start):
    
    dist = {}
    prev = {}
    unvisited = set()
    visited = [] 
    heappush(visited, (0, start))
    
    for i in adjacency_list:
        unvisited.add(i)
        dist.update({i: sys.maxsize})
    
    dist.update({start:0})
    
    while visited:
        current = heappop(visited)
        u = current[1]
        
        if u not in unvisited:
            continue
        
        unvisited.remove(u)
        
        for i in adjacency_list.get(u):
            coords_1 = (nodes.get(u).lat, nodes.get(u).lon)
            coords_2 = (nodes.get(i).lat, nodes.get(i).lon)
            alt = dist.get(u) + geo.vincenty(coords_1, coords_2).km
            if alt < dist.get(i):
                dist.update({i:alt})
                prev.update({i:u})
                heappush(visited,(alt, i))
    
    return dist, prev


# ## Алгоритм Левита 

# In[17]:


def levit(adjacency_list, hospital_nodes, start):
    #print("\nImplementing Levit algorithm...")
    #start_time = time.time()
    
    '''dist = {}
    prev = {}
    
    M0 = []
    M1 = []
    M2 = []
    
    for i in range(0, row_num):
        M2.append(adj_list.index[i])
        dist.update({adj_list.index[i]: sys.maxsize})
        
    M2.remove(start)
    M1.append(start)'''
    
    dist = {}
    prev = {}

    q = [] #heap
    heappush(q, (0, start))
    
    for i in adjacency_list:
        dist.update({i: sys.maxsize})
    
    dist.update({start:0})
    
    while q:
        current = heappop(q)
        u = current[1]
        
        for i in adjacency_list.get(u):
            coords_1 = (nodes.get(u).lat, nodes.get(u).lon)
            coords_2 = (nodes.get(i).lat, nodes.get(i).lon)
            alt = dist.get(u) + geo.vincenty(coords_1, coords_2).km
            if alt < dist.get(i):
                dist.update({i:alt})
                prev.update({i:u})
                heappush(q,(alt, i))
    
    #while M0
    
    #final_time = (time.time() - start_time)
    #print("--- %s seconds ---" % final_time)
    #print("Done")
    
    return dist, prev


# ## Алгоритм A*

# In[18]:


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

def a_star(adjacency_list, nodes, start = '534451360', end = '2979387711', metric = 'manh'):
    #print("\nImplementing A* algorithm...")
    start_time = time.time()
    
    opened = []
    closed = set()
    dist = {}
    fscore = {}
    prev = {}
    
    for i in adjacency_list:
        dist.update({i: sys.maxsize})
    
    start_point = (nodes.get(start).lat, nodes.get(start).lon)
    end_point = (nodes.get(end).lat, nodes.get(end).lon)
    dist.update({start: 0})  
    #fscore.update({start: heuristic_cost_estimate(start_point, end_point, metric)})
    
    fscore = heuristic_cost_estimate(start_point, end_point, metric);
    heappush(opened, (fscore, start))
    
    while opened:
        current_node = heappop(opened)
        u = current_node[1]
        
        if u == end:
            final_time = (time.time() - start_time)
            #print("--- %s seconds ---" % final_time)
            #print("Done")
            return (reconstruct_path(prev, u), dist.get(end))
        if u in closed:
                continue
        #opened.remove(current_node)
        closed.add(u)
        
        for neighbour in adjacency_list.get(u):
            
            if neighbour in closed:
                continue
            #if neighbour not in opened:
                #heappush(opened,(alt, i))
            
            coords_1 = (nodes.get(u).lat, nodes.get(u).lon)
            coords_2 = (nodes.get(neighbour).lat, nodes.get(neighbour).lon)
            tentative_dist = dist.get(u) + geo.vincenty(coords_1, coords_2).km # g
            if tentative_dist >= dist.get(neighbour):
                continue
            start_point = coords_2
            end_point = (nodes.get(end).lat, nodes.get(end).lon)
            fscore = tentative_dist + heuristic_cost_estimate(start_point, end_point, metric)
            heappush(opened,(fscore, neighbour))
            prev.update({neighbour: u})
            dist.update({neighbour: tentative_dist})
    
    
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


# ## Поиск кратчайшей дороги до больницы

# In[19]:


def find_ways_to_hospital(nodes, adjacency_list, hospital_nodes, start):
    # ----------------------- Инициализация -----------------------
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
    
    # ----------------------- Dijkstra algorithm -----------------------
    print("\nImplementing Dijkstra algorithm...")
    start_time = time.time()
    
    d, p = dijkstra(adjacency_list, start)
    final_time = (time.time() - start_time)
    d_res.append(final_time)
    print("--- %s seconds ---" % final_time)
    
    for node in hospital_nodes:
        found_ways.update({node: (reconstruct_path(p, node), d.get(node))})
    #final_time = (time.time() - start_time)
    #print("--- %s seconds ---" % final_time)
    
    dijkstra_ways = found_ways
    dijkstra_shortest = (found_ways.get(min(found_ways, key = get_key))[0],
                         found_ways.get(min(found_ways, key = get_key))[1])
    
    # записываем только пути от Дейкстры потому что в остальных они почти всегда совпадают!
    write_into_csv_shortest_paths(found_ways)
    
    found_ways.clear()
    
    # ----------------------- Levit algorithm -----------------------
    print("\nImplementing Levit algorithm...")
    start_time = time.time()
    
    d, p = levit(adjacency_list, hospital_nodes, start)
    final_time = (time.time() - start_time)
    l_res.append(final_time)
    print("--- %s seconds ---" % final_time)
    
    for node in hospital_nodes:
        found_ways.update({node: (reconstruct_path(p, node), d.get(node))})
        
    levit_ways = found_ways
    levit_shortest = (found_ways.get(min(found_ways, key = get_key))[0],
                         found_ways.get(min(found_ways, key = get_key))[1])
    
    found_ways.clear()
    
    # ----------------------- A Star algorithm (Euclid) -----------------------
    print("\nImplementing A* algorithm (Euclid)...")
    start_time = time.time()
    for node in hospital_nodes:
        found_ways.update({node: a_star(adjacency_list, nodes, start, node, 'eucl')})
    final_time = (time.time() - start_time)
    ae_res.append(final_time)
    print("--- %s seconds ---" % final_time)
            
    a_star_ways_e = found_ways
    a_star_shortest_e = (a_star_ways_e.get(min(a_star_ways_e, key = get_key))[0],
                       a_star_ways_e.get(min(a_star_ways_e, key = get_key))[1])
    
    # ----------------------- A Star algorithm (Manh) -----------------------
    print("\nImplementing A* algorithm (Manh)...")
    start_time = time.time()
    for node in hospital_nodes:
        found_ways.update({node: a_star(adjacency_list, nodes, start, node, 'manh')})
    final_time = (time.time() - start_time)
    am_res.append(final_time)
    print("--- %s seconds ---" % final_time)
            
    a_star_ways_m = found_ways
    a_star_shortest_m = (a_star_ways_m.get(min(a_star_ways_m, key = get_key))[0],
                       a_star_ways_m.get(min(a_star_ways_m, key = get_key))[1])
    
    # ----------------------- A Star algorithm (Cheb) -----------------------
    print("\nImplementing A* algorithm (Cheb)...")
    start_time = time.time()
    for node in hospital_nodes:
        found_ways.update({node: a_star(adjacency_list, nodes, start, node, 'cheb')})
    final_time = (time.time() - start_time)
    ac_res.append(final_time)
    print("--- %s seconds ---" % final_time)
            
    a_star_ways_c = found_ways
    a_star_shortest_c = (a_star_ways_c.get(min(a_star_ways_c, key = get_key))[0],
                       a_star_ways_c.get(min(a_star_ways_c, key = get_key))[1])
    
    final_time = (time.time() - main_time)
    
    print("\n--- %s seconds ---" % final_time)
    #print("Done")
    return dijkstra_shortest, levit_shortest, a_star_shortest_e, a_star_shortest_m, a_star_shortest_c, dijkstra_ways, levit_ways, a_star_ways_e,a_star_ways_m, a_star_ways_c


# ## Нахождение подъездов к больницам

# In[20]:


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


# ## Интерфейс (типа) 

# In[21]:


def find_way_from_user(nodes, ways, bounds, reachable_nodes, adjacency_list, hospital_nodes):
    # ----------------------- Ввод -----------------------
    
    print("\nEnter your coordinates\n")
    print('Latitude between {0} and {1};\n'.format(bounds[0],bounds[2]))
    print('Longitude between {0} and {1}\n'.format(bounds[1],bounds[3]))
    
    lat = input("Latitude: ")
    lon = input("Longitude: ")
    
    '''# рандомная точка
    lat = r.uniform(float(bounds[0]),float(bounds[2]))
    lon = r.uniform(float(bounds[1]),float(bounds[3]))'''
    
    print ('\nNode: (',lat,lon,')\n');
    
    try:
        if not(float(lat) > float(bounds[0]) and float(lat) < float(bounds[2]) and 
               float(lon) > float(bounds[1]) and float(lon) < float(bounds[3])):
            print("Out of bounds")
            return False
    except Exception:
        print("Error in coordinates")
        return False
    # ----------------------- Инициализация -----------------------
    print('Looking for the closest reachable road...');
    start = find_closest_node(nodes,reachable_nodes,(lat,lon))
    
    # ----------------------- Поиск путей -----------------------

    a,b,c_e,c_m,c_c, a_ways, b_ways, c_ways_e,c_ways_m,c_ways_c = find_ways_to_hospital(nodes, adjacency_list, hospital_nodes, start)
    
    # ----------------------- Графический вывод -----------------------
    # Dijkstra
    print('\n------------- Found with Dijkstra -------------')
    found_ways = a_ways
    shortest_way = a
    
    calc_dist(shortest_way)
    draw_ways(found_ways,shortest_way,url='pictures/test_dij.svg')
    
    print('\n------------- Found with Levit -------------')
    found_ways = b_ways
    shortest_way = b
    
    calc_dist(shortest_way)
    draw_ways(found_ways,shortest_way,url='pictures/test_levit.svg')
    
    print('\n------------- Found with A* (Euclid) -------------')
    found_ways = c_ways_e
    shortest_way = c_e
    
    calc_dist(shortest_way)
    draw_ways(found_ways,shortest_way,url='pictures/test_a_star_e.svg')
    
    print('\n------------- Found with A* (Manh) -------------')
    found_ways = c_ways_m
    shortest_way = c_m
    
    calc_dist(shortest_way)
    draw_ways(found_ways,shortest_way,url='pictures/test_a_star_m.svg')
    
    print('\n------------- Found with A* (Cheb) -------------')
    found_ways = c_ways_c
    shortest_way = c_c
    
    calc_dist(shortest_way)
    draw_ways(found_ways,shortest_way,url='pictures/test_a_star_c.svg')
    
    # ----------------------- Условие завершения -----------------------
    while True:
        answer = input('\nquit? Y/N \n')
        if answer == 'Y':
            return True
        elif answer == 'N':
            return False
        
def calc_dist(shortest_way):
    print('\nDistance to the hospital: {} km.'
          '\nApproximate time it will take you to get there: {} minutes.'
          .format(round(shortest_way[1], 2), 
            round((float(shortest_way[1])/40.0)*60.0)))
def draw_ways(found_ways, shortest_way,url):
    ways_to_draw = []
    for way in found_ways:
        ways_to_draw.append(found_ways.get(way)[0])
    draw_map(nodes, ways, bounds, url, ways_to_draw, shortest_way[0])


# ## Execute

# In[22]:


def main():
    
    '''nodes, ways, bounds = parse_osm()
    nodes, ways, h_coord = delete_transitional_nodes(nodes, ways)
    
    draw_map(nodes, ways, bounds)
    
    adj_list, adj_matrix = form_adj(nodes, ways)
    # calculating distances between nodes in km
    v_set = vertex_set(adj_list, nodes)
    reachable_nodes = DFS(v_set, '532159053')
    
    # for hospital nodes that are not connected to road
    # works too slow
    
    #h_nodes = find_closest_nodes(h_coord, nodes, reachable_nodes)
    
    f = open('docs/hospital_nodes.txt', 'r')
    for node in f:
        h_nodes = [line.strip() for line in f]
    f.close()
    
    quit = False
    
    while not quit:
        quit = work_name(nodes, ways, bounds, reachable_nodes, adj_list, h_nodes, v_set)
     
    #write_into_csv(adj_list, adj_matrix)
    print("\nAll done")'''


# ## Задание 3

# In[58]:


def TSP(h_nodes, bounds, start = '532159053'):
    
    '''# рандомная точка
    lat = r.uniform(float(bounds[0]),float(bounds[2]))
    lon = r.uniform(float(bounds[1]),float(bounds[3]))'''
    
     # ----------------------- Ввод -----------------------
    
    print("\nEnter your coordinates\n")
    print('Latitude between {0} and {1};\n'.format(bounds[0],bounds[2]))
    print('Longitude between {0} and {1}\n'.format(bounds[1],bounds[3]))
    
    lat = input("Latitude: ")
    lon = input("Longitude: ")
    
    '''# рандомная точка
    lat = r.uniform(float(bounds[0]),float(bounds[2]))
    lon = r.uniform(float(bounds[1]),float(bounds[3]))'''
    
    print ('\nNode: (',lat,lon,')\n');
    
    try:
        if not(float(lat) > float(bounds[0]) and float(lat) < float(bounds[2]) and 
               float(lon) > float(bounds[1]) and float(lon) < float(bounds[3])):
            print("Out of bounds")
            return False
    except Exception:
        print("Error in coordinates")
        return False
    
    # ----------------------- Инициализация -----------------------
    print('Looking for the closest reachable road...');
    start = find_closest_node(nodes,reachable_nodes,(lat,lon))
    
    node_list = h_nodes.copy()
    node_list.append(start)
    paths, m = create_matrix(node_list)
    #print(node_list)
    
    # ----------------------- TSP_nn -----------------------
    print("\nTSP (Nearest Neighbour)")
    start_time = time.time()
    
    route_nn, route_nn_len = TSP_nn(node_list, paths, m, start)
    draw_map(nodes, ways, bounds, 'pictures/task3test_nn.svg', route_nn, [], True)
    write_into_csv_tsp_routes(route_nn)
    
    final_time = time.time() - start_time
    print("\n--- %s seconds ---" % final_time)
    
    
    # ----------------------- TSP_greedy -----------------------
    '''print("\nTSP (Greedy)")
    start_time = time.time()
    
    route_greedy, route_greedy_len = TSP_greedy(node_list, paths, m, start)
    draw_map(nodes, ways, bounds, 'pictures/task3test_nn.svg', route_greedy, [], True)
    
    final_time = time.time() - start_time
    print("\n--- %s seconds ---" % final_time)
    
    return round(route_nn_len, 2) - round(route_greedy_len, 2)'''
        
def TSP_nn(node_list, paths, m, start):
    
    opened = set()
    for i in range(0,11):
        opened.add(i)
    
    i = r.randrange(11)
    first = i
    route_ids = [i]
    
    while opened:
        opened.remove(i)
        #print(i)
        
        min_array = {}
        
        for j in range(0,11):
            if j in opened and j != i and j != first:
                min_array.update({j: m[i][j]})
        
        if len(min_array)>0 :
            minimum = min(min_array, key = min_array.get)
            route_ids.append(minimum)
        i = minimum
    
    # формируем дорогу
    before_start = []
    new_route_ids = []
    for i in range(0,11):
        if route_ids[i] == 10:
            for j in range(i, 11):
                new_route_ids.append(route_ids[j])
            break
        else:
            before_start.append(route_ids[i])
    for j in range(0, len(before_start)):
        new_route_ids.append(route_ids[j])
    route_ids = new_route_ids
    
    tsp_route = []
    tsp_route_len = 0
    for i in range(0,10):
        tsp_route.append(paths[route_ids[i]][route_ids[i+1]])
        tsp_route_len = tsp_route_len + m[route_ids[i]][route_ids[i+1]]
    tsp_route.append(paths[route_ids[len(node_list)-1]][route_ids[0]])
    tsp_route_len = tsp_route_len + m[route_ids[len(node_list)-1]][route_ids[0]]
    
    print("\nDistance of the route: {} km.".format(round(tsp_route_len, 2)))
    
    #draw_map(nodes, ways, bounds, 'pictures/task3test_nn.svg', tsp_route)
    
    return tsp_route, tsp_route_len
    
def TSP_greedy(node_list, paths, m, start):
    
    edges = []
    ignored_edges = []
    route_ids = []
    
    for i in range(0,11):
        for j in range(i+1,11):
            heappush(edges, (m[i][j], (i,j)))
    
    opened = set()
    closed = set()
    
    current = heappop(edges)
    opened.add(current[1][0])
    opened.add(current[1][1])
    
    route_ids.append((current[1][0],current[1][1]))
    
    while opened:
        if len(edges) == 0:
            #print('OUT')
            edges = ignored_edges.copy()
            ignored_edges.clear()
        current = heappop(edges)
        
        print(current[1])
        
        if current[1][0] in closed or current[1][1] in closed:
            continue
        if check_cycle(route_ids, current):
            continue
        
        '''if current[1][0] in opened and current[1][1] in opened:
            heappush(ignored_edges, current)
            continue'''
        
        '''if current[1][0] == 10 and 10 in opened:
            closed.add(10)
            opened.remove(10)
            continue
        if current[1][1] == 10 and 10 in opened:
            closed.add(10)
            opened.remove(10)
            continue'''
        
        route_ids.append((current[1][0],current[1][1]))
        print(current)
        
        if current[1][0] in opened:
            opened.remove(current[1][0])
            closed.add(current[1][0])
        else:
            opened.add(current[1][0])
        if current[1][1] in opened:
            opened.remove(current[1][1])
            closed.add(current[1][1])
        else:
            opened.add(current[1][1])
    # формируем дорогу
    new_route_ids = []
    node_for_search = 10
    N = 11
    
    print(route_ids)
    
    while route_ids:
        for elem in route_ids:
            if elem[0] == node_for_search:
                print(1,node_for_search)
                new_route_ids.append(elem)
                node_for_search = elem[1]
                route_ids.remove(elem)
                print(route_ids)
                break
            if elem[1] == node_for_search:
                print(2,node_for_search)
                new_route_ids.append(elem)
                node_for_search = elem[0]
                route_ids.remove(elem)
                print(route_ids)
                break
        
    route_ids = new_route_ids
    print(route_ids)
            
    #print(opened,closed)
        
    tsp_route = []
    tsp_route_len = 0
    for elem in route_ids:
        tsp_route.append(paths[elem[0]][elem[1]])
        tsp_route_len = tsp_route_len + m[elem[0]][elem[1]]
    tsp_route.append(paths[route_ids[10][route_ids[0]]])
    tsp_route_len = tsp_route_len + m[route_ids[10][route_ids[0]]]
            
    print("\nDistance of the route: {} km.".format(round(tsp_route_len, 2)))
    
    #draw_map(nodes, ways, bounds, 'pictures/task3test_greedy.svg', tsp_route)
    
    return tsp_route, tsp_route_len
    
def check_cycle(route, ttuple):
    print('\nCHECKING\n')
    print(route)
    print(ttuple)
    addition = ttuple[1]
    visited = {}
    
    for i in range(0,11):
        visited.update({i:False})
        
    visited.update({addition[0]:True})
    
    current_id_1 = addition[0]
    current_id_2 = addition[1]
    
    for counter in range(0, len(route)+1):
        print(visited)
        for i in range(0,len(route)):
            for elem in route:
                if elem[0] == current_id_1:
                    if not visited.get(elem[1]):
                        visited.update({elem[1]:True})
                        current_id_1 = elem[1]
                    else:
                        return True
                elif elem[1] == current_id_1:
                    if not visited.get(elem[0]):
                        visited.update({elem[0]:True})
                        current_id_1 = elem[0]
                    else:
                        return True
            for elem in route:
                if elem[0] == current_id_2:
                    if not visited.get(elem[1]):
                        visited.update({elem[1]:True})
                        current_id_2 = elem[1]
                    else:
                        return True
                elif elem[1] == current_id_2:
                    if not visited.get(elem[0]):
                        visited.update({elem[0]:True})
                        current_id_2 = elem[0]
                    else:
                        return True
    return False
    
def create_matrix(node_list):
    
    print("\nCreating adjacency matrix for the nodes...")
    
    M = np.zeros([len(node_list),len(node_list)])
    paths = np.zeros([len(node_list),len(node_list)],dtype=object)
    
    for i in range(0, 11):
        for j in range (i+1,11):
            #print(node_list[i],node_list[j])
            path, dist = a_star(adjacency_list, nodes, node_list[i], node_list[j], 'eucl')
            #print(dist)
            M[i][j] = dist
            M[j][i] = dist
            paths[i][j] = path
            paths[j][i] = path
    return paths, M


# ### Средняя разница между длинами пути в Nearest Neighbour и Greedy составила -1.07км
# 
# * Greedy почему-то уступает Nearest Neighbour не смотря на статистику?

# ## Задание 1 (execution)

# In[74]:


# инициализация данных

nodes, ways, bounds = parse_osm()
nodes, ways = delete_transitional_nodes(nodes, ways)
    
draw_map(nodes, ways, bounds)
    
adjacency_list, adj_list, adj_matrix = form_adj(nodes, ways)
    
# calculating distances between nodes in km
reachable_nodes = DFS(adjacency_list, '532159053')
    
# for hospital nodes that are not connected to road
#h_nodes = find_closest_nodes(h_coord, nodes, reachable_nodes)
    
f = open('docs/hospital_nodes.txt', 'r')
h_nodes = [line.strip() for line in f]
f.close()

#write_into_csv(adj_list, adj_matrix)


# ## Задание 2 (execution)

# In[ ]:


'''# инициализация данных

nodes, ways, bounds = parse_osm()
nodes, ways = delete_transitional_nodes(nodes, ways)
    
#draw_map(nodes, ways, bounds)
    
adjacency_list, adj_list, adj_matrix = form_adj(nodes, ways)
    
# calculating distances between nodes in km
#v_set = vertex_set(adj_list, nodes)
reachable_nodes = DFS(adjacency_list, '532159053')
    
# for hospital nodes that are not connected to road
#h_nodes = find_closest_nodes(h_coord, nodes, reachable_nodes)
    
f = open('docs/hospital_nodes.txt', 'r')
h_nodes = [line.strip() for line in f]
f.close()'''


# In[80]:


quit = False

d_res = []
l_res = []
ae_res = []
am_res = []
ac_res = []
    
i = 0
quit = False
while not quit:
    #i = i + 1
    quit = find_way_from_user(nodes, ways, bounds, reachable_nodes, adjacency_list, h_nodes)
    #find_way_from_user(nodes, ways, bounds, reachable_nodes, adjacency_list, h_nodes)
    #if i == 100:
        #quit = True


# In[57]:


# оценка времени в среднем

'''d_average = 0
l_average = 0
ae_average = 0
am_average = 0
ac_average = 0

for i in range(1,10):
    d_average = sum(d_res)/len(d_res)
    l_average = sum(l_res)/len(l_res)
    ae_average = sum(ae_res)/len(ae_res)
    am_average = sum(am_res)/len(am_res)
    ac_average = sum(ac_res)/len(ac_res)
print(d_average,l_average,ae_average,am_average,ac_average)'''


# ### Среднее время выполнения:
# * Дейкстра: 1.3164с
# * Левит: 1.47с
# * А* (евкл.): 2.91с
# * А* (манхэт.): 2.985с
# * А* (чеб.): 2.97с
# 
# разные эвристические функции иногда дают разные результаты (~0.1-1.5км)

# ## Задание 3 (execution)

# In[ ]:


'''# инициализация данных

nodes, ways, bounds = parse_osm()
nodes, ways = delete_transitional_nodes(nodes, ways)
    
#draw_map(nodes, ways, bounds)
    
adjacency_list, adj_list, adj_matrix = form_adj(nodes, ways)
    
# calculating distances between nodes in km
#v_set = vertex_set(adj_list, nodes)
reachable_nodes = DFS(adjacency_list, '532159053')
    
# for hospital nodes that are not connected to road
#h_nodes = find_closest_nodes(h_coord, nodes, reachable_nodes)
    
f = open('docs/hospital_nodes.txt', 'r')
h_nodes = [line.strip() for line in f]
f.close()'''


# In[62]:


TSP(h_nodes, bounds)


# In[54]:


# для оценивания работы двух алгоритмов

'''difference = []
for i in range (0,10):
    print("\n!!!!!!!!!!!!!!!!!!!!!! Iteration %s !!!!!!!!!!!!!!!!!!!!!!!!!!\n"%i)
    difference.append(TSP(h_nodes, bounds, '532159053'))
print("Average difference in distance betweeb NN and Greedy:",sum(difference)/len(difference))

TSP(h_nodes, bounds)'''

