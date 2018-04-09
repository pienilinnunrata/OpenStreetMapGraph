# OpenStreetMapGraph

Задание по Теории конечных графов и ее приложения (3 курс, СПбГУ)

**ФИО студентки: *Барсукова Юлия***

**Группа: *331***

**Город: *Калининград***

## Инструкция

Проект написан на Python 3.6 в Jupyter Notebook

Реализовано:
* Парсинг выгруженного с OpenStreetMap .osm файла;
* [Визуализация графа в .svg с узлами (end_node и crossroad_node) и наиболее важными дорогами](https://github.com/pienilinnunrata/OpenStreetMapGraph/blob/master/pictures/kaliningrad_map.svg);
* Вывод матрицы смежности в .csv (слишком много весит и посему не была загружена в репозиторий);
* [Вывод списка смежности в .csv](https://github.com/pienilinnunrata/OpenStreetMapGraph/blob/master/csv/adjacency_list.csv). 

Использованные средства языка: 
* lxml библиотека для парсинга .osm файла;
* svgwrite библиотека для отрисовки карты в .svg;
* pandas библиотека для записи в файлы .csv.

## TODO:
* Реализовать алгоритмы поиска наикратчайшего пути.

Как сказал Уильям Сомерсет Моэм:

>Сколько бы мы это не отрицали, но в глубине души мы знаем: всё, что с нами случилось, мы заслужили.

![Kaliningrad Map](https://github.com/pienilinnunrata/OpenStreetMapGraph/blob/master/docs/kaliningrad_map.png)
