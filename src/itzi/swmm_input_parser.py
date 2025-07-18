# coding=utf8
"""
Copyright (C) 2016-2025 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

import os
from collections import namedtuple


class SwmmInputParser(object):
    """A parser for swmm input text file"""

    # list of sections keywords
    sections_kwd = [
        "title",  # project title
        "option",  # analysis options
        "junction",  # junction node information
        "outfall",  # outfall node information
        "divider",  # flow divider node information
        "storage",  # storage node information
        "conduit",  # conduit link information
        "pump",  # pump link
        "orifice",  # orifice link
        "weir",  # weir link
        "outlet",  # outlet link
        "xsection",  # conduit, orifice, and weir cross-section geometry
        "coordinate",  # coordinates of drainage system nodes
        "vertice",  # coordinates of interior vertex points of links
    ]
    link_types = ["conduit", "pump", "orifice", "weir", "outlet"]
    # define object containers
    junction_values = ["x", "y", "elev", "ymax", "y0", "ysur", "apond"]
    Junction = namedtuple("Junction", junction_values)
    Link = namedtuple("Link", ["in_node", "out_node", "vertices"])
    # coordinates container
    Coordinates = namedtuple("Coordinates", ["x", "y"])

    def __init__(self, input_file):
        # read and parse the input file
        assert os.path.isfile(input_file)
        self.inp = dict.fromkeys(self.sections_kwd)
        self.read_inp(input_file)

    def section_kwd(self, sect_name):
        """verify if the given section name is a valid one.
        Return the corresponding section keyword, None if unknown
        """
        # check done in lowercase, without final 's'
        section_valid = sect_name.lower().rstrip("s")
        result = None
        for kwd in self.sections_kwd:
            if kwd.startswith(section_valid):
                result = kwd
        return result

    def read_inp(self, input_file):
        """Read the inp file and generate a dictionary of lists"""
        current_section = None
        with open(input_file, "r") as inp:
            for line in inp:
                # got directly to next line if comment or empty
                if line.startswith(";") or not line.strip():
                    continue
                # retrive current standard section name
                elif line.startswith("["):
                    current_section = self.section_kwd(line.strip().strip("[] "))
                elif current_section is None:
                    continue
                else:
                    if self.inp[current_section] is None:
                        self.inp[current_section] = []
                    self.inp[current_section].append(line.strip().split())

    def get_juntions_ids(self):
        """return a list of junctions ids (~name)"""
        return [j[0] for j in self.inp["junction"]]

    def get_juntions_as_dict(self):
        """return a dict of namedtuples"""
        d = {}
        values = []
        for coor in self.inp["coordinate"]:
            for j in self.inp["junction"]:
                name = j[0]
                if coor[0] == name:
                    j_val = [float(v) for v in j[1:]]
                    values = [float(coor[1]), float(coor[2])] + j_val
                    d[name] = self.Junction._make(values)
        return d

    def get_nodes_id_as_dict(self):
        """return a dict of id:coordinates"""
        # sections to search
        node_types = ["junction", "outfall", "divider", "storage"]
        # a list of all nodes id
        nodes = []
        for n_t in node_types:
            # add id only if there are values in the dict entry
            if self.inp[n_t]:
                for line in self.inp[n_t]:
                    nodes.append(line[0])

        # A coordinates dict
        coords_dict = {}
        if self.inp["coordinate"]:
            for coords in self.inp["coordinate"]:
                coords_dict[coords[0]] = self.Coordinates(float(coords[1]), float(coords[2]))
        # fill the dict
        node_dict = {}
        for node_id in nodes:
            if node_id in coords_dict:
                node_dict[node_id] = coords_dict[node_id]
            else:
                node_dict[node_id] = None
        return node_dict

    def get_links_id_as_dict(self):
        """return a list of id:Link"""
        links_dict = {}
        # loop through all types of links
        for k in self.link_types:
            links = self.inp[k]
            if links is not None:
                for ln in links:
                    ID = ln[0]
                    vertices = self.get_vertices(ID)
                    # names of link, inlet and outlet nodes
                    links_dict[ID] = self.Link(in_node=ln[1], out_node=ln[2], vertices=vertices)
        return links_dict

    def get_vertices(self, link_name):
        """For a given link name, return a list of Coordinates objects"""
        vertices = []
        if isinstance(self.inp["vertice"], list):
            for vertex in self.inp["vertice"]:
                if link_name == vertex[0]:
                    vertex_c = self.Coordinates(float(vertex[1]), float(vertex[2]))
                    vertices.append(vertex_c)
        return vertices
