# coding=utf8
"""
Copyright (C) 2016  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""


from __future__ import division
import pandas as pd
import dateutil.parser
import matplotlib.pyplot as plt
import messenger as msgr


class ResultsReader(object):
    def __init__(self, results):
        """results is deserialized object
        """
        self.results = results
        self.records = self.results['records']

        # list nodes and node values
        self.nodes_id = self.records[0]['nodes'].keys()
        if self.nodes_id:
            self.node_values = self.records[0]['nodes'][self.nodes_id[0]].keys()
        else:
            self.node_values = []

        # list links and links values
        self.links_id = self.records[0]['links'].keys()
        if self.links_id:
            self.link_values = self.records[0]['links'][self.links_id[0]].keys()
        else:
            self.link_values = []

    def verif_node_id(self, node_id):
        """Verify if the node ID is registered in the file
        """
        if node_id not in self.nodes_id:
            msgr.fatal(u"Unknown node ID: '{}'".format(node_id))
        return self

    def verif_node_value(self, value):
        """
        """
        if value not in self.node_values:
            msgr.fatal(u"Unknown node value: '{}'".format(value))
        return self

    def _get_node_records(self, node_id):
        """return a pandas dataframe of records for one node
        """
        recs = [r['nodes'][node_id] for r in self.records]
        dates = [self.format_date(r['date']) for r in self.records]
        return pd.DataFrame(recs, index=dates)

    def _get_node_value(self, node_id, value):
        """return a pandas Series of a specified node value
        """
        return self._get_node_records(node_id)[value]

    def plot_node_values(self, node_id, values):
        """plot a list of values ID of a given node
        """
        for v_id in values:
            try:
                plt.plot(self._get_node_value(node_id, v_id), label=v_id)
            except ValueError:
                msgr.warning(_(u"Cannot plot '{}'".format(v_id))
                continue
            plt.ylabel(u"value")
            plt.xlabel(u"elapsed time")
        plt.title(u"Node {}".format(node_id))
        plt.legend()
        plt.show()
        return self

    def node_values_to_csv(self, node_id, file_name):
        """export values of a node to a csv file (or to a string if none is given)
        """
        if file_name:
            self._get_node_records(node_id).to_csv(path_or_buf=file_name)
        else:
            print self._get_node_records(node_id).to_csv()
        return self

    def format_date(self, date):
        """take a string as input
        try to convert to an int, then to datetime
        """
        try:
            return int(date)
        except ValueError:
            pass
        try:
            return dateutil.parser.parse(date)
        except ValueError:
            msgr.fatal(u"Cannot parse date: '{}'".format(date))
