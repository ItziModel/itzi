from __future__ import division
import os, sys
import swmm_error
import numpy as np

import swmm
import swmm_c


INP = '/home/laurent/Datos_geo/kolkata/drainage/100_FE_network only2.inp'


def test_python(swmm5, dict_nodes, dict_links):
    swmm_network = swmm.SwmmNetwork(swmm5, dict_nodes, dict_links)
    swmm_network.update()
    #~ print(swmm_network.links['link_type'])
    for k, l in swmm_network.links.iteritems():
        print(k, l)
    for k, l in swmm_network.nodes.iteritems():
        print(k, l)


def main():
    # load INP
    swmm5 = swmm.Swmm5()
    swmm5.swmm_open(input_file=INP,
                         report_file=os.devnull,
                         output_file='')
    swmm5.swmm_start()

    # parse swmm network
    swmm_parser = swmm.SwmmInputParser(INP)
    dict_links = swmm_parser.get_links_id_as_dict()
    dict_nodes = swmm_parser.get_nodes_id_as_dict()

    test_python(swmm5, dict_nodes, dict_links)

    swmm5.swmm_end()
    swmm5.swmm_close()



if __name__ == "__main__":
    sys.exit(main())

