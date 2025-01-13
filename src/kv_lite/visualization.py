# Copyright (c) 2024 Adrian Röfer, Robot Learning Lab, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import re
import networkx as nx

from collections import defaultdict
from itertools   import product
from typing import Dict, \
                   Union
                   

from . import spatial as kv

REGEX = re.compile('bla')

def generate_expression_graph(exprs : Dict[str, kv.KVExpr],
                              sym_colors  : Dict[re.Pattern, str],
                              expr_colors : Union[Dict[re.Pattern, str], str]='orange') -> nx.Graph:
    g = nx.Graph()
    symbols = set()
    edges   = defaultdict(set)

    for name, expr in exprs.items():
        str_symbols = {str(s) for s in expr.symbols}
        edges[name] |= str_symbols
        symbols |= str_symbols

        if isinstance(expr_colors, dict):
            for pattern, color in expr_colors.items():
                if pattern.match(name) is not None:
                    g.add_node(name, color=color)
                    break
            else:
                g.add_node(name, color='orange')
        else:
            g.add_node(name, color=expr_colors)
    
    for str_sym in symbols:
        for pattern, color in sym_colors.items():
            if pattern.match(str_sym) is not None:
                g.add_node(str_sym, color=color)
                break
        else:
            g.add_node(str_sym, color='blue')
    
    for expr_node, syms in edges.items():
        g.add_edges_from(list(product([expr_node], syms)))
    
    return g

def graph_to_html(g : nx.Graph, path,
                  dims=(1800, 1200),
                  show=False,
                  physics=False,
                  notebook=False):
    from pyvis import network

    nt = network.Network(f'{dims[1]}px', f'{dims[0]}px')
    nt.from_nx(g)
    nt.toggle_physics(physics)
    nt.show_buttons(filter_=['physics'])
    if show:
        nt.show(path, notebook=notebook)
    else:
        nt.save_graph(path)
