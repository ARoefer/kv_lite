from dataclasses import dataclass
from typing      import Any, \
                        Iterable, \
                        List, \
                        Union

from . import spatial as gm


class DirectedEdge():
    def __init__(self, parent, child) -> None:
        self.parent = parent
        self.child  = child

    def eval(self, graph, current_tf : gm.KVArray) -> gm.KVArray:
        pass


@dataclass
class Frame:
    name : str


class FrameView():
    """Presents a view of a frame with changed reference and transform."""
    def __init__(self, frame : Frame, reference : str, transform : gm.KVArray) -> None:
        self._frame = frame
        self.reference = reference
        self.transform = transform
    
    @property
    def name(self) -> str:
        return self._frame.name
    
    @property
    def frame(self) -> Frame:
        return self._frame
    
    @property
    def dtype(self) -> type:
        return type(self._frame)

    def __str__(self):
        return f'T ({self.name} -> {self.reference}):\n{self.transform}'

class FKChainException(Exception):
    pass


class Graph():
    def __init__(self) -> None:
        self._nodes = {}
        self._incoming_edges  = {}
        self._named_edges     = {}
        self._inv_named_edges = {}

        self._nodes['world'] = Frame('world')

    def add_frame(self, frame : Frame):
        self._nodes[frame.name] = frame

    def remove_frame(self, frame : Union[Frame, str]):
        name = frame.name if isinstance(frame, Frame) else frame

        if name not in self._nodes:
            raise KeyError(f'Frame "{name}" is unknown')
        
        if name in self._incoming_edges:
            del self._incoming_edges[name]
        
        del self._nodes[name]

    def get_frames(self) -> List[str]:
        return list(self._nodes.keys())

    def get_frame(self, name : str) -> Frame:
        if name not in self._nodes:
            raise KeyError(f'Unknown frame "{name}"')
        return self._nodes[name]
    
    def has_frame(self, name : str) -> bool:
        return name in self._nodes

    def get_fk(self, target_frame : str, source_frame : str = 'world'):
        if target_frame not in self._nodes:
            raise KeyError(f'Target frame "{target_frame}" is not known.')
        
        if source_frame not in self._nodes:
            raise KeyError(f'Source frame "{source_frame}" is not known.')

        # Chain to self
        if target_frame == source_frame:
            return FrameView(self._nodes[target_frame], source_frame, gm.eye(4))

        p_target = self._get_path(target_frame, source_frame)
        p_source = self._get_path(source_frame, target_frame)

        # Cases
        # t == s -> Identity transform
        # t != s and both are roots -> Exception
        # t != s and both belong to different roots -> Exception
        # t != s and 

        if len(p_target) == 0 and len(p_source) == 0:
            raise FKChainException(f'Cannot look up {source_frame} T {target_frame}: Both are distinct roots')

        # Target is root and source is its child
        if len(p_target) == 0:
            if p_source[-1].parent == target_frame:
                s_T_t = gm.Transform.inverse(self._gen_tf(p_source))
                return FrameView(self._nodes[target_frame], source_frame, s_T_t)
            elif p_source[-1].parent != target_frame:
                raise FKChainException(f'Cannot look up {source_frame} T {target_frame}: {target_frame} is a root and {source_frame} is not in its subtree')

        # Source is root and target is its child
        if len(p_source) == 0: 
            if p_target[-1].parent == source_frame:
                s_T_t = self._gen_tf(p_target)
                return FrameView(self._nodes[target_frame], source_frame, s_T_t)
            elif p_target[-1].parent != source_frame:
                raise FKChainException(f'Cannot look up {source_frame} T {target_frame}: {source_frame} is a root and {target_frame} is not in its subtree')

        # Frames share a root
        if p_target[-1].parent == p_source[-1].parent:
            r_T_t = self._gen_tf(p_target)
            s_T_r = gm.Transform.inverse(self._gen_tf(p_source))
            s_T_t = s_T_r.dot(r_T_t)
            return FrameView(self._nodes[target_frame], source_frame, s_T_t)

        # Target is child of source 
        if p_target[-1].parent == source_frame:
            s_T_t = self._gen_tf(p_target)
            return FrameView(self._nodes[target_frame], source_frame, s_T_t)

        # Source is child of target 
        if p_source[-1].parent == target_frame:
            s_T_t = gm.Transform.inverse(self._gen_tf(p_source))
            return FrameView(self._nodes[target_frame], source_frame, s_T_t)
        
        raise FKChainException(f'Cannot look up {source_frame} T {target_frame}: The frames have different roots {p_target[-1].parent} and {p_source[-1].parent}')
    
    def add_edge(self, edge : DirectedEdge, name : str=None):
        if edge.parent not in self._nodes:
            raise KeyError(f'Cannot insert edge as {edge.parent} is not a node in the graph')

        if edge.child not in self._nodes:
            raise KeyError(f'Cannot insert edge as {edge.child} is not a node in the graph')

        if name is not None and name in self._named_edges:
            raise KeyError(f'Cannot insert named edge as "{name}", as name is already taken.')

        old_edge = self._incoming_edges[edge.child] if edge.child in self._incoming_edges else None
        self._incoming_edges[edge.child] = edge

        # Check for circular dependencies
        path = self._get_path(edge.parent, edge.child)
        if len(path) > 0 and path[-1].parent == edge.child:
            if old_edge is not None:
                self._incoming_edges[edge.child] = old_edge
            else:
                del self._incoming_edges[edge.child]
            raise RuntimeError(f'Adding edge {edge.parent} -> {edge.child} introduces a circle: '' <- '.join([e.child for e in  path]))

        if name is not None:
            self._named_edges[name] = edge
            self._inv_named_edges[id(edge)] = name

    def remove_edge(self, edge : Union[DirectedEdge, tuple]):
        if isinstance(edge, DirectedEdge):
            parent, child = edge.parent, edge.child
        else:
            parent, child = edge
        
        if child not in self._incoming_edges:
            raise KeyError(f'Node {child} does not have an incoming edge.')
            
        if parent != self._incoming_edges[child].parent:
            raise KeyError(f'Edge of {child} is {self._incoming_edges[child].parent} -> {child}, not {parent} -> {child}.')

        if id(edge) in self._inv_named_edges:
            name = self._inv_named_edges[id(edge)]
            del self._inv_named_edges[id(edge)]
            del self._named_edges[name]

        del self._incoming_edges[child]

    def get_edge(self, name : str) -> DirectedEdge:
        if name not in self._named_edges:
            raise KeyError(f'Edge "{name}" is not in graph.')
        return self._named_edges[name]

    def get_edges(self) -> List[DirectedEdge]:
        return list(self._incoming_edges.values())

    def get_incoming_edge(self, node_name : str) -> DirectedEdge:
        if node_name not in self._nodes:
            raise KeyError(f'Unknown frame "{node_name}".')
        
        return self._incoming_edges[node_name] if node_name in self._incoming_edges else None

    def _gen_tf(self, chain : Iterable[DirectedEdge]) -> gm.KVArray:
        tf = gm.Transform.identity()
        for e in chain:
            tf = e.eval(self, tf)
        return tf

    def _get_path(self, start : str, end : str):
        out = []
        current = start
        
        while current != end and current in self._incoming_edges:
            e = self._incoming_edges[current]
            out.append(e)
            current = e.parent

        return out

