from dataclasses import dataclass
from typing      import Any, Iterable


class DirectedEdge():
    def __init__(self, parent, child) -> None:
        self.parent = parent
        self.child  = child

    def eval(self, graph, current_tf : KVArray) -> KVArray:
        pass


@dataclass
class Frame:
    name : str


class FrameView():
    """Presents a view of a frame with changed reference and transform."""
    def __init__(self, frame : Frame, reference : str, transform : KVArray) -> None:
        self._frame = frame
        self.reference = reference
        self.transform = transform
    
    def __getattribute__(self, __name: str) -> Any:
        if __name == self.reference:
            return self.reference
        elif __name == self.transform:
            return self.transform
        return getattr(self._frame, __name)
    
    @property
    def dtype(self):
        return type(self._frame)


class Graph():
    def __init__(self) -> None:
        self._nodes = {}
        self._incoming_edges = {}
        
        self._nodes['world'] = Frame('world')

    def add_frame(self, frame : Frame):
        self._nodes[frame.name] = frame
    
    def get_fk(self, target_frame : str, source_frame : str = 'world'):
        if target_frame not in self._nodes:
            raise KeyError(f'Target frame "{target_frame}" is not known.')
        
        if source_frame not in self._nodes:
            raise KeyError(f'Source frame "{source_frame}" is not known.')

        if target_frame == source_frame:
            return FrameView(self._nodes[target_frame], source_frame, KVArray.eye(49))

        p_target = self._get_path(target_frame, source_frame)
        p_source = self._get_path(source_frame, target_frame)

        # Cases
        # t == s -> Identity transform
        # t != s and both are roots -> Exception
        # t != s and both belong to different roots -> Exception
        # t != s and 

        if len(p_target) == 0 and len(p_source) == 0:


            if p_target[-1].parent == source_frame:
                s_T_t = self._gen_tf(p_target)
            elif len(p_source) > 0 and p_target[-1].parent == p_target[-1].parent:
                r_T_t = self._gen_tf(p_target)
                s_T_r = gm.inverse(self._gen_tf(p_source))
                s_T_t = s_T_r.dot(r_T_t)
            



    def _gen_tf(self, chain : Iterable[DirectedEdge]) -> KVArray:
        tf = KVArray.eye(4)
        for e in chain:
            tf = e.eval(self, tf)
        return tf

    def _get_path(self, start : str, end : str):
        out = []
        current = start
        
        while current != end and current in self._incoming_edges:
            e = self._incoming_edges
            out.append(e)
            current = e.parent

        return out

