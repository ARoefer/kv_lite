from dataclasses import dataclass
from pathlib     import Path
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

    @property
    def symbols(self) -> frozenset:
        """Symbols (DoFs) appearing in this edge's transform. Base has none; subclasses
        carrying a transform override this. `get_mass_matrix` relies on it to find the
        active edges for a requested symbol set."""
        return frozenset()


@dataclass
class Frame:
    name : str

    def __post_init__(self):
        self.name = Path(self.name)


class FrameView():
    """Presents a view of a frame with changed reference and transform."""
    def __init__(self, frame : Frame, reference : Path, transform : gm.KVArray) -> None:
        self._frame = frame
        self.reference = reference
        self.transform = transform
    
    @property
    def name(self) -> Path:
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
    def __init__(self, root_node=Path('world')) -> None:
        self._nodes = {}
        self._incoming_edges  = {}
        self._named_edges     = {}
        self._inv_named_edges = {}
        self._root_node       = root_node

        self._nodes[root_node] = Frame(root_node)

    def reset_root(self, name : str):
        if name not in self._nodes:
            raise KeyError(f'Cannot make unknown frame {name} root.')
        self._root_node = name

    @property
    def root_node(self) -> str:
        return self._root_node

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

    def get_frame(self, name : Path) -> Frame:
        if name not in self._nodes:
            raise KeyError(f'Unknown frame "{name}"')
        return self._nodes[name]
    
    def has_frame(self, name : Path) -> bool:
        return name in self._nodes

    def get_fk(self, target_frame : str, source_frame : str = Path('world')):
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
    
    def add_edge(self, edge : DirectedEdge, name : Path=None):
        if edge.parent not in self._nodes:
            raise KeyError(f'Cannot insert edge as {edge.parent} is not a node in the graph')

        if edge.child not in self._nodes:
            raise KeyError(f'Cannot insert edge as {edge.child} is not a node in the graph')

        if edge.child == edge.parent:
            raise ValueError(f'Cannot create edges of a node to itself. Attempted on: {edge.child}')

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

    def has_edge(self, name : Path) -> bool:
        return name in self._named_edges

    def get_edge(self, name : Path) -> DirectedEdge:
        if name not in self._named_edges:
            raise KeyError(f'Edge "{name}" is not in graph.')
        return self._named_edges[name]

    def get_edges(self) -> List[DirectedEdge]:
        return list(self._incoming_edges.values())

    def get_incoming_edge(self, node_name : Path) -> DirectedEdge:
        if node_name not in self._nodes:
            raise KeyError(f'Unknown frame "{node_name}".')
        
        return self._incoming_edges[node_name] if node_name in self._incoming_edges else None

    def _get_children_index(self) -> dict:
        """Builds a parent -> [child, ...] index from the incoming-edge map, on demand.

        Not cached: maintaining an outgoing structure would add invalidation burden to
        add_edge/remove_edge for no measured benefit. Revisit only if profiling shows it.
        """
        children = {}
        for child, edge in self._incoming_edges.items():
            children.setdefault(edge.parent, []).append(child)
        return children

    def get_subtree(self, root : Path) -> List[Path]:
        """All frame names in the subtree rooted at `root`, inclusive of `root`."""
        if root not in self._nodes:
            raise KeyError(f'Unknown frame "{root}"')

        children = self._get_children_index()
        out      = []
        stack    = [root]
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(children.get(current, []))
        return out

    def get_gravity_vector(self, symbols, gravity=(0, 0, -9.81), source_frame : Path = Path('world')) -> gm.KVArray:
        """Generalized gravity forces G(q) as an n-vector, n = len(symbols).

        `G = dU/dq` with the potential energy `U = -sum_i m_i * g . p_i`, where `g` is the
        gravitational acceleration vector (default `(0, 0, -9.81)`) and `p_i` is body `i`'s
        CoM position in `source_frame`. Sign convention matches the manipulator equation
        `M q'' + C q' + G = tau`. Bodies not depending on `symbols` contribute zero.
        """
        g = gm.KVArray([gravity[0], gravity[1], gravity[2]])
        U = 0
        for name in self.get_subtree(source_frame):
            body = self._nodes[name]
            if getattr(body, 'inertial', None) is None:
                continue
            p  = self.get_fk(name, source_frame).transform.dot(body.inertial.origin)[:3, 3]
            U  = U - body.inertial.mass * g.dot(p)
        if not isinstance(U, (gm.KVArray, gm.KVExpr)):   # no bodies in the subtree
            return gm.zeros(len(symbols))
        return gm.KVArray(U.jacobian(symbols)).reshape((len(symbols),))

    def get_coriolis_matrix(self, symbols, velocities, source_frame : Path = Path('world')) -> gm.KVArray:
        """Coriolis/centrifugal matrix C(q, q') as an n x n KVArray, so that `C @ q'` gives
        the Coriolis/centrifugal generalized forces.

        Uses Christoffel symbols of the first kind:
            C[i,j] = sum_k 0.5 * (dM[i,j]/dq_k + dM[i,k]/dq_j - dM[j,k]/dq_i) * q'_k
        `velocities` are the q'_k symbols, positionally matched to `symbols`. Correct but
        symbolically expensive for large systems (an n**3 derivative tensor).
        """
        n = len(symbols)
        if n != len(velocities):
            raise ValueError(f'symbols ({n}) and velocities ({len(velocities)}) must have equal length.')

        M  = self.get_mass_matrix(symbols, source_frame)
        dM = gm.KVArray(M.jacobian(symbols)).reshape((n, n, n))   # dM[i,j,k] = dM_ij / dq_k

        C = gm.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                acc = 0
                for k in range(n):
                    acc = acc + 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i]) * velocities[k]
                C[i, j] = acc
        return C

    def body_jacobian(self, body_name : Path, symbols, source_frame : Path = Path('world')) -> gm.KVArray:
        """6xn spatial body Jacobian of `body_name` in `source_frame`, [omega; v] ordering.

        Column `i` is the twist induced by `symbols[i]`; columns for symbols the body's FK
        does not depend on are exactly zero.
        """
        fk = self.get_fk(body_name, source_frame).transform
        J  = gm.zeros((6, len(symbols)), dtype=object)
        for i, s in enumerate(symbols):
            J[:, i] = gm.jacobian_column(fk, s)
        return J

    def get_mass_matrix(self, symbols, source_frame : Path = Path('world')) -> gm.KVArray:
        """Joint-space mass matrix M(q) as an n x n symbolic KVArray, n = len(symbols).

        Built from the Jacobian formulation  M = sum_i J_i^T I_i J_i  over all bodies
        (frames carrying an `inertial`), with each J_i and I_i expressed in `source_frame`,
        in [omega; v] (angular-first) ordering. Symbol ordering of the output follows the
        order of `symbols`.

        Caveats:
        - **Non-inertial source_frame.** If `source_frame`'s own pose depends on any of the
          requested symbols, the result describes relative motion and is not a valid
          rigid-body mass matrix (fictitious-force terms are missing). This raises rather
          than silently returning a wrong matrix.
        - **Expression size.** M has n**2 configuration-dependent entries; symbolic
          expansion may blow up for very high-DoF systems. This is the naive reference
          implementation a future CRBA version would be validated against.
        """
        symbol_set = set(symbols)
        sym_index  = {s: i for i, s in enumerate(symbols)}
        n          = len(symbols)

        # 1. Guard the reference frame: it must not move under any requested symbol. Walk
        #    the incoming-edge chain from source_frame up to its root and collect the DoFs
        #    it depends on — this works even when source_frame is itself a (world-detached)
        #    root, unlike a get_fk(world, source) lookup.
        if source_frame not in self._nodes:
            raise KeyError(f'source_frame "{source_frame}" is not known.')
        src_symbols = set()
        cursor      = source_frame
        while cursor in self._incoming_edges:
            edge        = self._incoming_edges[cursor]
            src_symbols |= set(edge.symbols)
            cursor       = edge.parent
        clash = symbol_set & src_symbols
        if clash:
            raise ValueError(f'source_frame "{source_frame}" depends on requested symbols {clash}; '
                             f'the result would not be a valid rigid-body mass matrix.')

        # 2./3. Active edges -> flood down -> per-body union of active symbols (dedup by body).
        body_symbols = {}
        for edge in self._incoming_edges.values():
            active = symbol_set & set(edge.symbols)
            if not active:
                continue
            for name in self.get_subtree(edge.child):
                if getattr(self._nodes[name], 'inertial', None) is not None:
                    body_symbols.setdefault(name, set()).update(active)

        # 4. Accumulate one J^T I J term per body.
        M = gm.zeros((n, n), dtype=object)
        for name, active in body_symbols.items():
            body = self._nodes[name]
            fk   = self.get_fk(name, source_frame).transform
            I6   = gm.spatial_inertia(body.inertial, fk)
            J    = gm.zeros((6, n), dtype=object)
            for s in active:
                J[:, sym_index[s]] = gm.jacobian_column(fk, s)
            M = M + J.T.dot(I6).dot(J)

        # 5. Symmetrize against asymmetry introduced during symbolic expansion.
        return 0.5 * (M + M.T)

    def _gen_tf(self, chain : Iterable[DirectedEdge]) -> gm.KVArray:
        tf = gm.Transform.identity()
        for e in chain:
            tf = e.eval(self, tf)
        return tf

    def _get_path(self, start : Path, end : Path):
        out = []
        current = start
        
        while current != end and current in self._incoming_edges:
            e = self._incoming_edges[current]
            out.append(e)
            current = e.parent

        return out

    def get_root(self, start : Path, filter : set[Path]=None, return_path : bool=False) -> Path | list[Path]:
        # The node is disconnected
        if start not in self._nodes:
            raise KeyError(f'Unknown node {start}')

        if start not in self._incoming_edges:
            return [start] if return_path else start
        
        e = self._incoming_edges[start]
        # End of recursion because of filter
        if filter is not None and e.parent not in filter:
            return [start] if return_path else start
        out = self.get_root(e.parent, filter, return_path=return_path)
        return out + [start] if return_path else out
