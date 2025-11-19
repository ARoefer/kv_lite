# Copyright (c) 2025 Adrian Röfer, Robot Learning Lab, University of Freiburg
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

import numpy as np
import rerun

from pathlib import Path

from . import math as gm

from .graph import FKChainException
from .model import Model


class ModelRerunBroadcaster(object):
    def __init__(self, model : Model, ref_frame : str):
        self._ref_frame = ref_frame
        self._ref_name_rerun = '-'.join(ref_frame.parts)
        self.refresh_model(model)

    def refresh_model(self, model : Model):
        fk_expression = []
        names = []
        for f_name in model.get_frames():
            try:
                fk_expression.append(model.get_fk(f_name, self._ref_frame).transform)
                names.append(f_name)
            except FKChainException:
                pass

        self._expr_frames = gm.stack(fk_expression, axis=0)
        self._names = ['-'.join(n.parts) for n in names]

    
    def update(self, q : np.ndarray | dict[gm.Symbol, float], time_name="step", axis_length=0.2, root_override=np.eye(4)):
        # Decode dictionary eval
        if isinstance(q, dict):
            try:
                series_dim = len(next(iter(q.values())))
            except TypeError:
                series_dim = 1
            _q = np.empty((series_dim, len(self._expr_frames.symbols)))
            for x, s in enumerate(self._expr_frames.ordered_symbols):
                _q[:, x] = q[s]
            q = _q

        rerun.set_time(time_name, sequence=0)
        rerun.log(f'/', rerun.Transform3D(), static=True)

        ref_Ts_frames = self._expr_frames(q)
        if ref_Ts_frames.ndim >= 4:
            indexer = [rerun.TimeColumn(time_name, sequence=np.arange(len(ref_Ts_frames)))]
        else:
            indexer = None
            # Adding fake time dimension
            ref_Ts_frames = ref_Ts_frames[None]

        for n, f in zip(self._names, ref_Ts_frames.transpose(1, 0, 2, 3)):
            if n == self._ref_name_rerun:
                rerun.log(f'{n}/label', rerun.Boxes3D(sizes=[0, 0, 0], fill_mode='solid', labels=[n]))
                rerun.log(n, rerun.Transform3D(translation=root_override[:3, 3],
                                               mat3x3=root_override[:3, :3],
                                               axis_length=axis_length))
            else:
                data_name = f'{self._ref_name_rerun}/{n}'
                rerun.log(f'{data_name}/label', rerun.Boxes3D(sizes=[0, 0, 0], fill_mode='solid', labels=[n]))

                if indexer is None:
                    rerun.log(data_name, rerun.Transform3D(translation=f[0, :3, 3],
                                                                        mat3x3=f[0, :3, :3],
                                                                        axis_length=axis_length))
                else:
                    rerun.send_columns(data_name,
                                    indexes=indexer,
                                    columns=rerun.Transform3D.columns(translation=f[..., :3, 3],
                                                                        mat3x3=f[..., :3, :3],
                                                                        axis_length=[axis_length]*len(f)))

    @property
    def symbols(self):
        return self._expr_frames.symbols

    @property
    def ordered_symbols(self):
        return self._expr_frames.ordered_symbols
    
    def set_symbol_order(self, symbols):
        self._expr_frames.set_symbol_order(symbols)

