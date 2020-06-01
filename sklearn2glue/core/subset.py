 # Copyright (c) 2020 Fabien Georget <fabien.georget@epfl.ch>, EPFL
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Define a subset state for a cluster found by a scikit-learn clustering model"""

import numpy as np

from glue.core.subset import SubsetState

#  from sklearn2glue.core.model import ClusteringModel
from glue.core.decorators import memoize


class ClusterSubsetState(SubsetState):
    """A subset state for a cluster found by a scikit-learn clustering model.
    """

    def __init__(self, model, cluster_number):
        super().__init__()
        self._model = model
        self._cluster = cluster_number

    @property
    def model(self):
        return self._model

    @property
    def attributes(self):
        return self._model.attributes

    @property
    def subset_state(self):  # convenience method, mimic interface of Subset
        return self

    def copy(self):
        return ClusterSubsetState(self._model, self._cluster_number)

    @memoize
    def to_mask(self, data, view=None):
        pred = self.model.predict(data)
        result = np.asarray(pred == self._cluster)
        if view is not None:
            result = result[view]
        return result


