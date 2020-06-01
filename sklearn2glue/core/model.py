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

"""Wrappers for scikit-learn models"""

import numpy as np

from glue.core.decorators import memoize
from glue.core.exceptions import IncompatibleAttribute

class ClusteringModel:
    """A wrapper for a scikit-learn clustering model."""
    def __init__(self, model, data, attributes):

        self._model = model
        self._fitdata = data
        self._attributes = attributes

    @property
    def model(self):
        return self._model

    @property
    def fit_data(self):
        return self._data

    @property
    def attributes(self):
        return self._attributes

    def build_feature_matrix(self, data):
        return np.column_stack([data[att].ravel() for att in self._attributes])

    @memoize
    def predict(self, data):
        y_pred = self.model.predict(self.build_feature_matrix(data))
        predict = y_pred.reshape(data.shape)
        return predict