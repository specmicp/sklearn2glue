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
    def __init__(self, model, data, attributes, fit=True):
        """
        Parameters
        ----------
        model : sklearn2glue.core.model.ClusteringModel
            The wrapped scikit learn model to use
        data : glue.core.data.Data
            The data to use/used to fit the model
        attributes : list(str)
            The list of attributes to use
        fit : bool, optional
            If true, fit the model during object instanciation. The default is True.

        """
        self._model = model
        self._data = data
        self._attributes = attributes
        self._component_ids = [data.id[att] for att in attributes]

        if fit:
            features = self.build_feature_matrix(data)
            pred = self._model.fit_predict(features)
            pred = pred.reshape(data.shape)
        else:
            pred = self.predict(data)
        self._n_clusters = pred.max()+1

    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def model(self):
        return self._model

    @property
    def fitted_data(self):
        return self._data

    @property
    def attributes(self):
        """The list of attributes (as string) used to create the model"""
        return self._attributes

    def build_feature_matrix(self, data):
        """Build the feature matrix corresponding to the attributes of the model

        Raises:
        -------
            IncompatibleAttribute: if the cid cannot be found or calculated
        """
        return np.column_stack([data[cid].ravel() for cid in self._component_ids])

    @memoize
    def predict(self, data):
        """Returns the labelling for the data."""
        y_pred = self.model.predict(self.build_feature_matrix(data))
        predict = y_pred.reshape(data.shape)
        return predict
