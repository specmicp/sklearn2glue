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

from sklearn2glue.core.model import ClusteringModel
from sklearn2glue.core.subset import ClusterSubsetState

def create_model(data, attribute_list, model_class, *args, **kwargs):
    """Create a custering model using the Glue data"""
    skmodel = model_class(*args, **kwargs)
    wrapped_model = ClusteringModel(skmodel, data, attribute_list)

    return wrapped_model

def fit_and_create_clusters(dc, data_label, attribute_list, model_class, *args, name_pattern=None, **kwargs):
    """Fit a clustering model and create clusters."""
    data = dc[data_label]
    wrapped_model = create_model(data, attribute_list, model_class, *args, **kwargs)

    if name_pattern is None:
        name_pattern = "cluster_{0}"
    for i in range(wrapped_model.n_clusters):
        state = ClusterSubsetState(wrapped_model, i)
        dc.new_subset_group(name_pattern.format(i), state)