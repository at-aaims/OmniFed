# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: all

all: python_conda_install clean local

python_conda_install:
	cd installation_scripts && chmod a+x *.sh && ./install_python_conda.sh

clean:
	rm -rf ./logs && mkdir ./logs

local:
	echo "build over local machine"
	cd installation_scripts && ./install_mpi.sh && pip install -r requirements.txt

docker:
	echo "installing docker" && cd installation_scripts && ./install_docker.sh

lxc:
	echo "installing linux containers lxc" && cd installation_scripts && ./install_lxc.sh

singularity:
	echo "setting up containerization with singularity" && cd installation_scripts && ./install_singularity.sh

shifter:
	echo "no support for shifter yet..."

mqtt:
	echo "installing mqtt server..." && cd installation_scripts && ./install_mqtt.sh

pytorch:
	# use argument "USE_MPI" to build pytorch from source with MPI support. currently not using it (default is Gloo)
	echo "building pytorch from source..." && cd installation_scripts && ./install_pytorch_from_source.sh "NO_MPI"