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
	cd scripts && chmod a+x *.sh && ./install_python_conda.sh

clean:
	rm -rf ./logs && mkdir ./logs

local:
	echo "build over local machine"
	pip install -r requirements.txt

docker:
	echo "installing docker" && cd scripts && ./install_docker.sh

lxc:
	echo "installing linux containers lxc" && cd scripts && ./install_lxc.sh

singularity:
	echo "setting up containerization with singularity" && cd scripts && ./install_singularity.sh

shifter:
	echo "no support for shifter yet..."

mqtt:
	echo "installing mqtt server..." && cd scripts && ./install_mqtt.sh

pytorch:
	# use argument "USE_MPI" to build pytorch from source with MPI support. currently not using it
	echo "building pytorch from source..." && cd scripts && ./install_pytorch_from_source.sh "NO_MPI"