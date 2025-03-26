.PHONY: all

all: python_conda_install clean pytorch

python_conda_install:
	cd scripts && chmod a+x *.sh && ./install_python_conda.sh

clean:
	rm -rf ./logs && mkdir ./logs

local:
	echo "build over local machine"
	pip install -r requirements.txt

# first check if each exists...
docker:
	echo "installing docker" && cd scripts && ./install_docker.sh

lxc:
	echo "install and setup linux containers"

singularity:
	echo "set up containerization with singularity"

shifter:
	echo "no support for shifter yet..."

pytorch:
	# use argument "USE_MPI" to build pytorch from source with MPI support
	echo "build pytorch from source." && cd scripts && ./install_pytorch_from_source.sh "NO_MPI"