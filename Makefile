.PHONY: all

all: python_conda_install clean pytorch

python_conda_install:
	cd scripts && chmod a+x *.sh && ./install_python_conda.sh

clean:
	rm -rf ./logs && mkdir ./logs

local:
	echo "build over local machine"

# first check if each exists...
docker:
	echo "install docker and applicable images (if any)..."

lxc:
	echo "install and setup linux containers"

singularity:
	echo "set up containerization with singularity"

shifter:
	echo "no support for shifter yet..."

pytorch:
	# build pytorch from source
	# check if building locally or over containers. if using container, check if appropriate image exists
	# checkpoint a specific version with the repo before building...
	echo "add support to build pytorch over rocm/cuda/cpu." && cd scripts && ./install_pytorch_from_source.sh