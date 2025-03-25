clean:
	rm -rf ./logs/*

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
	# check if building locally or over containers. if using container, check if appropriate image exists
	# checkpoint a specific version with the repo before building...
	echo "add support to build pytorch over rocm or cuda/cpu. MPI/Gloo/NCCL"