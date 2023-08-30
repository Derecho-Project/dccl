## Here is a set of scripts to evaluate DCCL in Azure's HPC environment

# STEP 1. Set up the environment in Azure
Create a Virtual Machine Scale Set with the following settings:
- Orchestration mode: "Uniform"
- Image: "Ubuntu-HPC 2204 -x64 Gen2"
- Security type: Standard
- Run with Azure Spot discount: check it for lower price
- Size: Standard_HC44-16rs (Standard_HC44_32rs and Standard_HC44 also work, we can also use HB and N series)
- User name and SSH public key: set accordingly
- Networking: allow port 22
- use this [link](https://learn.microsoft.com/en-us/rest/api/compute/virtual-machine-scale-sets/convert-to-single-placement-group?tabs=HTTP) to change the VMSS in single placement group mode.

# STEP 2. Run experiment
- Run `./get_nodes.sh` to generate the ndoe list files, myrankfile, myhostfile.
- Run `./deploy.sh` to deploy the code in the VMSS. It takes around 10 min.
- Run `./run_remote.sh` to do the experiments. The data will be collected in a set of .tar.bz2 files.
