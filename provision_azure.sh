#!/bin/bash
# Provision an Azure Compute Fleet (spot) for torformer torus_probe training.
#
# Verified available GPU SKUs in allowed regions:
#   westus3  -> NC24ads_A100_v4, NC48ads_A100_v4, NC96ads_A100_v4  (A100 family)
#   uksouth  -> NC24ads_A100_v4, NC48ads_A100_v4                   (A100 family)
#
# The fleet tries the largest A100 size first (most GPUs = fastest training),
# falling back to smaller sizes if unavailable at spot price.
#
# Prerequisites:
#   brew install azure-cli    (or: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash)
#   az login
#
# Usage:
#   bash provision_azure.sh              # provision fleet, print SSH when VM is up
#   bash provision_azure.sh --destroy    # tear down everything
#   LOCATION=uksouth bash provision_azure.sh   # use uksouth instead of westus3

set -euo pipefail

LOCATION="${LOCATION:-westus3}"
RG="torformer-fleet-rg"
FLEET_NAME="torformer-fleet"
VNET_NAME="torformer-vnet"
SUBNET_NAME="torformer-subnet"
NSG_NAME="torformer-nsg"
ADMIN_USER="azureuser"
REPO_URL="https://github.com/satvikprasad/torformer.git"
API_VER="2024-11-01"
SUB=$(az account show --query id -o tsv)

if [[ "${1:-}" == "--destroy" ]]; then
    echo "Deleting resource group $RG ..."
    az group delete --name "$RG" --yes --no-wait
    echo "Deletion queued. Monitor with: az group show -n $RG"
    exit 0
fi

SSH_KEY_FILE="${HOME}/.ssh/id_rsa"
if [ ! -f "${SSH_KEY_FILE}.pub" ]; then
    echo "Generating SSH key at ${SSH_KEY_FILE} ..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_FILE" -N ""
fi
SSH_PUB_KEY=$(cat "${SSH_KEY_FILE}.pub")

echo "Creating resource group: $RG ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

echo "Creating VNet + NSG ..."
az network vnet create \
    --resource-group "$RG" --name "$VNET_NAME" --location "$LOCATION" \
    --address-prefix 10.0.0.0/16 \
    --subnet-name "$SUBNET_NAME" --subnet-prefix 10.0.0.0/24 \
    --output none

az network nsg create \
    --resource-group "$RG" --name "$NSG_NAME" --location "$LOCATION" --output none

az network nsg rule create \
    --resource-group "$RG" --nsg-name "$NSG_NAME" \
    --name AllowSSH --priority 1000 \
    --protocol Tcp --direction Inbound --access Allow \
    --source-address-prefixes '*' --destination-port-ranges 22 \
    --output none

az network vnet subnet update \
    --resource-group "$RG" --vnet-name "$VNET_NAME" --name "$SUBNET_NAME" \
    --network-security-group "$NSG_NAME" --output none

SUBNET_ID=$(az network vnet subnet show \
    --resource-group "$RG" --vnet-name "$VNET_NAME" --name "$SUBNET_NAME" \
    --query id -o tsv)
NSG_ID=$(az network nsg show \
    --resource-group "$RG" --name "$NSG_NAME" --query id -o tsv)

CLOUD_INIT_B64=$(python3 -c "
import base64
admin='${ADMIN_USER}'
repo='${REPO_URL}'
script = '''#cloud-config
package_update: true
package_upgrade: false
packages:
  - git
  - curl
  - build-essential
runcmd:
  - DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nvidia-driver-535
  - su - ''' + admin + ''' -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  - git clone ''' + repo + ''' /home/''' + admin + '''/torformer
  - chown -R ''' + admin + ''':''' + admin + ''' /home/''' + admin + '''/torformer
  - printf '#!/bin/bash\nset -e\nexport WANDB_RUN=\"\${WANDB_RUN:-torus-probe}\"\nexport PYTHONPATH=\"/home/''' + admin + '''/torformer:\${PYTHONPATH:-}\"\ncd /home/''' + admin + '''/torformer/nanochat\nbash runs/torus_probe.sh 2>&1 | tee /home/''' + admin + '''/torus_probe.log\n' > /home/''' + admin + '''/run_probe.sh
  - chmod +x /home/''' + admin + '''/run_probe.sh
  - chown ''' + admin + ''':''' + admin + ''' /home/''' + admin + '''/run_probe.sh
power_state:
  mode: reboot
  timeout: 60
  condition: True
'''
print(base64.b64encode(script.encode()).decode())
")

FLEET_BODY=$(python3 -c "
import json
body = {
    'location': '${LOCATION}',
    'properties': {
        'vmSizesProfile': [
            {'name': 'Standard_NC96ads_A100_v4'},
            {'name': 'Standard_NC48ads_A100_v4'},
            {'name': 'Standard_NC24ads_A100_v4'},
        ],
        'spotPriorityProfile': {
            'capacity': 1,
            'evictionPolicy': 'Delete',
            'allocationStrategy': 'PriceCapacityOptimized',
            'maintain': True,
        },
        'computeProfile': {
            'baseVirtualMachineProfile': {
                'storageProfile': {
                    'imageReference': {
                        'publisher': 'Canonical',
                        'offer': '0001-com-ubuntu-server-jammy',
                        'sku': '22_04-lts-gen2',
                        'version': 'latest',
                    },
                    'osDisk': {
                        'createOption': 'FromImage',
                        'managedDisk': {'storageAccountType': 'Premium_LRS'},
                        'diskSizeGB': 256,
                    },
                },
                'osProfile': {
                    'computerNamePrefix': 'torfleet',
                    'adminUsername': '${ADMIN_USER}',
                    'linuxConfiguration': {
                        'disablePasswordAuthentication': True,
                        'ssh': {
                            'publicKeys': [{'path': '/home/${ADMIN_USER}/.ssh/authorized_keys', 'keyData': '${SSH_PUB_KEY}'}]
                        },
                    },
                    'customData': '${CLOUD_INIT_B64}',
                },
                'networkProfile': {
                    'networkApiVersion': '2020-11-01',
                    'networkInterfaceConfigurations': [{
                        'name': 'torfleet-nic',
                        'properties': {
                            'primary': True,
                            'networkSecurityGroup': {'id': '${NSG_ID}'},
                            'ipConfigurations': [{
                                'name': 'torfleet-ipconfig',
                                'properties': {
                                    'subnet': {'id': '${SUBNET_ID}'},
                                    'publicIPAddressConfiguration': {
                                        'name': 'torfleet-pip',
                                        'properties': {'publicIPAllocationMethod': 'Dynamic', 'idleTimeoutInMinutes': 30},
                                        'sku': {'name': 'Standard', 'tier': 'Regional'},
                                    },
                                },
                            }],
                        },
                    }]
                },
            }
        },
    },
}
print(json.dumps(body))
")

echo "Creating Compute Fleet: $FLEET_NAME ..."
FLEET_URL="https://management.azure.com/subscriptions/${SUB}/resourceGroups/${RG}/providers/Microsoft.AzureFleet/fleets/${FLEET_NAME}?api-version=${API_VER}"
az rest --method PUT --url "$FLEET_URL" --body "$FLEET_BODY" --output none

echo "Fleet creation requested. Waiting for a spot VM instance (~2-5 min) ..."

for i in $(seq 1 30); do
    sleep 20
    PUBLIC_IP=$(az network public-ip list \
        --resource-group "$RG" \
        --query "[?ipAddress != null].ipAddress | [0]" -o tsv 2>/dev/null || true)
    if [[ -n "${PUBLIC_IP:-}" && "$PUBLIC_IP" != "None" ]]; then
        break
    fi
    echo "  ... waiting for public IP (${i}/30)"
done

echo ""
if [[ -z "${PUBLIC_IP:-}" || "$PUBLIC_IP" == "None" ]]; then
    echo "VM not yet allocated. To check later:"
    echo "  az network public-ip list --resource-group $RG --query '[].ipAddress' -o tsv"
    echo "  az rest --method GET --url '${FLEET_URL}' --query properties.provisioningState -o tsv"
else
    VM_SIZE_ALLOC=$(az vm list --resource-group "$RG" \
        --query "[0].hardwareProfile.vmSize" -o tsv 2>/dev/null || echo "unknown")
    echo "Fleet VM is up at: $PUBLIC_IP  (size: $VM_SIZE_ALLOC)"
    echo ""
    echo "cloud-init installs NVIDIA drivers and reboots (~2-3 min). Then:"
    echo ""
    echo "  ssh ${ADMIN_USER}@${PUBLIC_IP}"
    echo "  nvidia-smi                   # verify GPU is visible post-reboot"
    echo "  bash ~/run_probe.sh          # launch baseline vs toroidal ablation"
    echo "  tail -f ~/torus_probe.log    # follow training progress"
    echo ""
    echo "Stream logs from your laptop:"
    echo "  ssh ${ADMIN_USER}@${PUBLIC_IP} 'tail -f ~/torus_probe.log'"
fi
echo ""
echo "Teardown:  bash provision_azure.sh --destroy"
