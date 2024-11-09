from web3 import Web3
from solcx import compile_standard, set_solc_version
import json

# Install and set Solc version
set_solc_version('0.8.0')

# Load the Solidity contract
with open('blockchain/MaliciousDomainRegistry.sol', 'r') as file:
    contract_source_code = file.read()

# Compile the contract
compiled_sol = compile_standard({
    "language": "Solidity",
    "sources": {
        "MaliciousDomainRegistry.sol": {
            "content": contract_source_code
        }
    },
    "settings": {
        "outputSelection": {
            "*": {
                "*": [
                    "abi", "metadata", "evm.bytecode"
                ]
            }
        }
    }
})

# Save the compiled contract
with open('blockchain/MaliciousDomainRegistry.json', 'w') as file:
    json.dump(compiled_sol, file)

# Connect to the blockchain (e.g., local Ganache instance)
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))

# Ensure connection is successful
if not w3.is_connected():
    raise ConnectionError("Failed to connect to the Ethereum node.")

# Set the default account (use the first account in the list)
w3.eth.defaultAccount = w3.eth.accounts[0]

# Get the bytecode and ABI
bytecode = compiled_sol['contracts']['MaliciousDomainRegistry.sol']['MaliciousDomainRegistry']['evm']['bytecode']['object']
abi = compiled_sol['contracts']['MaliciousDomainRegistry.sol']['MaliciousDomainRegistry']['abi']

# Deploy the contract
MaliciousDomainRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = MaliciousDomainRegistry.constructor().transact({'from': w3.eth.defaultAccount})
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Save the contract address and ABI for later use
contract_address = tx_receipt.contractAddress
with open('blockchain/contract_info.json', 'w') as file:
    json.dump({"address": contract_address, "abi": abi}, file)

print(f"Contract deployed at address: {contract_address}")
