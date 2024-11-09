import json
from web3 import Web3

# Load contract info
with open('blockchain/contract_info.json', 'r') as file:
    contract_info = json.load(file)

# Connect to the blockchain
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
w3.eth.default_account = w3.eth.accounts[0]

# Load the contract
contract = w3.eth.contract(
    address=contract_info['address'],
    abi=contract_info['abi']
)

def report_domain(domain, ip, is_malicious):
    
    votes = 3  # Assume 3 votes are required for consensus (for simplicity)
    if votes > 1:
        tx_hash = contract.functions.addDomain(domain, ip, is_malicious).transact()
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Domain {domain} reported as {'malicious' if is_malicious else 'benign'} and added to the blockchain.")
    else:
        print(f"Consensus not reached for domain {domain}.")


report_domain('example.com', '192.168.0.1', True)
