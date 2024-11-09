import socket
import threading
import json
import sys
from web3 import Web3
import joblib
import numpy as np
from urllib.parse import urlparse

# Load the trained model
model = joblib.load('../models/trained_model.pkl')

class Node:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []

    def start(self):
        threading.Thread(target=self.listen_for_peers).start()
        threading.Thread(target=self.connect_to_peers).start()

    def listen_for_peers(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)
        print(f"Listening for peers on {self.host}:{self.port}...")

        while True:
            client, address = server.accept()
            print(f"Accepted connection from {address}")
            threading.Thread(target=self.handle_peer, args=(client,)).start()

    def handle_peer(self, client):
        while True:
            try:
                data = client.recv(1024).decode()
                if data:
                    print(f"Received data: {data}")
                    self.broadcast_message(data)
            except:
                client.close()
                return False

    def connect_to_peers(self):
        while True:
            for peer in self.peers:
                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect(peer)
                    threading.Thread(target=self.handle_peer, args=(client,)).start()
                except:
                    continue

    def broadcast_message(self, message):
        for peer in self.peers:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect(peer)
                client.send(message.encode())
                client.close()
            except:
                continue

def get_domain_tld(url):
    """
    Extract the domain and TLD from a given URL.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.hostname
    tld = domain.split('.')[-1] if domain else ""
    return domain, tld

def get_ip_address(domain):
    """
    Resolve the IP address of the given domain.
    """
    try:
       ip = socket.gethostbyname(domain)
    except socket.gaierror:
       ip = '0.0.0.0'
    return ip

def extract_features(url):
    domain, tld = get_domain_tld(url)
    ip = get_ip_address(domain)
    url_len = len(url)
    https = 1 if url.startswith('https') else 0
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_digits = sum(c.isdigit() for c in url)
    
    features = [
    hash(domain), hash(tld), hash(ip),
    url_len, https, num_dots, num_hyphens, num_digits
    ]
    return features

def classify_and_report_domain(url):
    features = extract_features(url)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    
    print(f"Features: {features}")
    print(f"Prediction: {prediction}")
    
    is_malicious = prediction == 1
    if is_malicious:
       tx_receipt = report_to_blockchain(url)
    else:
       tx_receipt = None
    
    return is_malicious, tx_receipt
	
# Function to classify and report domain
def classify_and_report_domain(url, domain, tld, ip, url_len, https):
    features = [hash(url), hash(domain), hash(tld), hash(ip), url_len, https]
    prediction = model.predict([features])[0]

    is_malicious = bool(prediction)

    # If malicious, add to the blockchain (simplified)
    if is_malicious:
        web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        # Add to blockchain logic here
        tx_receipt = "Simulated Blockchain Transaction Receipt"
    else:
        tx_receipt = None

    return is_malicious, tx_receipt

# Function to extract features from a URL
def extract_features(url):
    domain = url.split("//")[-1].split("/")[0].split(":")[0]
    tld = domain.split(".")[-1]
    ip = socket.gethostbyname(domain)
    url_len = len(url)
    https = url.startswith("https")

    return domain, tld, ip, url_len, https