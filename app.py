from flask import Flask, request, render_template, redirect, url_for
from inte3 import classify_and_report_domain, extract_features

app = Flask(__name__)

BLACKLIST_FILE = '../blacklist.txt'

def add_to_blacklist(url):
    with open(BLACKLIST_FILE, 'a') as file:
        file.write(url + '\n')

def is_blacklisted(url):
    with open(BLACKLIST_FILE, 'r') as file:
        blacklisted_urls = file.read().splitlines()
    return url in blacklisted_urls

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def classify():
    url = request.form['url']
    # Extract features
    domain, tld, ip, url_len, https = extract_features(url)
    features = [hash(domain), hash(tld), hash(ip), url_len, https, len(url)]
    # Predict
    is_malicious, _ = classify_and_report_domain(url, domain, tld, ip, url_len, https)
    prediction = 'Malicious' if is_malicious else 'Benign'
    return render_template('result.html', url=url, domain=domain, tld=tld, ip=ip, url_len=url_len, https=https, prediction=prediction)

@app.route('/blacklist', methods=['POST'])
def blacklist():
    url = request.form['url']
    # Add to blacklist (e.g., save to a file or database)
    if not is_blacklisted(url):
        add_to_blacklist(url)
    return redirect(url_for('index'))
    #with open('blacklist.txt', 'a') as f:
        #f.write(url + '\n')
    #return 'URL has been blacklisted.'

if __name__ == '__main__':
    app.run(debug=True)

