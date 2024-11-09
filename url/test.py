import pandas as pd
import tldextract
import socket

# Define the path to your dataset
file_path = 'book19.csv'  # Replace with the actual path to your CSV file

# Define the path to the output CSV file
output_file_path = 'result19.csv'  # Replace with the desired output file path

# Read the dataset into a DataFrame
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
    print(df.head())  # Print the first few rows of the DataFrame to verify
except FileNotFoundError:
    print(f"File not found: {file_path}")
    df = pd.DataFrame()  # Create an empty DataFrame
except pd.errors.EmptyDataError:
    print("No data found in the file")
    df = pd.DataFrame()  # Create an empty DataFrame

# Ensure the DataFrame is not empty and contains a 'url' column
if not df.empty and 'url' in df.columns:
    data = []

    for url in df['url']:
        # Extract domain and tld
        extracted = tldextract.extract(url)
        domain = extracted.domain
        tld = extracted.suffix
        full_domain = f"{domain}.{tld}"

        # Extract IP address
        try:
            ip_address = socket.gethostbyname(full_domain)
        except Exception as e:
            ip_address = None

        # Calculate URL length
        url_length = len(url)

        # Check if URL uses HTTPS
        https_flag = url.startswith("https")

        # Append the extracted data to the list
        data.append([url, domain, tld, ip_address, url_length, https_flag])

    # Create a DataFrame from the extracted data
    result_df = pd.DataFrame(data, columns=["url", "domain", "tld", "ip", "url_len", "https"])

    # Save the DataFrame to a CSV file
    result_df.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")

    # Print the DataFrame
    print(result_df)
else:
    print("DataFrame is empty or 'url' column is missing. Please check your data source.")

