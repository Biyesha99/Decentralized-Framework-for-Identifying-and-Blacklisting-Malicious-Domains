# Decentralized-Framework-for-Identifying-and-Blacklisting-Malicious-Domains
Decentralized Framework for Identifying and Blacklisting Malicious Domains through IP Address Analysis
Chapter 1: Introduction

1.1	Project Overview
This project aims to develop a framework that leverages machine learning and blockchain to detect and blacklist malicious domains, thereby improving web security and safeguarding users from cyber threats such as phishing, malware, and fraudulent websites. The framework is designed to identify malicious URLs by analyzing a set of extracted features, i.e., domain information, URL structure, and IP addresses, and classifying each URL as either benign or malicious. The intended audience includes cybersecurity experts, network administrators, and organizations seeking to enhance their threat detection mechanisms.
Scope and Approach
The project's scope includes building a machine learning-based classification model, designing a decentralized blockchain solution to store and manage blacklisted URLs, and creating a peer-to-peer (P2P) network for consensus on malicious domains. The machine learning model, specifically using the Random Forest Classifier, analyzes URLs to identify potential threats based on features engineered from domain properties, URL length, and other structural indicators. Blockchain integration, through smart contracts, provides a record of malicious URLs, ensuring data integrity and preventing unauthorized tampering. The framework includes a P2P network to allow nodes to achieve consensus on blacklisted domains, enhancing security and transparency.
Assumptions
•	The input data (URLs) is assumed to be comprehensive enough to train an accurate model.
•	Blockchain provides a reliable and secure method for storing blacklisted URLs.
•	The decentralized nature of the P2P network improves resilience and reduces vulnerability to single points of failure.
Outcomes
The project showcases a functional framework with an machine learning model achieving an accuracy of approximately 85-93% in identifying malicious URLs. By storing data on a blockchain, the system ensures data authenticity and resilience against tampering, while the P2P consensus mechanism provides additional reliability in threat detection. This framework offers an advanced, scalable solution for organizations aiming to protect users from online threats through a hybrid approach that combines artificial intelligence and blockchain technology.
1.2	Background
In the era of digital information, web security is a top priority for individuals, businesses, and governments due to the rapid increase in cyber threats like phishing, and malware distribution. Traditional URL blacklisting mechanisms often rely on centralized databases that are updated periodically; however, these can be limited by delays in identifying new threats and can create vulnerabilities due to their centralized nature. With the rise in malicious activities, a need exists for a more efficient and robust method to identify, track, and blacklist dangerous domains in real time. This project aims to address this challenge by using a machine learning-based classification model combined with blockchain technology for secure data management and a peer-to-peer network for consensus.
Blockchain provides a decentralized, tamper-resistant solution for managing and updating blacklists, while machine learning models enable rapid identification of malicious URLs. Previous works in this area have largely focused on either isolated machine learning models for threat detection or static blacklisting databases; however, combining these techniques into a unified framework that integrates decentralized consensus and automated classification represents a new approach.

1.3 Problem Statement
The primary aim of this research is to develop a decentralized framework for identifying and blacklisting malicious domains through IP address analysis. Existing methods are often restricted by centralization, delayed updates, or lack of scalability, creating security gaps that attackers can exploit. This project seeks to address these limitations by combining machine learning, blockchain, and peer-to-peer networking to create an automated, resilient, and transparent system for detecting and blocking malicious URLs in real time. By using the power of machine learning for accurate classification and blockchain for secure storage, the framework aims to offer a scalable solution to enhance cybersecurity and mitigate risks associated with malicious domains. 
Chapter 2: Related Work

2.1 Introduction to Malicious Domain Detection
In cybersecurity, detecting malicious domains is a critical task to prevent phishing, malware, and other online threats. Traditional methods of identifying such domains are increasingly challenged by evolving attack patterns and sophisticated techniques used by malicious actors. This chapter explores various approaches to malicious domain detection, from early signature-based methods to advanced machine learning and blockchain-based techniques, highlighting their limitations and the need for the hybrid approach adopted in this project.
Recent advancements in machine learning have significantly contributed to detecting phishing and malicious domains, with algorithms that analyze features like WHOIS data, DNS queries, and URL characteristics for suspicious patterns [1], [2]. Additionally, blockchain has gained attention for its decentralized approach to blacklisting malicious domains, offering tamper-proof solutions that are resistant to centralized system failures [9].

2.2 Signature-Based Detection and Its Limitations
Early approaches to identifying malicious domains relied on signature-based detection, which involves creating signatures of known malicious patterns to flag similar domains. Despite their initial success, these methods face several limitations, particularly their inability to detect zero-day attacks,  new threats for which signatures do not yet exist. Furthermore, signature-based methods require continuous updating, which limits their effectiveness against increasingly dynamic threats.
For example, phishing detection systems using traditional blacklists often fail to detect emerging threats due to their reliance on predefined signature patterns, which can be easily bypassed by attackers employing new domain-generation algorithms or techniques [1].

2.3 Machine Learning Approaches
2.3.1 Traditional Machine Learning Models
The use of machine learning has led to significant advancements in domain detection. Techniques such as Random Forests, Support Vector Machines (SVM), and logistic regression have been applied to classify domains based on features like URL length, domain age, and character frequency. However, traditional models tend to struggle with high-dimensional datasets and are subject to high false-positive rates. These models may misclassify benign domains as malicious, especially in cases where the domain structure appears anomalous but is not inherently harmful [2].
For example, one study applied SVM to DNS data to classify malicious domains, yet found that high-dimensional feature spaces lead to issues with model accuracy and overfitting [2]. Additionally, as the number of features grows, traditional machine learning models may become inefficient or prone to errors due to the curse of dimensionality.

2.3.2 Feature-Based Analysis
Feature engineering has played a key role in machine learning-based detection. Features such as the number of dots in a URL, the use of specific top-level domains (TLDs), and HTTPS presence have been found to significantly contribute to detection accuracy. However, these features alone are often insufficient due to the adaptive tactics of attackers, such as domain generation algorithms that create domains with seemingly random or misleading structures [2], [3].
Studies have shown that adversarial techniques, such as homograph attacks or IDN-based attacks, can bypass feature-based detection methods, as they exploit domain name variations that are hard to distinguish from legitimate domains [5]. These issues highlight the need for more advanced techniques to capture complex domain patterns.

2.3.3 Advanced Machine Learning and Deep Learning Techniques
Recent advancements in deep learning have offered more sophisticated methods to capture complex patterns in-domain data. Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) have been applied to detect malicious domains by learning patterns in domain names and their usage over time. These models require large datasets and substantial computational power. Furthermore, their nature makes interpretation challenging, reducing transparency in decision-making, which is a critical aspect of cybersecurity applications [4].
Recent works on deep learning for phishing detection have shown that while CNNs can capture spatial relationships between characters in domain names, these models can suffer from a lack of interpretability, leading to difficulties in explaining the rationale behind their classifications [4], [9]. While CNNs have shown promise in malware detection using binary features [4], their application in domain name classification is still an area of active research.

2.4 Blockchain for Decentralized Blacklisting
Blockchain has gained traction as a solution for decentralized and tamper-resistant domain blacklisting. The immutability of blockchain offers a secure framework for storing blacklisted domains, ensuring that once a domain is marked as malicious, it cannot be altered without consensus. Blockchain's decentralized nature offers protection against single points of failure, which is a significant concern for centralized domain detection systems [9].
However, despite its advantages, blockchain introduces challenges such as latency in updating records and scalability issues for real-time applications. The need for consensus in decentralized systems can lead to delays, making it difficult to maintain up-to-date blacklists in high-traffic networks [9]. This challenge has led to the exploration of hybrid systems that combine blockchain’s benefits with the speed of centralized detection methods [6].

2.5 Challenges in Current Approaches
High False-Positive Rates: Even advanced models often mislabel benign domains as malicious, which consumes user trust and makes widespread adoption challenging. These high false-positive rates are particularly evident in machine learning models that struggle with high-dimensional feature spaces or in systems relying solely on feature-based detection [2], [3].
Resource Demands: Real-time detection at scale requires significant computational and network resources, which are often prohibitive in practical deployments. This issue is exacerbated in deep learning models, which require large datasets and substantial computational infrastructure [4].
Single-Point Failure in Centralized Approaches: Centralized detection systems are vulnerable to attacks, as a single compromised point can threaten the integrity of the entire system. Blockchain-based solutions mitigate this by offering decentralized, tamper-proof storage, but they come with their own set of scalability and latency challenges [9].

2.6 Need for a Hybrid Decentralized Approach
To address these challenges, a hybrid approach combining machine learning with blockchain has been proposed. This project integrates IP-based analysis through machine learning for primary detection and leverages blockchain to ensure tamper-proof storage of blacklisted domains. By utilizing blockchain’s decentralized nature, the project aims to overcome the vulnerabilities associated with centralized systems and increase reliability. Furthermore, the integration of IP address analysis offers a new layer of security, addressing gaps in existing approaches and providing an additional layer of verification [9].
Hybrid systems have been proposed in previous work, where machine learning models are used to classify domains based on features like WHOIS and DNS data, and blockchain is used for secure, immutable blacklisting [2], [9]. This combination of methods promises a more robust solution to malicious domain detection, ensuring higher accuracy and resilience against evolving threats.


Chapter 03: Tools and Techniques

In this chapter, we describe the key tools and methods employed in the development of the solution for the identification and blacklisting of the domains through the analysis of the IP address data sources. It should be noted that these tools and methodologies are chosen with reference to their applicability in the decentralized context, as well as their effective application in the context of machine learning and dealing with malicious domains.

3.1 Techniques
•	Machine Learning (Random Forest Classifier): The first way of processing and categorizing domains is connected with using the Random Forest classifier, which belongs to the machine learning models. This model is used due to the fact that it can scale for large data and offers high fidelity in binary categorization that detects between benign and malicious domains.
•	IP Address Analysis: IP addresses are used as one of the major components in the detection variable. By this technique, the internet domain names are resolved to their IP addresses and then assessed using heuristics that give insight into their bad behaviors such as geolocation and hosting providers.
•	Feature Extraction and Engineering: There are eight attributes extracted from the URL; the IP address, TLD, domain name, URL length, HTTPS status, and geolocation. These features are passed through an SS7 parser and converted to forms suitable for the training of a machine-learning algorithm.
•	Oversampling (SMOTE): Feature selection is performed in order to reduce the dimensionality of the proposed model, while Synthetic Minority Over-sampling Technique (SMOTE) is utilized in order to overcome the problem of class imbalance of the dataset, and thus the proposed model will not become inclined to the majority class.

3.2 Tools
•	Python: It reflects the fundamental language of which the solution was created. It also supports easily scalable machine learning libraries, data preprocessing, and other elements of distributive systems.
•	Scikit-learn: This is a Python library that is used in data analysis and is most useful in training models, making predictions, and model evaluation. It includes functionalities for Preprocessing data; feature extraction and selection; and Model Building including Random Forests, Logistic Regression, SVM, and others.
•	Pandas and NumPy: All of these libraries are basic in the manipulation and preprocessing of data. Panda is used for data reading and manipulation and NumPy is used for array calculation.
•	Flask: A lightweight web framework for developing the interface in which users enter the URLs for classifications. The Flask framework handles server request and also represents the classification results.
•	Web3.py: An interface for working with applications developed on the base of the Ethereum platform in the Python language. It is to fulfill the blockchain components of the task and make the users enter the banned links and maintain them in a distributed way.
•	Ganache: A regional version of blockchain that is used for experimenting and training in the functions of smart contracts without directly engaging with a real blockchain.
•	HTML, CSS, JavaScript: These are for the front end, to design the input form for the URL and for rendering the output. Now, CSS is used to bring more and more application aesthetic looks.

3.3 Decentralized Framework
•	Blockchain (Ganache and Solidity): For the decentralized aspect, this paper simulates the Ethereum blockchain platform with Ganache. The submission and validation of malicious domain reports are dealt with using smart contracts that are built using the programming language known as Solidity. This makes the blacklisting more secure and clear.

3.4 Program File for Writing, Reading, Modifying Blacklist
•	Text Files for Blacklist Storage: As part of the design of the project, the use of a basic text file for blacklisted URLs is done as well. More advanced implementations would use the database but for simplicity, hereby research project has resorted to the use of text files.

3.5 Version Control
•	Git and GitHub: Version control is achieved by using Git and GitHub during the course of the project development process to manage the code.
These tools and techniques help to make the system efficient and sufficiently large to recognize the malicious domains and, simultaneously, apply the decentralized approach to make it transparent.


Chapter 04: Methodology

This chapter includes an overview of the method used in the total research work and the plan adopted in putting up and deploying a decentralized system for the identification of dates and the subsequent blacklisting through the Analysis of the IP address. The approach used here is meant to provide a high level of results’ replicability, only widely known tools and methods are used. The next few subtopics explain how each stage of the process unfolds in the best detail.

4.1 Data Collection
This work began with data collection for the training and testing of the machine learning model of the project. Two types of datasets were collected:
•	Benign domains: The datasets were obtained from official or public domains of standard websites like Google Safe Search and various GitHub repositories.
•	Malicious domains: Scraped data was from threat intelligence sources, public blacklists, and phishing repositories.
The data was stored in CSV format and contained columns such as:
•	URL
•	Domain name
•	Top-level domain (TLD)
•	IP address
•	HTTPS status
•	URL length
•	Label (benign or malicious)

4.2 Data Preprocessing
Preprocessing of data was an important step in a way that the data to be fed to the learning models was in good condition. The following operations were performed:
•	Feature Extraction: Out of the raw data, new features like domain names, number/length of IP addresses, URL, HTTPS status, and TLD were extracted.
•	Missing Data Handling: For each data type consistencies were preserved, any row having missing values was either dropped or filled with appropriate information.
•	Label Encoding: The target variable (malicious or benign) was converted to a numeric of 0 for benign and 1 for malicious.
•	IP Address Conversion: Such information on the table of translation was extracted in the form of numbers; the IPs were converted into numeric formats with the help of the devised functions for the model.
•	Class Imbalance Handling: The imbalance was corrected by applying another method known as SMOTE (Synthetic Minority Over-sampling Technique) where new synthesized samples of benign and malicious samples in the dataset were developed.

4.3 Develop Machine Learning Model
The method of machine learning employed in this undertaking was the Random Forest Classifier. Therefore, this was chosen this model because it had a high classification rate or performance in tasks that required classification and the manner in which it coped with a lot of features was efficient. The following steps were carried out:
•	Train-Test Split: The collected raw data were first preprocessed and then split into the training dataset, which was equal to 80% in computation to all the data collected, while the testing set was the remaining 20 % of the whole data.
•	Model Training: A Random Forest model was developed using the training data. Other more specific parameters of the model are the number of decision trees, the maximum depth of the trees, and the minimum sample size to split them were tuned to provide better accuracy.
•	Model Evaluation: The model was also evaluated to a test set and indicators that include accuracy, precision, recall, and F1-score were calculated for the evaluation of the model. Classification results were also illustrated further by the use of the confusion matrix.

4.4 Model Comparison
Apart from the Random Forest model, other models of machine learning algorithms were considered to check the efficiency; Logistic Regression to the Random Forest model, other machine learning algorithms were explored to compare their performance. These included:
•	Logistic Regression
•	K-Nearest Neighbors (KNN)
•	Gradient Boosting
In understanding the comparison of these models to the Random Forest Classifier, all of these models were trained and tested on the same data and their performance statistics are the same. Hyperparameters of each model were tuned by grid search in an attempt to optimize the models further.

4.5 Blockchain Implementation
To implement the decentralized framework, the following steps were taken:
•	Smart Contract Development: Blacklisted URLs and all other interactions were executed and managed using smart contracts written in Solidity. The contract was envisaged to help the users report nasty domains that required validation, and on the blockchain, the data was stored.
•	Deployment on Ganache: For the deployment of the smart contracts, Ganache – the Ethereum test blockchain – was used. This was quite handy in testing without having to make a deployment on the live blockchain platforms.
•	Web3 Integration: The Ethereum blockchain was used for this project with the help of the Web3.py module for integration management. This made it possible for the Python application to post transactions to the blockchain while confirming and archiving the blacklisted URLs.

Data Storage Mechanism in Blockchain
The blockchain component uses smart contracts to store blacklisted domains in a decentralized, immutable ledger. This design prevents unauthorized alterations, as each update to the blacklist (e.g., a newly identified malicious domain) is permanently stored on-chain. The blockchain’s consensus mechanism further ensures that only verified malicious domains are added, upholding data integrity and trustworthiness.
Data Integrity and Security
Blockchain's decentralized structure ensures data integrity by making it tamper-resistant. Once a domain is flagged and added to the blacklist, it cannot be deleted or altered without network consensus. This feature is crucial for a project focused on cybersecurity, where preserving the history of flagged domains is critical for reliable and transparent data tracking.

4.6 User Interface Development
Using, Flask, HTML, and CSS a very basic, but easy-to-use web interface was created.
• Take into consideration the assessment outcomes as the domain name, IP address, URL, and the extent to which it uses an HTTPS connection.
• Determine if a given URL is to be blacklisted based on the kind of URL; these being either malicious or benign. This allowed the Python application to submit transactions to the blockchain, verifying and storing the blacklisted URLs.
A simple web-based interface was developed using Flask, HTML, and CSS. The user interface allowed users to:
•	Enter a URL for analysis.
•	View the analysis results, including domain name, IP address, URL length, and HTTPS status.
•	Decide whether to blacklist a URL based on its classification (malicious or benign).
Backend requests were managed by the Flask framework, whereas the classification was done by the Random Forest classifier.

4.7 P2P Network Integration
Decentralized communication between nodes was facilitated by creating a peer-to-peer (P2P network). The steps included:
•	Node Creation: Ideally, any nod in the network could listen for peer connections and exchange information about the blacklisted domain.
•	Peer Communication: The nodes used sockets to convey information with each other, directed at new blacklisted domains so that all the nodes were aware of the change.

4.8 Testing and Validation
Testing was performed at several levels:
•	Unit Testing: In this method each component including the machine learning classifier, the blockchain smart contracts, and the P2P networking elements were also validated separately.
•	Integration Testing: The final design was evaluated by testing the entire application, from the user interface, the newly developed machine learning model, and the blockchain.
•	Performance Testing: The proposed system was evaluated using sample URLs containing various types of Web pages to validate its correct categorization as well as the inter-node communication in the P2P network.
This methodology provided for a systematic approach to providing the solution to the problem of identifying and black listing the malicious domains which as a result of the analysis were found to belong to the IP address and using decentralization technology.


Chapter 05: Results and Discussion

In this chapter, the results of the machine learning model developed for identifying and categorizing domain names as either benign or malicious are presented, along with a comprehensive analysis and interpretation of the findings. This section will examine the model's performance metrics, evaluate its effectiveness, and discuss the implications of these results. Key metrics like precision, recall, and F1-score are explored in detail, along with the confusion matrix to better understand the model’s classification behavior. Additionally, user interfaces will be discussed, providing insight into how users interact with the model's predictions. Figures are included to support the analysis.

5.1 Model Performance Metrics
5.1.1 Accuracy, Precision, Recall, and F1-Score
The performance metrics obtained from the Random Forest Classifier model are summarized in the classification report:
 
Figure 5.1: Model Classification Report
•	Accuracy: The model achieved an accuracy of 85%, which represents the proportion of total predictions that were correct.
•	Precision: Precision for classifying benign domains is 0.81, while for malicious domains, it is 0.91. This indicates that the model is generally precise in identifying true positives, particularly for malicious domains.
•	Recall: The recall for benign domains is 0.92, indicating a high sensitivity to benign cases, whereas the recall for malicious domains is 0.78.
•	F1-Score: The F1-scores for benign and malicious domains are 0.86 and 0.84, respectively. The F1-score combines precision and recall, offering a balanced metric to evaluate model performance in cases of class imbalance.
These metrics highlight that the model is particularly effective at detecting benign domains, though there is room for improvement in its ability to recall malicious domains. Improving recall for malicious domains could reduce the occurrence of false negatives, which is crucial for applications requiring high security.

5.2 Confusion Matrix Analysis

•	True Positives (TP): 1,227 benign domains were correctly classified as benign.
•	False Positives (FP): 105 benign domains were incorrectly classified as malicious.
•	True Negatives (TN): 1,023 malicious domains were correctly classified as malicious.
•	False Negatives (FN): 297 malicious domains were incorrectly classified as benign.
The confusion matrix indicates that while the model has high accuracy, false negatives (i.e., malicious domains incorrectly classified as benign) remain a significant issue. False negatives in security applications can lead to serious risks, as they may allow potentially harmful domains to be overlooked. In contrast, false positives (benign domains classified as malicious) may result in unnecessary restrictions on legitimate domains.
The false positive and false negative rates should be carefully monitored, especially if the model is deployed in real-world scenarios where security and user experience are critical. By reducing false negatives, the model’s effectiveness in identifying threats can be enhanced, contributing to a more secure system.

Model Comparison:
Nonetheless, when using additional models such as Logistic Regression, KNN, Gradient Boosting, the model with better accuracy and generalization ability was identified to be Random Forest. The Gradient Boosting Classifier provided the best results though it was inferior to all in terms of overall classification accuracy and F1-score.

5.3 Custom Threshold Evaluation
The model was evaluated with a custom threshold of 0.8484 to balance sensitivity and specificity. Adjusting the threshold can shift the model’s focus either toward minimizing false positives or false negatives, depending on the requirements. After applying this threshold, the model retained an accuracy of 85%, demonstrating that threshold adjustments offer an avenue for performance optimization.

5.4 Comparison with Existing Solutions
Discussing results in the context of existing research is crucial to highlight the advantages and limitations of the proposed approach. Existing models often struggle with high false negative rates, particularly in cases involving domain categorization where malicious patterns are complex and diverse. In comparison, this model attempts to balance precision and recall effectively, though it still encounters challenges with accurately identifying all malicious domains.
The model's use of SMOTE (Synthetic Minority Over-sampling Technique) helped address the class imbalance, a common limitation in previous solutions. However, further improvements could involve more advanced oversampling or ensemble techniques to refine performance in minority classes.

5.5 User Interface and Interaction
The interface allows users to input domain links, view the predicted classification, and decide whether to blacklist flagged domains. By incorporating a user-driven element, the system becomes adaptable and allows for human oversight, enhancing the model's reliability in practice.

5.6 Discussion of Findings
The results indicate that the model provides a strong foundation for detecting malicious domains, achieving an accuracy of 85% with a balance between precision and recall. However, the model’s limitations, particularly in false negatives, suggest that further refinement is necessary for real-world deployment.

In conclusion, the findings demonstrate that while the model is capable of identifying malicious domains with moderate success, future work should focus on optimizing recall for malicious domains to minimize security risks. Integrating additional features, refining data preprocessing, or employing more sophisticated models may enhance detection capabilities further

References (IEEE Style)

1.	Zou, J., Han, Y., Soewito, B., and Tamma, R., “Phishing Detection Using Machine Learning Algorithms,” IEEE Access, vol. 9, pp. 66029-66041, 2021. DOI: 10.1109/ACCESS.2021.3075695.
2.	S. Soni, K. Kumar, and S. Arora, "A Machine Learning-Based Approach to Detect Malicious Domains Using WHOIS and DNS Features," IEEE Access, vol. 8, pp. 33595-33605, 2020, doi: 10.1109/ACCESS.2020.2973976.
3.	N. S. Takur, P. P. Mujumdar, and A. S. Sharma, "Classification of Malicious URLs Using Machine Learning Techniques," in 2021 International Conference on Artificial Intelligence and Machine Vision (AIMV), Pune, India, 2021, pp. 195-200, doi: 10.1109/AIMV53313.2021.00042.
4.	A. Saxe and K. Berlin, "Deep neural network-based malware detection using two-dimensional binary program features," in 2015 10th International Conference on Malicious and Unwanted Software (MALWARE), Fajardo, Puerto Rico, 2015, pp. 11-20, doi: 10.1109/MALWARE.2015.7413680.
5.	J. Ma, L. K. Saul, S. Savage, and G. M. Voelker, "Beyond blacklists: Learning to detect malicious web sites from suspicious URLs," in Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '09), Paris, France, 2009, pp. 1245-1254, doi: 10.1145/1557019.1557153.
6.	J. Soska and N. Christin, "Automatically Detecting Vulnerable Websites Before They Turn Malicious," in Proceedings of the 23rd USENIX Security Symposium, San Diego, CA, USA, 2014, pp. 625-640.
7.	M. Stringhini, C. Kruegel, and G. Vigna, "Detecting Spammers on Social Networks," in Proceedings of the 26th Annual Computer Security Applications Conference (ACSAC '10), Austin, TX, USA, 2010, pp. 1-9, doi: 10.1145/1920261.1920263.
8.	S. Marchal, J. Francois, R. State, and T. Engel, "PhishStorm: Detecting Phishing With Streaming Analytics," IEEE Transactions on Network and Service Management, vol. 11, no. 4, pp. 458-471, Dec. 2014, doi: 10.1109/TNSM.2014.2362102.
9.	Z. A. Baig, A. M. Khan, and H. Iqbal, "A survey on machine learning techniques for cyber security in the last decade," IEEE Access, vol. 9, pp. 137340-137379, 2021, doi: 10.1109/ACCESS.2021.3118321.

