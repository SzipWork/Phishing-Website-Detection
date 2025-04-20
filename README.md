# Phishing-Website-Detection
This project focuses on detecting phishing websites using machine learning algorithms. By analyzing features extracted from URLs and website metadata, the model aims to classify websites as either phishing or legitimate.

## Dataset

The project utilizes several datasets that include labeled examples of phishing and benign websites. These datasets are used to train and evaluate the performance of the models. The primary datasets referenced in the notebook include:

- `/content/online-valid_ds.csv`
- `/content/urldata.csv`
- `phish_file.csv`
- `legit_file.csv`
- `Benign_url_file.csv`

## Feature Extraction

Features were engineered based on URL structure, domain information, and metadata, including:

- Presence of IP address in the URL
- Use of HTTPS protocol
- Length and complexity of the URL
- Use of suspicious symbols or characters
- Domain registration and expiration details (via WHOIS)
- Number of redirects and external links

## Dependencies

The project was developed using the following Python libraries:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `BeautifulSoup`, `requests`, `re`, `urllib`, `ipaddress`, `datetime`
- `whois`

To install the necessary dependencies, use:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn beautifulsoup4 python-whois
```


## Machine Learning Models
Multiple classification algorithms were implemented and compared to determine the most effective model, including:

- Logistic Regression

- Decision Tree Classifier

- Random Forest Classifier

- K-Nearest Neighbors (KNN)

- Support Vector Classifier (SVC)

- Gradient Boosting Classifier

- Voting Classifier (Ensemble)

- Stacking Classifier (Ensemble)


## Evaluation Metrics
Model performance was assessed using standard classification metrics:

- Accuracy Score

- Confusion Matrix

- Classification Report (Precision, Recall, F1-score)


## Results
Ensemble techniques, particularly the Voting and Stacking Classifiers, demonstrated superior performance in terms of classification accuracy and robustness, highlighting the advantages of combining multiple models.


## Usage Instructions
- Clone this repository or download the notebook file.

- Ensure all required datasets are placed in the correct file paths as specified in the notebook.

- Run the Jupyter Notebook PhishingWebsiteDetection.ipynb in an appropriate environment.

- Evaluate the model performance and modify the feature set or classifiers as desired.


## Contributing
Feel free to submit issues or pull requests to enhance this project.
