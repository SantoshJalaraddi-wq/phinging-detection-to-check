import re
import pickle
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'url_length', 'num_dots', 'num_hyphens', 'num_at',
            'num_question_marks', 'num_equals', 'num_slashes',
            'num_www', 'num_percent', 'num_plus', 'num_dollar',
            'has_https', 'has_ip', 'url_depth', 'num_digits',
            'num_letters', 'num_special_chars', 'domain_length',
            'path_length', 'has_subdomain', 'num_subdomains',
            'has_login', 'has_secure', 'num_parameters'
        ]
    
    def extract_features(self, url):
        """Extract 24 features from a URL"""
        features = []
        
        # Basic URL features
        features.append(len(url))  # url_length
        features.append(url.count('.'))  # num_dots
        features.append(url.count('-'))  # num_hyphens
        features.append(url.count('@'))  # num_at
        features.append(url.count('?'))  # num_question_marks
        features.append(url.count('='))  # num_equals
        features.append(url.count('/'))  # num_slashes
        features.append(url.count('www'))  # num_www
        features.append(url.count('%'))  # num_percent
        features.append(url.count('+'))  # num_plus
        features.append(url.count('$'))  # num_dollar
        
        # HTTPS check
        features.append(1 if 'https' in url else 0)  # has_https
        
        # IP address check
        parsed = urlparse(url)
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        features.append(1 if re.search(ip_pattern, url) else 0)  # has_ip
        
        # URL depth
        features.append(url.count('/') - 2 if url.count('/') > 2 else 0)  # url_depth
        
        # Character counts
        features.append(sum(c.isdigit() for c in url))  # num_digits
        features.append(sum(c.isalpha() for c in url))  # num_letters
        features.append(sum(not c.isalnum() for c in url))  # num_special_chars
        
        # Domain and path analysis
        domain = parsed.netloc
        path = parsed.path
        features.append(len(domain))  # domain_length
        features.append(len(path))  # path_length
        
        # Subdomain analysis
        subdomains = domain.split('.')
        features.append(1 if len(subdomains) > 2 else 0)  # has_subdomain
        features.append(len(subdomains) - 2 if len(subdomains) > 2 else 0)  # num_subdomains
        
        # Keyword presence
        features.append(1 if 'login' in url.lower() else 0)  # has_login
        features.append(1 if 'secure' in url.lower() else 0)  # has_secure
        
        # Parameters
        features.append(url.count('&') + 1 if '?' in url else 0)  # num_parameters
        
        return features
    
    def train(self, X, y):
        """Train the Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X, y)
        return self.model
    
    def predict(self, url):
        """Predict if a URL is phishing or legitimate"""
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        features = self.extract_features(url)
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probability = self.model.predict_proba(features_array)[0]
        
        return {
            'url': url,
            'is_phishing': bool(prediction),
            'confidence': float(probability[prediction]),
            'label': 'Phishing' if prediction == 1 else 'Legitimate'
        }
    
    def save_model(self, filename='phishing_model.pkl'):
        """Save the trained model"""
        if self.model:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {filename}")
    
    def load_model(self, filename='phishing_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filename}")


def create_sample_dataset():
    """Create a sample dataset for training"""
    # Sample URLs (you should use a real dataset like PhishTank)
    legitimate_urls = [
        'https://www.google.com',
        'https://www.facebook.com',
        'https://www.amazon.com',
        'https://www.microsoft.com',
        'https://www.apple.com',
        'https://www.github.com',
        'https://www.linkedin.com',
        'https://www.twitter.com',
        'https://www.youtube.com',
        'https://www.netflix.com'
    ]
    
    phishing_urls = [
        'http://www.paypal-secure-login.com/verify',
        'http://amaz0n-deals.xyz/offer',
        'http://www.faceb00k-login.net/secure',
        'http://192.168.1.1/login',
        'http://www.g00gle-verify.com/account',
        'http://secure-banking-update.tk/login',
        'http://www.micros0ft-support.xyz/help',
        'http://apple-id-verify.com/secure',
        'http://www.paypa1.com/verify-account',
        'http://netf1ix-billing.com/update'
    ]
    
    # Create dataset
    detector = PhishingDetector()
    X = []
    y = []
    
    print("Processing legitimate URLs...")
    for url in legitimate_urls:
        features = detector.extract_features(url)
        X.append(features)
        y.append(0)  # 0 = legitimate
    
    print("Processing phishing URLs...")
    for url in phishing_urls:
        features = detector.extract_features(url)
        X.append(features)
        y.append(1)  # 1 = phishing
    
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("PHISHING LINK DETECTOR")
    print("=" * 60)
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Create and train on sample data
    print("\n[1] Creating sample dataset...")
    X, y = create_sample_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    print("\n[2] Training Random Forest model...")
    detector.train(X_train, y_train)
    
    # Evaluate
    print("\n[3] Evaluating model...")
    y_pred = detector.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Phishing']))
    
    # Save model
    detector.save_model()
    
    # Test with new URLs
    print("\n" + "=" * 60)
    print("TESTING NEW URLs")
    print("=" * 60)
    
    test_urls = [
        'https://www.google.com',
        'http://www.paypal-secure-login.com/verify',
        'https://www.amazon.com/deals',
        'http://amaz0n-deals.xyz/offer'
    ]
    
    for url in test_urls:
        result = detector.predict(url)
        status = "⚠️  PHISHING" if result['is_phishing'] else "✓ Legitimate"
        confidence = result['confidence'] * 100
        print(f"\nURL: {url}")
        print(f"Status: {status}")
        print(f"Confidence: {confidence:.2f}%")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        url = input("\nEnter URL to check: ").strip()
        if url.lower() in ['quit', 'exit', 'q']:
            break
        
        if not url:
            continue
        
        try:
            result = detector.predict(url)
            status = "⚠️  PHISHING DETECTED!" if result['is_phishing'] else "✓ Legitimate URL"
            confidence = result['confidence'] * 100
            
            print(f"\nResult: {status}")
            print(f"Confidence: {confidence:.2f}%")
            
            if result['is_phishing']:
                print("⚠️  WARNING: This URL appears to be malicious!")
                print("   Do not enter sensitive information on this site.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()