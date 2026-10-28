from flask import Flask, request, jsonify, render_template_string
from phishing_detector import PhishingDetector
import numpy as np

app = Flask(__name__)
detector = PhishingDetector()

# Load pre-trained model
try:
    detector.load_model('phishing_model.pkl')
except:
    print("No pre-trained model found. Please train first.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Link Detector</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        h1 { color: #333; text-align: center; }
        input[type="text"] { width: 100%; padding: 15px; margin: 10px 0; 
                            border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
        button { background: #4CAF50; color: white; padding: 15px 30px; 
                border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
        button:hover { background: #45a049; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; }
        .safe { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .confidence { font-size: 18px; font-weight: bold; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Phishing Link Detector</h1>
        <p>Enter a URL to check if it's legitimate or potentially malicious:</p>
        
        <form id="checkForm">
            <input type="text" id="urlInput" placeholder="https://example.com" required>
            <button type="submit">Check URL</button>
        </form>
        
        <div id="result" class="result" style="display: none;">
            <h3 id="resultTitle"></h3>
            <p id="resultMessage"></p>
            <div class="confidence" id="confidenceScore"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('checkForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const url = document.getElementById('urlInput').value;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url})
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                const resultTitle = document.getElementById('resultTitle');
                const resultMessage = document.getElementById('resultMessage');
                const confidenceScore = document.getElementById('confidenceScore');
                
                resultDiv.style.display = 'block';
                
                if (data.is_phishing) {
                    resultDiv.className = 'result danger';
                    resultTitle.textContent = '⚠️ PHISHING DETECTED!';
                    resultMessage.textContent = 'This URL appears to be malicious. Do not enter sensitive information.';
                } else {
                    resultDiv.className = 'result safe';
                    resultTitle.textContent = '✓ Legitimate URL';
                    resultMessage.textContent = 'This URL appears to be safe.';
                }
                
                confidenceScore.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        result = detector.predict(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for programmatic access"""
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    result = detector.predict(url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)