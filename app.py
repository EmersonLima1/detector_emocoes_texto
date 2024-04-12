from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
import nltk
nltk.download('stopwords')

# Caminho para os arquivos de modelos
model_path = 'models/my_model.h5'
count_vectorizer_path = 'models/CountVectorizer.pkl'
label_encoder_path = 'models/encoder.pkl'

# Carregando o modelo
model = load_model(model_path)

# Carregando o CountVectorizer
with open(count_vectorizer_path, 'rb') as f:
    count_vectorizer = pickle.load(f)

# Carregando o LabelEncoder
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Inicialize o aplicativo Flask
app = Flask(__name__)

# Inicialize o stemmer
ps = PorterStemmer()

# Função de pré-processamento
def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line) # Remova caracteres não alfabéticos
    review = review.lower() # Converta para minúsculas
    review = review.split() # Divida em palavras
    # Aplique stemming e remova palavras de parada
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

# Rota para a página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Rota para processar a frase submetida
@app.route('/detect', methods=['POST'])
def detect():
    # Obtenha a frase submetida pelo usuário
    phrase = request.form['phrase']
    
    # Pré-processar a frase
    processed_phrase = preprocess(phrase)
    
    # Transforme a frase usando CountVectorizer
    array = count_vectorizer.transform([processed_phrase]).toarray()
    
    # Faça a previsão usando o modelo carregado
    pred = model.predict(array)
    
    # Obtenha o índice da classe prevista
    predicted_class_index = np.argmax(pred, axis=1)
    
    # Obtenha a emoção correspondente ao índice previsto
    emotion = label_encoder.inverse_transform(predicted_class_index)[0]
    
    # Renderize a página com a emoção detectada
    return render_template('index.html', emotion=emotion)

# Executar o aplicativo
if __name__ == '__main__':
    app.run(debug=True)
