from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import json

# Configuração inicial
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['CRITERIA_FILE'] = 'criteria.json'

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar diretório de uploads se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configurar Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Listar modelos disponíveis para debug
        models = genai.list_models()
        available_models = [model.name for model in models]
        logger.info(f"Modelos disponíveis: {available_models}")
        
        # Usar modelo disponível
        if 'models/gemini-1.5-pro' in available_models:
            GEMINI_MODEL = 'gemini-1.5-pro'
        elif 'models/gemini-pro' in available_models:
            GEMINI_MODEL = 'gemini-pro'
        elif 'models/gemini-1.0-pro' in available_models:
            GEMINI_MODEL = 'gemini-1.0-pro'
        else:
            GEMINI_MODEL = None
            logger.warning("Nenhum modelo Gemini Pro encontrado")
            
        logger.info(f"Usando modelo: {GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"Erro ao configurar Gemini: {e}")
        GEMINI_MODEL = None
else:
    logger.warning("GEMINI_API_KEY não encontrada. Usando apenas classificação por critérios.")
    GEMINI_MODEL = None

# Baixar recursos do NLTK
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

class CriteriaManager:
    def __init__(self, criteria_file):
        self.criteria_file = criteria_file
        self.default_criteria = {
            "productive_keywords": [
                "reunião", "projeto", "trabalho", "relatório", "prazo", "cliente",
                "entrega", "solicitação", "orçamento", "contrato", "desenvolvimento",
                "apresentação", "análise", "negócio", "empresa", "equipe", "gestão"
            ],
            "unproductive_keywords": [
                "spam", "promoção", "oferta", "desconto", "newsletter", "corrente",
                "loteria", "prêmio", "ganhador", "grátis", "urgente", "importante"
            ],
            "required_patterns": [
                r"\b(projeto|tarefa|atividade)\b.*\b(prazo|data|entrega)\b",
                r"\b(reunião|encontro)\b.*\b(agenda|agendar|marcar)\b",
                r"\b(relatório|documento|análise)\b.*\b(entregar|enviar|preparar)\b"
            ],
            "min_length": 50,
            "max_length": 5000,
            "productive_weight": 2,
            "unproductive_weight": 3
        }
        self.load_criteria()
    
    def load_criteria(self):
        """Carrega critérios do arquivo ou usa os padrões"""
        try:
            if os.path.exists(self.criteria_file):
                with open(self.criteria_file, 'r', encoding='utf-8') as f:
                    self.criteria = json.load(f)
                logger.info("Critérios carregados do arquivo")
            else:
                self.criteria = self.default_criteria
                self.save_criteria()
                logger.info("Usando critérios padrão")
        except Exception as e:
            logger.error(f"Erro ao carregar critérios: {e}")
            self.criteria = self.default_criteria
    
    def save_criteria(self):
        """Salva critérios no arquivo"""
        try:
            with open(self.criteria_file, 'w', encoding='utf-8') as f:
                json.dump(self.criteria, f, ensure_ascii=False, indent=2)
            logger.info("Critérios salvos com sucesso")
        except Exception as e:
            logger.error(f"Erro ao salvar critérios: {e}")
    
    def update_criteria(self, new_criteria):
        """Atualiza critérios"""
        self.criteria.update(new_criteria)
        self.save_criteria()
    
    def get_criteria(self):
        """Retorna critérios atuais"""
        return self.criteria

class EmailProcessor:
    def __init__(self, criteria_manager):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('portuguese'))
        self.criteria_manager = criteria_manager
        
    def preprocess_text(self, text):
        """Pré-processa o texto do email"""
        if not text:
            return ""
            
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Záéíóúâêîôûãõç\s]', '', text)
        
        # Tokenização
        tokens = word_tokenize(text, language='portuguese')
        
        # Remover stop words e aplicar stemming
        filtered_tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(filtered_tokens)
    
    def extract_text_from_pdf(self, file_path):
        """Extrai texto de arquivos PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF: {e}")
            return None
    
    def extract_text_from_txt(self, file_path):
        """Extrai texto de arquivos TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Erro ao ler arquivo TXT: {e}")
            return None

class EmailClassifier:
    def __init__(self, criteria_manager):
        self.processor = EmailProcessor(criteria_manager)
        self.criteria_manager = criteria_manager
    
    def analyze_with_criteria(self, text):
        """Analisa o email baseado nos critérios definidos"""
        criteria = self.criteria_manager.get_criteria()
        text_lower = text.lower()
        
        score = 0
        reasons = []
        details = {}
        
        # Usar pesos configuráveis
        productive_weight = criteria.get('productive_weight', 2)
        unproductive_weight = criteria.get('unproductive_weight', 3)
        
        # 1. Verificar palavras-chave produtivas
        productive_matches = []
        for keyword in criteria['productive_keywords']:
            if keyword in text_lower:
                productive_matches.append(keyword)
                score += productive_weight
        
        if productive_matches:
            reasons.append(f"Palavras-chave produtivas encontradas: {', '.join(productive_matches[:5])}")
            details['productive_keywords_found'] = productive_matches
        
        # 2. Verificar palavras-chave improdutivas
        unproductive_matches = []
        for keyword in criteria['unproductive_keywords']:
            if keyword in text_lower:
                unproductive_matches.append(keyword)
                score -= unproductive_weight
        
        if unproductive_matches:
            reasons.append(f"Palavras-chave improdutivas encontradas: {', '.join(unproductive_matches[:5])}")
            details['unproductive_keywords_found'] = unproductive_matches
        
        # 3. Verificar padrões requeridos
        pattern_matches = []
        for pattern in criteria['required_patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                pattern_matches.append(pattern)
                score += 5
        
        if pattern_matches:
            reasons.append(f"Padrões produtivos identificados: {len(pattern_matches)}")
            details['patterns_matched'] = pattern_matches
        
        # 4. Verificar comprimento do texto
        text_length = len(text)
        if text_length < criteria['min_length']:
            score -= 5
            reasons.append(f"Texto muito curto ({text_length} caracteres)")
        elif text_length > criteria['max_length']:
            score -= 2
            reasons.append(f"Texto muito longo ({text_length} caracteres)")
        else:
            score += 2
            reasons.append(f"Comprimento adequado do texto ({text_length} caracteres)")
        
        details['text_length'] = text_length
        
        # 5. Verificar se parece exigir ação
        action_indicators = ['preciso', 'necessito', 'solicito', 'por favor', 'urgente', 'importante']
        action_found = any(indicator in text_lower for indicator in action_indicators)
        
        if action_found:
            score += 3
            reasons.append("Email parece exigir ação/resposta")
            details['requires_action'] = True
        
        # 6. Verificar contexto de negócio
        business_indicators = ['empresa', 'negócio', 'cliente', 'projeto', 'equipe', 'gestão']
        business_context = any(indicator in text_lower for indicator in business_indicators)
        
        if business_context:
            score += 3
            reasons.append("Contexto de negócio identificado")
            details['business_context'] = True
        
        # Determinar classificação baseada no score
        if score >= 5:
            classification = "Produtivo"
        elif score >= 0:
            classification = "Neutro"
        else:
            classification = "Improdutivo"
        
        return {
            'classification': classification,
            'score': score,
            'reasons': reasons,
            'details': details
        }
    
    def classify_with_gemini(self, text):
        """Classifica o email usando Gemini AI"""
        try:
            if not GEMINI_MODEL:
                raise Exception("Nenhum modelo Gemini disponível")
                
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            prompt = f"""
            Analise este email e classifique como "Produtivo" ou "Improdutivo".
            
            CRITÉRIOS:
            - Produtivo: Emails relacionados a trabalho, negócios, projetos, solicitações importantes, reuniões, prazos
            - Improdutivo: Spam, newsletters não solicitadas, emails pessoais irrelevantes, propagandas, correntes
            
            EMAIL:
            {text[:2000]}
            
            Responda APENAS com uma das duas palavras: "Produtivo" ou "Improdutivo"
            """
            
            response = model.generate_content(prompt)
            classification = response.text.strip()
            
            return classification if classification in ["Produtivo", "Improdutivo"] else "Produtivo"
            
        except Exception as e:
            logger.error(f"Erro na classificação com Gemini: {e}")
            raise
    
    def classify_email(self, text):
        """Classifica o email usando critérios ou Gemini AI"""
        if not text:
            return {
                'classification': 'Improdutivo',
                'score': -10,
                'reasons': ['Texto vazio ou não extraído'],
                'details': {}
            }
        
        # Primeiro, análise com critérios
        criteria_analysis = self.analyze_with_criteria(text)
        
        # Se tiver Gemini API, usar para validação adicional
        if GEMINI_MODEL and GEMINI_API_KEY:
            try:
                gemini_classification = self.classify_with_gemini(text)
                # Se houver discordância significativa, reconsiderar
                if (gemini_classification == "Produtivo" and 
                    criteria_analysis['classification'] in ["Neutro", "Improdutivo"] and
                    criteria_analysis['score'] >= -2):
                    criteria_analysis['classification'] = "Produtivo"
                    criteria_analysis['reasons'].append("Reclassificado pelo Gemini AI")
                
                criteria_analysis['gemini_validation'] = gemini_classification
            except Exception as e:
                logger.error(f"Erro na validação Gemini: {e}")
        
        return criteria_analysis
    
    def generate_response(self, email_text, classification, analysis_details):
        """Gera resposta automática baseada na classificação e análise"""
        try:
            if GEMINI_MODEL and GEMINI_API_KEY:
                model = genai.GenerativeModel(GEMINI_MODEL)
                
                prompt = f"""
                Com base neste email classificado como "{classification}", gere uma resposta profissional e adequada em português.
                
                DETALHES DA ANÁLISE:
                - Score: {analysis_details.get('score', 0)}
                - Razões: {', '.join(analysis_details.get('reasons', []))}
                
                EMAIL ORIGINAL:
                {email_text[:1500]}
                
                INSTRUÇÕES:
                - Seja profissional e educado
                - Mantenha a resposta concisa (máximo 100 palavras)
                - Adapte o tom conforme a classificação
                - Não inclua marcadores ou formatação
                - Responda em português brasileiro
                """
                
                response = model.generate_content(prompt)
                return response.text.strip()
            else:
                return self.generate_fallback_response(classification, analysis_details)
                
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return self.generate_fallback_response(classification, analysis_details)
    
    def generate_fallback_response(self, classification, analysis_details):
        """Gera resposta automática fallback"""
        reasons = analysis_details.get('reasons', [])
        
        if classification == "Produtivo":
            return "Agradecemos seu email produtivo. Analisaremos sua solicitação com prioridade e retornaremos em breve. Nossa equipe já está trabalhando na solução."
        elif classification == "Neutro":
            return "Obrigado pelo seu contato. Analisaremos sua mensagem e retornaremos caso necessário. Agradecemos pela compreensão."
        else:
            return "Obrigado pelo seu contato. Identificamos que este email não requer uma ação específica de nossa equipe no momento."

# Instâncias globais
criteria_manager = CriteriaManager(app.config['CRITERIA_FILE'])
classifier = EmailClassifier(criteria_manager)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_email():
    try:
        email_text = ""
        file = None
        
        # Verificar se foi enviado arquivo
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Extrair texto baseado no tipo de arquivo
                    if filename.lower().endswith('.pdf'):
                        email_text = classifier.processor.extract_text_from_pdf(filepath)
                    else:
                        email_text = classifier.processor.extract_text_from_txt(filepath)
                    
                    # Limpar arquivo temporário
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    
                    if not email_text:
                        return jsonify({'error': 'Não foi possível extrair texto do arquivo'}), 400
                else:
                    return jsonify({'error': 'Tipo de arquivo não permitido. Use .txt ou .pdf'}), 400
        
        # Verificar se foi enviado texto direto
        if not email_text and 'text' in request.form:
            email_text = request.form['text'].strip()
        
        if not email_text:
            return jsonify({'error': 'Nenhum conteúdo de email fornecido'}), 400
        
        # Pré-processar texto
        processed_text = classifier.processor.preprocess_text(email_text)
        
        # Classificar email
        analysis_result = classifier.classify_email(email_text)
        
        # Gerar resposta automática
        response = classifier.generate_response(email_text, analysis_result['classification'], analysis_result)
        
        return jsonify({
            'classification': analysis_result['classification'],
            'score': analysis_result['score'],
            'reasons': analysis_result['reasons'],
            'details': analysis_result.get('details', {}),
            'response': response,
            'processed_text': processed_text[:500] + '...' if len(processed_text) > 500 else processed_text
        })
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        return jsonify({'error': 'Erro interno no processamento do email'}), 500

@app.route('/criteria', methods=['GET', 'POST'])
def manage_criteria():
    """Endpoint para gerenciar critérios"""
    if request.method == 'GET':
        return jsonify(criteria_manager.get_criteria())
    else:
        try:
            new_criteria = request.get_json()
            criteria_manager.update_criteria(new_criteria)
            return jsonify({'success': True, 'message': 'Critérios atualizados com sucesso'})
        except Exception as e:
            logger.error(f"Erro ao atualizar critérios: {e}")
            return jsonify({'error': 'Erro ao atualizar critérios'}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf'}

# Configuração para produção
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)