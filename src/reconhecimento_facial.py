import os
import pickle

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

_stderr_nativo = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(_devnull, 2)
    import cv2
    import numpy as np
    from deepface import DeepFace
    import tensorflow as tf
finally:
    os.dup2(_stderr_nativo, 2)
    os.close(_devnull)
    os.close(_stderr_nativo)

from analise_emocoes import analisar_emocao, desenhar_emocao, cor_emocao

tf.get_logger().setLevel('ERROR')

LIMITE_DISTANCIA = 0.55
CONFIANCA_MINIMA = 0.10
INTERVALO_EMOCAO = 15
INTERVALO_IDENTIDADE = 60

_ultimo_resultado = {}
_contador_frames = 0
_cache_embeddings = {}

def obter_ultimos_resultados():
    return _ultimo_resultado.copy()

def aquecer_modelos(caminho_imagem=None):
    try:
        DeepFace.build_model("Facenet")
        if caminho_imagem and os.path.exists(caminho_imagem):
            imagem = cv2.imread(caminho_imagem)
            if imagem is not None:
                analisar_emocao(imagem)
        return True
    except Exception:
        return False


def carregar_detector():
    possiveis_caminhos = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(__file__), '../haarcascades/haarcascade_frontalface_default.xml'),
        os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
        'haarcascade_frontalface_default.xml',
    ]
    
    for caminho in possiveis_caminhos:
        if os.path.exists(caminho):
            detector = cv2.CascadeClassifier(caminho)
            if not detector.empty():
                return detector
    
    return None

def detectar_rostos(frame, detector, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50)):
    if frame is None or detector is None:
        return []
    
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cinza = cv2.equalizeHist(cinza)
    
    rostos = detector.detectMultiScale(
        cinza,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    
    return rostos

def detectar_rosto_referencia(frame, detector):
    if frame is None or detector is None:
        return []

    deteccoes = detectar_rostos(frame, detector)
    if len(deteccoes) == 0:
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidatos = []
        for imagem in (cinza, cv2.equalizeHist(cinza)):
            candidatos.extend(
                detector.detectMultiScale(
                    imagem,
                    scaleFactor=1.03,
                    minNeighbors=3,
                    minSize=(30, 30),
                )
            )
        deteccoes = candidatos

    if len(deteccoes) == 0:
        return []

    return [max(deteccoes, key=lambda rosto: rosto[2] * rosto[3])]

def contar_rostos(frame, rostos):
    if frame is None:
        return frame
    
    cv2.putText(frame, f"Rostos: {len(rostos)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

def recortar_rosto(frame, rosto, margem=0.25):
    altura, largura = frame.shape[:2]
    x, y, w, h = rosto
    margem_x = int(w * margem)
    margem_y = int(h * margem)
    inicio_x = max(0, x - margem_x)
    inicio_y = max(0, y - margem_y)
    fim_x = min(largura, x + w + margem_x)
    fim_y = min(altura, y + h + margem_y)
    return frame[inicio_y:fim_y, inicio_x:fim_x]

def processar_rostos(frame, detector):
    if frame is None or detector is None:
        return frame, []
    
    rostos = detectar_rostos(frame, detector)
    
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Rosto", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    frame = contar_rostos(frame, rostos)
    
    return frame, rostos

def extrair_embedding(imagem):
    try:
        if imagem is None or imagem.size == 0:
            return None
        
        if imagem.shape[0] > 160 or imagem.shape[1] > 160:
            imagem = cv2.resize(imagem, (160, 160))
        
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        
        embedding = DeepFace.represent(
            img_path=imagem_rgb,
            model_name='Facenet',
            enforce_detection=False,
            detector_backend='skip'
        )
        
        vetor = np.array(embedding[0]['embedding'], dtype=np.float32)
        norma = np.linalg.norm(vetor)
        
        if norma > 0:
            vetor = vetor / norma
        
        return vetor
    
    except Exception as e:
        return None

def extrair_embedding_rosto(rosto_recortado):
    if rosto_recortado is None or rosto_recortado.size == 0:
        return None
    return extrair_embedding(rosto_recortado)

def carregar_referencias_salvas(pasta="data/referencias/"):
    referencias = {}
    
    if not os.path.exists(pasta):
        return referencias
    
    arquivos = os.listdir(pasta)
    pkls = [f for f in arquivos if f.endswith('.pkl')]
    
    if not pkls:
        return referencias
    
    print(f"📂 Carregando {len(pkls)} referência(s)...")
    
    for arquivo in pkls:
        nome = os.path.splitext(arquivo)[0]
        nome_exibicao = nome.replace('_', ' ').upper()
        caminho = os.path.join(pasta, arquivo)
        
        try:
            with open(caminho, 'rb') as f:
                referencia = pickle.load(f)
            referencias[nome_exibicao] = referencia
            print(f"   ✅ {nome_exibicao} carregado!")
        except Exception as e:
            print(f"   ❌ Falha ao carregar {nome}: {e}")
    
    return referencias

def reconhecer_pessoa(rosto_recortado, referencias):
    if rosto_recortado is None or not referencias:
        return None, 0.0
    
    embedding_atual = extrair_embedding_rosto(rosto_recortado)
    
    if embedding_atual is None:
        return None, 0.0
    
    melhor_nome = None
    menor_distancia = float('inf')
    
    for nome, embedding_referencia in referencias.items():
        if isinstance(embedding_referencia, list):
            distancias = [
                1 - np.dot(embedding_atual, embedding)
                for embedding in embedding_referencia
            ]
            distancia = min(distancias)
        else:
            distancia = 1 - np.dot(embedding_atual, embedding_referencia)
        
        if distancia < menor_distancia and distancia < LIMITE_DISTANCIA:
            menor_distancia = distancia
            melhor_nome = nome
    
    if melhor_nome is not None:
        confianca = max(0, 1 - (menor_distancia / LIMITE_DISTANCIA))
        return melhor_nome, confianca
    
    return None, 0.0

def processar_rostos_com_reconhecimento_e_emocao(frame, detector, referencias):
    global _ultimo_resultado, _contador_frames, _cache_embeddings
    
    if frame is None or detector is None:
        return frame, []
    
    rostos = detectar_rostos(frame, detector)
    
    if len(rostos) == 0:
        _ultimo_resultado = {}
        frame = contar_rostos(frame, rostos)
        return frame, rostos
    
    _contador_frames += 1
    
    if _contador_frames % INTERVALO_EMOCAO == 0:
        for i, (x, y, w, h) in enumerate(rostos):
            rosto_recortado = recortar_rosto(frame, (x, y, w, h))
            
            nome, confianca_nome, _, _ = _ultimo_resultado.get(
                i, (None, 0.0, "neutro", 0.0)
            )
            deve_reconhecer = (
                bool(referencias)
                and (
                    nome is None
                    or _contador_frames % INTERVALO_IDENTIDADE == 0
                )
            )
            if deve_reconhecer:
                nome, confianca_nome = reconhecer_pessoa(rosto_recortado, referencias)
            
            emocao, confianca_emocao = analisar_emocao(rosto_recortado)
            
            _ultimo_resultado[i] = (nome, confianca_nome, emocao, confianca_emocao)
    
    for i, (x, y, w, h) in enumerate(rostos):
        nome, conf_nome, emocao, conf_emocao = _ultimo_resultado.get(
            i, (None, 0.0, "neutro", 0.0)
        )
        
        if nome is not None and conf_nome > CONFIANCA_MINIMA:
            nome_exibir = nome
        else:
            nome_exibir = None
        
        frame = desenhar_emocao(frame, (x, y, w, h), emocao, conf_emocao, nome_exibir)
    
    frame = contar_rostos(frame, rostos)
    
    return frame, rostos