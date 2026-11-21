import os

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

tf.get_logger().setLevel('ERROR')

CONFIANCA_MINIMA_EMOCAO = 0.10

mapa_de_emocoes = {
    'angry': 'raiva',
    'disgust': 'nojo',
    'fear': 'medo',
    'happy': 'felicidade',
    'sad': 'tristeza',
    'surprise': 'surpresa',
    'neutral': 'neutro'
}

cores_de_emocoes = {
    'felicidade': (0, 255, 0),    # Verde
    'tristeza': (255, 0, 0),      # Azul
    'raiva': (0, 0, 255),         # Vermelho
    'medo': (128, 0, 128),        # Roxo
    'nojo': (0, 165, 255),        # Laranja
    'surpresa': (0, 255, 255),    # Amarelo
    'neutro': (128, 128, 128)     # Cinza
}

def analisar_emocao(rosto_recortado):
    try:
        if rosto_recortado is None or rosto_recortado.size == 0:
            return "neutro", 0.0
        
        rosto_rgb = cv2.cvtColor(rosto_recortado, cv2.COLOR_BGR2RGB)
        
        resultado = DeepFace.analyze(
            img_path=rosto_rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='skip'
        )
        
        if resultado and len(resultado) > 0:
            emocao = resultado[0]['dominant_emotion']
            confianca = resultado[0]['emotion'][emocao] / 100
            
            emocao_pt = mapa_de_emocoes.get(emocao, emocao)
            
            return emocao_pt, confianca
        
        return "neutro", 0.0
    
    except Exception as e:
        return "neutro", 0.0

def cor_emocao(emocao):
    return cores_de_emocoes.get(emocao, (255, 255, 255))

def desenhar_emocao(frame, rosto, emocao, confianca, nome=None):
    x, y, w, h = rosto
    cor = cor_emocao(emocao)
    primeiro_nome = nome.split()[0].lower() if nome else None
    
    if primeiro_nome is not None and confianca > CONFIANCA_MINIMA_EMOCAO:
        texto = f"{primeiro_nome} - {emocao} ({confianca*100:.0f}%)"
    elif confianca > CONFIANCA_MINIMA_EMOCAO:
        texto = f"{emocao} ({confianca*100:.0f}%)"
    else:
        texto = emocao
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
    
    cv2.putText(frame, texto, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    return frame

def obter_estatisticas(emocoes):
    stats = {}
    for emocao in emocoes:
        stats[emocao] = stats.get(emocao, 0) + 1
    return stats