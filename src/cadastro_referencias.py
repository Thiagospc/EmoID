import cv2
import os
import pickle
import numpy as np
from reconhecimento_facial import (
    carregar_detector,
    detectar_rosto_referencia,
    extrair_embedding,
    recortar_rosto,
)

def processar_pasta_figures():  
    detector = carregar_detector()
    
    if detector is None:
        print("❌ Detector não carregado!")
        return
    
    pasta_figures = "figures/"
    
    if not os.path.exists(pasta_figures):
        print(f"❌ Pasta {pasta_figures} não encontrada!")
        return
    
    subpastas = [f for f in os.listdir(pasta_figures) 
                 if os.path.isdir(os.path.join(pasta_figures, f))]
    
    if not subpastas:
        print(f"❌ Nenhuma subpasta encontrada em {pasta_figures}")
        print("   Crie pastas com o nome de cada aluno dentro de figures/")
        return
    
    os.makedirs("data/referencias/", exist_ok=True)

    total_alunos = len(subpastas)
    for indice_aluno, aluno in enumerate(subpastas, 1):
        print(f"\r{indice_aluno}/{total_alunos} alunos", end="", flush=True)
        caminho_aluno = os.path.join(pasta_figures, aluno)
        nome_completo = aluno.replace('_', ' ').upper()
        
        arquivos = os.listdir(caminho_aluno)
        extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        imagens = [f for f in arquivos if f.lower().endswith(extensoes)]
        
        if not imagens:
            continue
        
        embeddings = []
        
        for imagem in imagens:
            caminho_imagem = os.path.join(caminho_aluno, imagem)
            foto = cv2.imread(caminho_imagem)
            if foto is None:
                continue
            
            rostos = detectar_rosto_referencia(foto, detector)
            
            if len(rostos) == 0:
                continue
            
            rosto = recortar_rosto(foto, rostos[0])
            
            embedding = extrair_embedding(rosto)
            
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) == 0:
            continue
        
        nome_arquivo = aluno.lower()
        caminho_pkl = f"data/referencias/{nome_arquivo}.pkl"
        nome_antigo = aluno.split('_')[0].lower()
        caminho_pkl_antigo = f"data/referencias/{nome_antigo}.pkl"
        if caminho_pkl_antigo != caminho_pkl and os.path.exists(caminho_pkl_antigo):
            os.remove(caminho_pkl_antigo)
        
        with open(caminho_pkl, 'wb') as f:
            pickle.dump(embeddings, f)
        

    print()

if __name__ == "__main__":
    processar_pasta_figures()
    print("\n✅ Referências geradas com sucesso!")