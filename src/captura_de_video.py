import cv2
from datetime import datetime
import json
import os
import pickle
import numpy as np
from reconhecimento_facial import (
    carregar_detector,
    processar_rostos,
    processar_rostos_com_reconhecimento_e_emocao,
    carregar_referencias_salvas,
    detectar_rostos,
    detectar_rosto_referencia,
    extrair_embedding,
    recortar_rosto,
    aquecer_modelos,
    obter_ultimos_resultados,
)

def gerar_referencias_da_pasta(pasta="figures/"):    
    detector = carregar_detector()
    
    if detector is None:
        return {}
    
    if not os.path.exists(pasta):
        return {}
    
    subpastas = [f for f in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, f))]
    
    if not subpastas:
        return {}
    
    os.makedirs("data/referencias/", exist_ok=True)
    
    referencias = {}
    
    total_alunos = len(subpastas)
    for indice_aluno, aluno in enumerate(subpastas, 1):
        print(f"\r{indice_aluno}/{total_alunos} alunos", end="", flush=True)
        caminho_aluno = os.path.join(pasta, aluno)
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
        
        referencias[nome_completo] = embeddings
    print()
    return referencias

def main():
    detector = carregar_detector()
    if detector is None:
        print("❌ Detector facial não encontrado.")
        return None

    referencias = carregar_referencias_salvas("data/referencias/")
    if not referencias:
        referencias = gerar_referencias_da_pasta("figures/")

        if not referencias:
            print("⚠️ Nenhuma pessoa registrada. Modo DETECÇÃO apenas.")
    else:
        resposta = input("\n🔁 Deseja recarregar as referências das fotos? (s/n): ").strip().lower()
        if resposta == 's':
            referencias = gerar_referencias_da_pasta("figures/")

    primeira_imagem = next(
        (caminho for caminho in [
            os.path.join("figures", pasta, arquivo)
            for pasta in os.listdir("figures")
            if os.path.isdir(os.path.join("figures", pasta))
            for arquivo in os.listdir(os.path.join("figures", pasta))
            if arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]),
        None,
    ) if os.path.exists("figures") else None
    print("🔄 Preparando modelos de reconhecimento...")
    aquecer_modelos(primeira_imagem)

    nome_arquivo = f"data/gravacao_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
    fps = 20.0
    resolucao = (640, 480)
    captura = cv2.VideoCapture(0)
    nome_janela = "Leitor de emoções"

    if not captura.isOpened():
        print("❌ Erro ao acessar webcam.")
        return None

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 960, 720)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao[0])
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao[1])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    gravador = cv2.VideoWriter(nome_arquivo, fourcc, fps, resolucao)
    caminho_eventos = os.path.splitext(nome_arquivo)[0] + "_eventos.jsonl"
    ultimo_evento_por_rosto = {}
    frame_count = 0

    try:
        with open(caminho_eventos, "w", encoding="utf-8") as arquivo_eventos:
            while True:
                retorno, frame = captura.read()

                if not retorno:
                    print("❌ Erro na captura do frame.")
                    break

                frame_count += 1
                frame = cv2.flip(frame, 1)
                if referencias:
                    frame, rostos = processar_rostos_com_reconhecimento_e_emocao(
                        frame, detector, referencias
                    )
                else:
                    frame, rostos = processar_rostos(frame, detector)

                for indice, resultado in obter_ultimos_resultados().items():
                    nome, confianca_nome, emocao, confianca_emocao = resultado
                    evento_atual = (nome, emocao)
                    if nome is None or evento_atual == ultimo_evento_por_rosto.get(indice):
                        continue

                    json.dump({
                        "nome": nome,
                        "emocao": emocao,
                        "confianca": float(confianca_emocao),
                        "segundo": frame_count / fps,
                    }, arquivo_eventos, ensure_ascii=False)
                    arquivo_eventos.write("\n")
                    arquivo_eventos.flush()
                    ultimo_evento_por_rosto[indice] = evento_atual

                gravador.write(frame)
                cv2.imshow(nome_janela, frame)
                tecla = cv2.waitKey(1) & 0xFF

                if tecla == ord('f'):
                    cv2.setWindowProperty(
                        nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                elif tecla == 27:
                    cv2.setWindowProperty(
                        nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(nome_janela, 960, 720)

                try:
                    janela_visivel = cv2.getWindowProperty(
                        nome_janela, cv2.WND_PROP_VISIBLE
                    ) >= 1
                except cv2.error:
                    janela_visivel = False

                if tecla == ord('q') or not janela_visivel:
                    break
    finally:
        captura.release()
        gravador.release()
        cv2.destroyAllWindows()

    return nome_arquivo


if __name__ == "__main__":
    main()