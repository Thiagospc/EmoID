import cv2
import json
import os
import sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from database import conectar, fechar, inserir_aluno, buscar_aluno_por_nome, inserir_aula, inserir_emocao
from reconhecimento_facial import (
    carregar_detector,
    detectar_rostos,
    reconhecer_pessoa,
    carregar_referencias_salvas,
    recortar_rosto,
)
from analise_emocoes import analisar_emocao

limite_confianca = 0.10

def processar_eventos(caminho_eventos, aula_id, conexao):
    registros_salvos = 0
    with open(caminho_eventos, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            if not linha.strip():
                continue

            evento = json.loads(linha)
            nome = evento["nome"]
            aluno_id = buscar_aluno_por_nome(conexao, nome)
            if aluno_id is None:
                aluno_id = inserir_aluno(conexao, nome, None)

            segundo = float(evento["segundo"])
            sucesso = inserir_emocao(
                conexao,
                aluno_id,
                aula_id,
                evento["emocao"],
                float(evento["confianca"]) * 100,
                segundo,
            )
            if sucesso:
                registros_salvos += 1

    return registros_salvos

def processar_video(caminho_video, aula_id, referencias, detector, conexao):
    captura = cv2.VideoCapture(caminho_video)
    
    if not captura.isOpened():
        print(f"❌ Erro ao abrir o vídeo!")
        return 0
    
    fps = captura.get(cv2.CAP_PROP_FPS)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps if fps > 0 else 0
    
    frame_count = 0
    registros_salvos = 0
    ultima_emocao_por_aluno = {}
    while True:
        ret, frame = captura.read()
        
        if not ret:
            break
        
        frame_count += 1
        print(
            f"\rProcessando emoções: {frame_count}/{total_frames}",
            end="",
            flush=True,
        )
        
        tempo_atual = frame_count / fps
        
        rostos = detectar_rostos(frame, detector)   
        
        if len(rostos) == 0:
            continue
        
        for i, (x, y, w, h) in enumerate(rostos):
            rosto_recortado = recortar_rosto(frame, (x, y, w, h))
            
            nome = None
            confianca_nome = 0.0
            if referencias:
                nome, confianca_nome = reconhecer_pessoa(rosto_recortado, referencias)
            
            emocao, confianca_emocao = analisar_emocao(rosto_recortado)
            
            if nome is None or confianca_nome < limite_confianca:
                continue

            if ultima_emocao_por_aluno.get(nome) == emocao:
                continue
            
            aluno_id = buscar_aluno_por_nome(conexao, nome)
            if aluno_id is None:
                aluno_id = inserir_aluno(conexao, nome, None)
            
            if aluno_id is None:
                print(f"   ⚠️ Não foi possível obter o ID do aluno {nome}")
                continue
            
            sucesso = inserir_emocao(
                conexao, aluno_id, aula_id,
                emocao, confianca_emocao * 100,
                tempo_atual
            )
            
            if sucesso:
                registros_salvos += 1
                ultima_emocao_por_aluno[nome] = emocao
    
    captura.release()
    print()
    return registros_salvos 
    
def main(caminho_video=None, usar_argumento=True):
    if caminho_video is None and usar_argumento and len(sys.argv) >= 2:
        caminho_video = sys.argv[1]

    if caminho_video is None:
        if os.path.exists("data/"):
            videos = [f for f in os.listdir("data/") if f.endswith(".avi")]
            if videos:
                print("\n📹 Vídeos disponíveis em data/:")
                for i, v in enumerate(videos, 1):
                    print(f"   {i} - {v}")
                
                escolha = input("\nEscolha o número do vídeo: ").strip()
                try:
                    idx = int(escolha) - 1
                    caminho_video = os.path.join("data/", videos[idx])
                except (ValueError, IndexError):
                    print("❌ Escolha inválida!")
                    return
            else:
                print("❌ Nenhum vídeo encontrado em data/")
                return
        else:
            print("❌ Pasta data/ não encontrada")
            return
    
    if not os.path.exists(caminho_video):
        print(f"❌ Vídeo não encontrado: {caminho_video}")
        return
    
    conexao = conectar()
    if conexao is None:
        print("❌ Não foi possível conectar ao banco")
        return
    
    detector = carregar_detector()
    if detector is None:
        print("❌ Detector não carregado")
        fechar(conexao)
        return
    
    referencias = carregar_referencias_salvas("data/referencias/")
    if not referencias:
        print("⚠️ Nenhuma referência encontrada!")
        fechar(conexao)
        return
    
    print(f"\n👤 Pessoas registradas: {list(referencias.keys())}")
    
    # criação de aula
    nome_arquivo = os.path.basename(caminho_video)
    captura = cv2.VideoCapture(caminho_video)
    fps = captura.get(cv2.CAP_PROP_FPS)
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao = total_frames / fps if fps > 0 else 0
    captura.release()
    
    aula_id = inserir_aula(conexao, nome_arquivo, duracao)
    if aula_id is None:
        print("❌ Erro ao criar aula no banco")
        fechar(conexao)
        return
    
    try:
        caminho_eventos = os.path.splitext(caminho_video)[0] + "_eventos.jsonl"
        if os.path.exists(caminho_eventos):
            registros = processar_eventos(caminho_eventos, aula_id, conexao)
        else:
            registros = processar_video(caminho_video, aula_id, referencias, detector, conexao)
    except Exception as e:
        conexao.rollback()
        print(f"❌ Erro durante o processamento do vídeo: {e}")
        fechar(conexao)
        return

    fechar(conexao)
    
    print("\n" + "=" * 60)
    print(f"{registros} registros salvos no banco")
    print("=" * 60)

if __name__=="__main__":
    main()