import os
import sys
import shutil
import subprocess
import base64
import numpy as np
import cv2
from threading import Lock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from database import conectar, fechar, listar_registros, inserir_aula, inserir_aluno, buscar_aluno_por_nome, inserir_emocao
from reconhecimento_facial import carregar_detector, detectar_rostos, reconhecer_pessoa, carregar_referencias_salvas, recortar_rosto
from analise_emocoes import analisar_emocao
from flask import Flask, render_template, request, redirect, url_for, send_file, abort
from cadastro_referencias import processar_pasta_figures

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    static_url_path='/static'
)

_lock_recursos_reconhecimento = Lock()
_detector_reconhecimento = None
_detector_carregado = False
_referencias_reconhecimento = {}
_referencias_carregadas = False


def obter_recursos_reconhecimento():
    global _detector_reconhecimento, _detector_carregado
    global _referencias_reconhecimento, _referencias_carregadas

    with _lock_recursos_reconhecimento:
        if not _detector_carregado:
            _detector_reconhecimento = carregar_detector()
            _detector_carregado = True
        if not _referencias_carregadas:
            _referencias_reconhecimento = carregar_referencias_salvas("data/referencias/")
            _referencias_carregadas = True

        return _detector_reconhecimento, _referencias_reconhecimento


def invalidar_referencias_reconhecimento():
    global _referencias_reconhecimento, _referencias_carregadas

    with _lock_recursos_reconhecimento:
        _referencias_reconhecimento = {}
        _referencias_carregadas = False


@app.route("/")
def dashboard():
    conexao = conectar()
    
    registros = []
    stats = {
        "total_alunos": 0,
        "total_aulas": 0,
        "total_registros": 0,
        "emocao_predominante": "N/A"
    }
    emocoes_json = {"labels": [], "valores": []}
    
    if conexao:
        registros = listar_registros(conexao)
        
        cursor = conexao.cursor()
        cursor.execute("SELECT COUNT(*) FROM alunos")
        stats["total_alunos"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM aulas")
        stats["total_aulas"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM registros_emocoes")
        stats["total_registros"] = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT emocao, COUNT(*) as total
            FROM registros_emocoes
            GROUP BY emocao
            ORDER BY total DESC
        """)
        emocoes = cursor.fetchall()
        
        if emocoes:
            stats["emocao_predominante"] = emocoes[0][0]
            emocoes_json["labels"] = [e[0] for e in emocoes]
            emocoes_json["valores"] = [e[1] for e in emocoes]
        
        cursor.close()
        fechar(conexao)
    
    return render_template(
        "dashboard.html",
        pagina="dashboard",
        registros=registros[:20],
        stats=stats,
        emocoes_json=emocoes_json
    )


@app.route("/gravar")
def gravar():
    return render_template("gravar.html", pagina="gravar")

@app.route("/gravar/gravando", methods=["POST"])
def gravando_video():
    dados = request.get_json()
    frame_base64 = dados.get("frame", "")

    if not frame_base64:
        return {"rostos": []}

    frame_base64 = frame_base64.split(",")[1]
    frame_bytes = base64.b64decode(frame_base64)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"rostos": []}

    detector, referencias = obter_recursos_reconhecimento()
    rostos = detectar_rostos(frame, detector)

    resultados = []

    for (x, y, w, h) in rostos:
        rosto = recortar_rosto(frame, (x, y, w, h))

        nome, confianca_nome = None, 0.0
        if referencias:
            nome, confianca_nome = reconhecer_pessoa(rosto, referencias)

        emocao, confianca_emocao = analisar_emocao(rosto)

        resultados.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "nome": nome,
            "emocao": emocao,
            "confianca": float(confianca_emocao),
        })

    return {"rostos": resultados}

@app.route("/gravar/salvar", methods=["POST"])
def salvar_video():
    dados = request.get_json()
    eventos = dados.get("eventos", [])

    conexao = conectar()
    if not conexao:
        return {"erro": "Sem conexão"}, 500

    nome_arquivo = f"gravacao_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
    aula_id = inserir_aula(conexao, nome_arquivo)

    salvos = 0
    for evento in eventos:
        nome = evento["nome"]
        aluno_id = buscar_aluno_por_nome(conexao, nome)
        if aluno_id is None:
            aluno_id = inserir_aluno(conexao, nome)

        sucesso = inserir_emocao(
            conexao,
            aluno_id,
            aula_id,
            evento["emocao"],
            evento["confianca"] * 100,
            evento["segundo"],
        )
        if sucesso:
            salvos += 1

    fechar(conexao)
    return {"salvos": salvos, "aula_id": aula_id, "nome_arquivo": nome_arquivo}

@app.route("/gravar/salvar-video", methods=["POST"])
def salvar_video_arquivo():
    aula_id = request.form.get("aula_id")
    video = request.files.get("video")

    if not video or not aula_id:
        return {"erro": "Dados incompletos"}, 400

    conexao = conectar()
    if not conexao:
        return {"erro": "Sem conexão"}, 500

    cursor = conexao.cursor()
    cursor.execute("SELECT nome_arquivo FROM aulas WHERE id = %s", (aula_id,))
    resultado = cursor.fetchone()
    cursor.close()
    fechar(conexao)

    if not resultado:
        return {"erro": "Aula não encontrada"}, 404

    nome_arquivo = resultado[0]
    nome_base = os.path.splitext(nome_arquivo)[0]

    caminho_webm = os.path.join("data", f"{nome_base}.webm")
    caminho_mp4 = os.path.join("data", f"{nome_base}.mp4")

    os.makedirs("data", exist_ok=True)
    video.save(caminho_webm)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", caminho_webm, "-c:v", "libx264", "-preset", "fast", "-crf", "23", caminho_mp4],
            check=True,
            capture_output=True,
        )
        os.remove(caminho_webm)
        return {"ok": True, "arquivo": caminho_mp4}
    except Exception as e:
        print(f"❌ Erro ao converter: {e}")
        return {"erro": str(e)}, 500


@app.route("/videos")
def videos():
    conexao = conectar()
    lista_videos = []

    if conexao:
        cursor = conexao.cursor()
        cursor.execute("""
            SELECT a.id, a.nome_arquivo, a.duracao_segundos, a.data_hora,
                   COUNT(r.id) as total_registros
            FROM aulas a
            LEFT JOIN registros_emocoes r ON r.aula_id = a.id
            GROUP BY a.id
            ORDER BY a.data_hora DESC
        """)

        for row in cursor.fetchall():
            lista_videos.append({
                "id": row[0],
                "nome": row[1],
                "duracao": row[2] or 0,
                "data_dt": row[3],
                "data": row[3].strftime("%d/%m/%Y %H:%M"),
                "registros": row[4]
            })

        cursor.close()
        fechar(conexao)

    if not lista_videos and os.path.exists("data"):
        arquivos = []
        for nome_arquivo in os.listdir("data"):
            caminho = os.path.join("data", nome_arquivo)
            if os.path.isfile(caminho) and nome_arquivo.lower().endswith(".avi"):
                data_modificacao = datetime.fromtimestamp(os.path.getmtime(caminho))
                arquivos.append({
                    "id": nome_arquivo,
                    "nome": nome_arquivo,
                    "duracao": 0,
                    "data_dt": data_modificacao,
                    "data": data_modificacao.strftime("%d/%m/%Y %H:%M"),
                    "registros": 0,
                })

        lista_videos = sorted(arquivos, key=lambda video: video["data_dt"], reverse=True)

    lista_videos = sorted(lista_videos, key=lambda video: video["data_dt"], reverse=True)

    return render_template("videos.html", pagina="videos", videos=lista_videos)


@app.route("/aula/<path:nome_arquivo>")
def aula(nome_arquivo):
    caminho_arquivo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', nome_arquivo))

    if not os.path.isfile(caminho_arquivo):
        abort(404)

    return render_template(
        "video_detalhes.html",
        pagina="videos",
        nome_arquivo=nome_arquivo,
        video_url=url_for('servir_video', nome_arquivo=nome_arquivo)
    )


def converter_avi_para_mp4(caminho_arquivo):
    if not caminho_arquivo.lower().endswith('.avi'):
        return caminho_arquivo

    pasta = os.path.dirname(caminho_arquivo)
    nome_base = os.path.splitext(os.path.basename(caminho_arquivo))[0]
    caminho_mp4 = os.path.join(pasta, f"{nome_base}.mp4")

    if os.path.isfile(caminho_mp4):
        return caminho_mp4

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", caminho_arquivo,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                caminho_mp4,
            ],
            check=True,
            capture_output=True,
        )
        return caminho_mp4
    except Exception as e:
        print(f"❌ Erro ao converter: {e}")
        return caminho_arquivo


@app.route("/video/<path:nome_arquivo>")
def servir_video(nome_arquivo):
    caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    caminho_arquivo = os.path.abspath(os.path.join(caminho_base, nome_arquivo))

    if not os.path.isfile(caminho_arquivo):
        abort(404)

    caminho_final = converter_avi_para_mp4(caminho_arquivo)
    mimetype = 'video/mp4' if caminho_final.lower().endswith('.mp4') else 'video/x-msvideo'
    return send_file(caminho_final, mimetype=mimetype, as_attachment=False)


@app.route("/alunos")
def alunos():
    conexao = conectar()
    lista_alunos = []
    
    if conexao:
        cursor = conexao.cursor()
        cursor.execute("SELECT id, nome FROM alunos ORDER BY nome")
        lista_alunos = cursor.fetchall()
        cursor.close()
        fechar(conexao)
    
    return render_template("alunos.html", pagina="alunos", alunos=lista_alunos)


@app.route("/alunos/adicionar", methods=["POST"])
def adicionar_aluno():
    nome = request.form.get("nome", "").strip()
    
    if not nome:
        return redirect(url_for("alunos"))
    
    fotos = {
        "frontal": request.files.get("foto_frente"),
        "cima": request.files.get("foto_cima"),
        "baixo": request.files.get("foto_baixo"),
        "direita": request.files.get("foto_direita"),
        "esquerda": request.files.get("foto_esquerda")
    }
    
    for rotulo, arquivo in fotos.items():
        if arquivo is None:
            return redirect(url_for("alunos"))
        if not arquivo.filename:
            return redirect(url_for("alunos"))
        
    nome_pasta = nome.lower().replace(' ', '_')
    pasta_aluno = os.path.join('figures', nome_pasta)
    os.makedirs(pasta_aluno, exist_ok=True)
    
    for rotulo, arquivo in fotos.items():
        extensao = os.path.splitext(arquivo.filename)[1]
        caminho = os.path.join(pasta_aluno, f"{rotulo}{extensao}")
        arquivo.save(caminho)
        
    conexao = conectar()
    if conexao:
        cursor = conexao.cursor()
        cursor.execute(
            "INSERT INTO alunos (nome) VALUES (%s)",
            (nome.upper(),)
        )
        conexao.commit()
        cursor.close()
        fechar(conexao)
        
    processar_pasta_figures()
    invalidar_referencias_reconhecimento()
    
    return redirect(url_for("alunos"))

@app.route("/alunos/remover/<int:aluno_id>")
def remover_aluno(aluno_id):
    conexao = conectar()
    if conexao:
        cursor = conexao.cursor()
        cursor.execute("SELECT nome FROM alunos WHERE id = %s", (aluno_id,))
        resultado = cursor.fetchone()
        
        if resultado:
            nome_pasta = resultado[0].lower().replace(' ', '_')
            shutil.rmtree(f"figures/{nome_pasta}", ignore_errors=True)
            caminho_pkl = f"data/referencias/{nome_pasta}.pkl"
            if os.path.exists(caminho_pkl):
                os.remove(caminho_pkl)

            invalidar_referencias_reconhecimento()
                
            
            cursor.execute("DELETE FROM alunos WHERE id = %s", (aluno_id,))
            conexao.commit()
        
        cursor.close()
        fechar(conexao)
    
    return redirect(url_for("alunos"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)