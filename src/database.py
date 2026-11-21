import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def conectar():
    try:
        conexao = psycopg2.connect(
            host=os.getenv("HOST"),
            database=os.getenv("DB"),
            user=os.getenv("USUARIO"),
            password=os.getenv("SENHA")
        )
        print(f"✅ Conectado ao banco '{os.getenv('DB')}'")
        return conexao
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return None


def fechar(conexao):
    if conexao:
        conexao.close()
        print("🔌 Conexão encerrada")


def inserir_aluno(conexao, nome):
    cursor = conexao.cursor()
    cursor.execute("INSERT INTO alunos (nome) VALUES (%s) RETURNING id", (nome,))
    aluno_id = cursor.fetchone()[0]
    conexao.commit()
    cursor.close()
    return aluno_id


def buscar_aluno_por_nome(conexao, nome):
    cursor = conexao.cursor()
    cursor.execute("SELECT id FROM alunos WHERE nome = %s", (nome,))
    resultado = cursor.fetchone()
    cursor.close()
    return resultado[0] if resultado else None


def inserir_aula(conexao, nome_arquivo, duracao_segundos=None):
    cursor = conexao.cursor()
    cursor.execute("INSERT INTO aulas (nome_arquivo, duracao_segundos) VALUES (%s, %s) RETURNING id", (nome_arquivo, duracao_segundos))
    aula_id = cursor.fetchone()[0]
    conexao.commit()
    cursor.close()
    return aula_id


def inserir_emocao(conexao, aluno_id, aula_id, emocao, confianca, segundo_no_video):
    try:
        cursor = conexao.cursor()
        cursor.execute("INSERT INTO registros_emocoes (aluno_id, aula_id, emocao, confianca, segundo_no_video) VALUES (%s, %s, %s, %s, %s)", (int(aluno_id), int(aula_id), emocao, float(confianca), float(segundo_no_video)))
        conexao.commit()
        cursor.close()
        return True
    except Exception as e:
        conexao.rollback()
        print(f"❌ Erro ao inserir emoção: {e}")
        return False


def listar_registros(conexao, aula_id=None):
    cursor = conexao.cursor()
    
    if aula_id:
        cursor.execute("SELECT a.nome, r.emocao, r.confianca, r.segundo_no_video, r.data_hora FROM registros_emocoes r JOIN alunos a ON r.aluno_id = a.id WHERE r.aula_id = %s ORDER BY r.segundo_no_video", (aula_id,))
    else:
        cursor.execute("SELECT a.nome, r.emocao, r.confianca, r.segundo_no_video, r.data_hora FROM registros_emocoes r JOIN alunos a ON r.aluno_id = a.id ORDER BY r.data_hora DESC")
    
    resultados = cursor.fetchall()
    cursor.close()
    return resultados


def listar_aulas(conexao):
    cursor = conexao.cursor()
    cursor.execute("SELECT a.id, a.nome_arquivo, a.duracao_segundos, a.data_hora, COUNT(r.id) as total_registros FROM aulas a LEFT JOIN registros_emocoes r ON r.aula_id = a.id GROUP BY a.id ORDER BY a.data_hora DESC")
    resultados = cursor.fetchall()
    cursor.close()
    return resultados


def buscar_aula_por_id(conexao, aula_id):
    cursor = conexao.cursor()
    cursor.execute("SELECT id, nome_arquivo, duracao_segundos, data_hora FROM aulas WHERE id = %s", (aula_id,))
    resultado = cursor.fetchone()
    cursor.close()
    return resultado


if __name__ == '__main__':
    conn = conectar()
    
    if conn:
        fechar(conn)