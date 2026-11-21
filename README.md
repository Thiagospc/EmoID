# 🎭 EmoID: Protótipo de Reconhecimento Facial de Emoções no Meio Acadêmico

![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)
![OpenCV](https://img.shields.io/badge/Vis%C3%A3o%20Computacional-OpenCV-green)
![PostgreSQL](https://img.shields.io/badge/Banco%20de%20Dados-PostgreSQL-blue)

## 📖 Sobre o Projeto

O **EmoID** é um protótipo de software desenvolvido como Trabalho de Conclusão de Curso (TCC) para o Bacharelado em Sistemas de Informação da Universidade Federal Rural da Amazônia (UFRA).

O sistema tem como objetivo auxiliar docentes na identificação do engajamento dos alunos durante as aulas, por meio da análise de expressões faciais em tempo real. A ferramenta também visa contribuir para a inclusão de alunos com deficiência (oferecendo um canal não verbal de manifestação emocional) e verificar a qualidade das metodologias pedagógicas adotadas.

O desenvolvimento baseou-se na metodologia **Design Science Research (DSR)**, na teoria das emoções básicas de **Ekman** e na Teoria do Controle-Valor de **Pekrun**.

## ✨ Funcionalidades Principais

*   **Captura e Reconhecimento Facial:** Identifica o rosto do aluno cadastrado em tempo real utilizando a biblioteca DeepFace (modelo Facenet).
*   **Análise de Emoções:** Classifica as expressões faciais em 7 emoções básicas (felicidade, tristeza, raiva, medo, nojo, surpresa e neutro).
*   **Cadastro de Alunos:** Interface web para cadastrar alunos e extrair seus *embeddings* faciais a partir de 5 fotos de referência.
*   **Gravação Web:** Permite a captura de vídeo diretamente pelo navegador e processamento em tempo real.
*   **Dashboard Analítico:** Painéis com estatísticas, distribuição de emoções e histórico das aulas processadas.
*   **Armazenamento Seguro:** Todos os registros são salvos em um banco de dados relacional PostgreSQL.

## 🛠️ Tecnologias Utilizadas

O projeto foi construído utilizando as seguintes tecnologias:

*   **Linguagem:** Python 3.10+
*   **Backend Web:** Flask
*   **Visão Computacional e Deep Learning:** OpenCV, DeepFace, TensorFlow
*   **Banco de Dados:** PostgreSQL
*   **Frontend:** HTML, Tailwind CSS (via CDN), JavaScript
*   **Gráficos:** Chart.js (frontend) e Matplotlib/Plotly (processamento backend)
*   **Conversão de Vídeo:** FFmpeg (para converter .avi em .mp4 para web)

## 📁 Estrutura do Projeto

```text
EmoID/
├── data/                      # Vídeos gravados (.avi, .mp4) e arquivos .pkl (referências faciais)
├── figures/                   # Fotos de referência dos alunos (organizadas por pastas)
├── haarcascades/              # Arquivo XML do classificador Haar Cascade
├── src/
│   ├── analise_emocoes.py     # Lógica do DeepFace para emoções
│   ├── cadastro_referencias.py# Geração dos embeddings (.pkl) a partir das fotos
│   ├── captura_de_video.py    # Script de captura local via terminal
│   ├── database.py            # Conexão e queries do PostgreSQL
│   ├── main.py                # Menu principal via terminal
│   ├── processar_video.py     # Processamento de vídeos salvos e envio ao banco
│   ├── reconhecimento_facial.py # Lógica de detecção e comparação de rostos
│   └── web/                   # Aplicação Web (Flask)
│       ├── app.py             # Servidor Flask e rotas
│       ├── static/            # Arquivos JS
│       └── templates/         # Arquivos HTML (Dashboard, Gravar, Vídeos, Alunos)
├── .env                       # Variáveis de ambiente (credenciais do banco)
├── requirements.txt           # Dependências do projeto
└── README.md                  # Este arquivo