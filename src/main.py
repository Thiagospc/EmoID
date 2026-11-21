import os
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

def limpar_tela():
    subprocess.run(
        ['clear' if os.name == 'posix' else 'cls'],
        check=False,
    )

def menu():
    print("""
============================================================
   🎭 EmoID - Sistema de Reconhecimento Facial de Emoções
============================================================
1 - 🎥 Capturar vídeo (detecção + reconhecimento + emoções)
2 - 👤 Cadastrar referências (fotos em figures/)
3 - 🚪 Sair
------------------------------------------------------------
""")

def main():
    while True:
        menu()
        opcao = input("Escolha uma opção: ").strip()
        
        if opcao == '1':
            limpar_tela()
            print("🎥 Iniciando captura de vídeo...\n")
            from captura_de_video import main as captura_main
            caminho_video = captura_main()

            if caminho_video is not None:
                print("\n🎬 Processando o vídeo mais recente...\n")
                caminho_processador = os.path.join(
                    os.path.dirname(__file__), "processar_video.py"
                )
                resultado = subprocess.run(
                    [sys.executable, caminho_processador, caminho_video],
                    check=False,
                )
                if resultado.returncode != 0:
                    print(
                        f"❌ O processamento foi encerrado com código "
                        f"{resultado.returncode}."
                    )

        elif opcao == '2':
            limpar_tela()
            print("👤 Gerando referências...\n")
            from cadastro_referencias import processar_pasta_figures
            processar_pasta_figures()

        elif opcao == '3':
            print("\n👋 Saindo...")
            break

        else:
            print("\n❌ Opção inválida! Tente novamente.")
            input("Pressione ENTER para continuar...")


if __name__ == "__main__":
    main()