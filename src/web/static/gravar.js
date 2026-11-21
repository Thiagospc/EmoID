const video = document.getElementById("video");
const videoContainer = document.getElementById("video-container");
const canvasOverlay = document.getElementById("canvas-overlay");
const ctxOverlay = canvasOverlay.getContext("2d");
const btnIniciar = document.getElementById("btn-iniciar");
const btnParar = document.getElementById("btn-parar");
const status = document.getElementById("status");
const framesEl = document.getElementById("frames");

let stream = null;
let intervalId = null;
let frameCount = 0;
let eventos = [];
let ultimoEvento = {};
let dataInicio = null;
let mediaRecorder = null;
let chunks = [];
let frameEmProcessamento = false;

function sincronizarCanvas() {
    if (!video.videoWidth || !video.videoHeight) return false;

    videoContainer.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

    if (canvasOverlay.width !== video.videoWidth) {
        canvasOverlay.width = video.videoWidth;
    }
    if (canvasOverlay.height !== video.videoHeight) {
        canvasOverlay.height = video.videoHeight;
    }

    return true;
}

video.addEventListener("loadedmetadata", sincronizarCanvas);

btnIniciar.addEventListener("click", async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        await video.play();

        if (!sincronizarCanvas()) {
            throw new Error("Não foi possível obter as dimensões do vídeo.");
        }

        status.textContent = "Gravando...";
        btnIniciar.disabled = true;
        btnParar.disabled = false;

        frameCount = 0;
        eventos = [];
        ultimoEvento = {};
        chunks = [];
        dataInicio = new Date().toISOString();
        framesEl.textContent = "0";

        // Inicia gravação do vídeo
        mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };
        mediaRecorder.start();

        intervalId = setInterval(enviarFrame, 200);
    } catch (err) {
        status.textContent = "Erro: " + err.message;
    }
});

btnParar.addEventListener("click", async () => {
    btnParar.disabled = true;
    btnIniciar.disabled = true;

    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }

    ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    status.textContent = "Salvando...";

    // Aguarda o MediaRecorder finalizar
    const blobVideo = await new Promise((resolve) => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            resolve(null);
            return;
        }
        mediaRecorder.onstop = () => {
            resolve(new Blob(chunks, { type: "video/webm" }));
        };
        mediaRecorder.stop();
    });

    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }

    try {
        const respostaEventos = await fetch("/gravar/salvar", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ eventos: eventos, data_inicio: dataInicio })
        });

        const dados = await respostaEventos.json();
        if (!respostaEventos.ok) {
            throw new Error(dados.erro || "Não foi possível salvar os registros.");
        }

        if (blobVideo && dados.aula_id) {
            const formData = new FormData();
            formData.append("video", blobVideo, "gravacao.webm");
            formData.append("aula_id", dados.aula_id);

            const respostaVideo = await fetch("/gravar/salvar-video", {
                method: "POST",
                body: formData
            });
            const resultadoVideo = await respostaVideo.json();
            if (!respostaVideo.ok || !resultadoVideo.ok) {
                throw new Error(resultadoVideo.erro || "Não foi possível salvar o vídeo.");
            }

            status.textContent = `Vídeo salvo! ${dados.salvos || 0} registros.`;
        } else {
            status.textContent = `Registros salvos: ${dados.salvos || 0}. Vídeo indisponível.`;
        }
    } catch (err) {
        status.textContent = "Erro ao salvar: " + err.message;
    } finally {
        btnIniciar.disabled = false;
    }
});

async function enviarFrame() {
    if (!sincronizarCanvas() || frameEmProcessamento) return;
    frameEmProcessamento = true;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const frame = canvas.toDataURL("image/jpeg", 0.6);

    try {
        const resposta = await fetch("/gravar/gravando", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frame })
        });

        const dados = await resposta.json();
        frameCount++;
        framesEl.textContent = frameCount;

        desenharOverlay(dados.rostos || []);

        (dados.rostos || []).forEach((rosto) => {
            if (!rosto.nome) return;

            const chave = `${rosto.nome}-${rosto.emocao}`;
            if (ultimoEvento[chave]) return;

            eventos.push({
                nome: rosto.nome,
                emocao: rosto.emocao,
                confianca: rosto.confianca,
                segundo: frameCount * 0.2
            });

            ultimoEvento[chave] = true;

            setTimeout(() => {
                delete ultimoEvento[chave];
            }, 5000);
        });
    } catch (err) {
        console.error(err);
    } finally {
        frameEmProcessamento = false;
    }
}

function desenharOverlay(rostos) {
    ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

    if (!sincronizarCanvas()) return;

    const escalaX = canvasOverlay.width / video.videoWidth;
    const escalaY = canvasOverlay.height / video.videoHeight;

    const cores = {
        felicidade: "#2ecc71",
        tristeza: "#3498db",
        raiva: "#e74c3c",
        medo: "#9b59b6",
        nojo: "#e67e22",
        surpresa: "#f1c40f",
        neutro: "#95a5a6"
    };

    rostos.forEach((rosto) => {
        const x = rosto.x * escalaX;
        const y = rosto.y * escalaY;
        const w = rosto.w * escalaX;
        const h = rosto.h * escalaY;
        const xInvertido = canvasOverlay.width - x - w;

        const cor = cores[rosto.emocao] || "#ffffff";

        ctxOverlay.strokeStyle = cor;
        ctxOverlay.lineWidth = 3;
        ctxOverlay.strokeRect(xInvertido, y, w, h);

        const nome = rosto.nome ? rosto.nome.split(" ")[0].toLowerCase() : "desconhecido";
        const texto = `${nome} - ${rosto.emocao} (${Math.round(rosto.confianca * 100)}%)`;

        ctxOverlay.font = "bold 16px sans-serif";
        ctxOverlay.fillStyle = cor;
        ctxOverlay.fillText(texto, xInvertido, Math.max(18, y - 8));
    });
}