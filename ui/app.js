// ui/app.js

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("review-form");
  const textarea = document.getElementById("texto");
  const resultadoDiv = document.getElementById("resultado");
  const historyList = document.getElementById("history-list");
  const btnAnalizar = document.getElementById("btn-analizar");

  // 🔹 Al cargar la página, traemos reseñas anteriores
  cargarHistorialInicial();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const texto = textarea.value.trim();
    if (!texto) {
      mostrarResultado("Por favor escribe una reseña antes de analizar.", "error");
      return;
    }

    btnAnalizar.disabled = true;
    btnAnalizar.textContent = "Analizando...";

    try {
      const response = await fetch("/analizar", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ texto }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const msg =
          errorData?.detail ??
          "Ocurrió un error al comunicarse con el servidor.";
        mostrarResultado(msg, "error");
        return;
      }

      const data = await response.json();
      // data = { sentimiento, probabilidad }

      mostrarResultado(
        `Sentimiento: ${data.sentimiento} (confianza: ${(data.probabilidad * 100).toFixed(1)}%)`,
        data.sentimiento
      );

      // Agregar reseña recién analizada al historial (con timestamp actual)
      agregarAlHistorial(texto, data.sentimiento, data.probabilidad, null);
      textarea.value = "";
    } catch (err) {
      console.error(err);
      mostrarResultado(
        "Ocurrió un error inesperado al analizar el texto.",
        "error"
      );
    } finally {
      btnAnalizar.disabled = false;
      btnAnalizar.textContent = "Analizar sentimiento";
    }
  });

  // --- Helpers UI ---------------------------------------------------------

  function mostrarResultado(mensaje, tipo) {
    resultadoDiv.textContent = mensaje;
    resultadoDiv.className = "resultado";

    if (tipo === "positivo") {
      resultadoDiv.classList.add("resultado-positivo");
    } else if (tipo === "negativo") {
      resultadoDiv.classList.add("resultado-negativo");
    } else if (tipo === "neutral") {
      resultadoDiv.classList.add("resultado-neutral");
    } else if (tipo === "error") {
      resultadoDiv.classList.add("resultado-error");
    }
  }

  function emojiPorSentimiento(sent) {
    if (sent === "positivo") return "😊";
    if (sent === "negativo") return "😞";
    return "😐"; // neutral
  }

  function agregarAlHistorial(texto, sentimiento, probabilidad, timestampIso = null) {
    const item = document.createElement("article");
    item.classList.add("history-item");

    if (sentimiento === "positivo") {
      item.classList.add("hist-positivo");
    } else if (sentimiento === "negativo") {
      item.classList.add("hist-negativo");
    } else if (sentimiento === "neutral") {
      item.classList.add("hist-neutral");
    }

    const fechaLocal = timestampIso
      ? new Date(timestampIso).toLocaleString("es-MX", {
          dateStyle: "short",
          timeStyle: "short",
        })
      : new Date().toLocaleString("es-MX", {
          dateStyle: "short",
          timeStyle: "short",
        });

    item.innerHTML = `
      <header class="history-header">
        <span class="history-sentiment">
          ${emojiPorSentimiento(sentimiento)} ${sentimiento.toUpperCase()}
        </span>
        <span class="history-prob">
          ${(probabilidad * 100).toFixed(1)}%
        </span>
        <span class="history-time">${fechaLocal}</span>
      </header>
      <p class="history-text">
        ${escapeHtml(texto)}
      </p>
    `;

    historyList.prepend(item);
  }

  async function cargarHistorialInicial() {
    try {
      const resp = await fetch("/resenas?limit=50");
      if (!resp.ok) {
        console.warn("No se pudo cargar historial de reseñas");
        return;
      }

      const data = await resp.json(); // [{timestamp, texto, sentimiento, probabilidad}, ...]

      historyList.innerHTML = "";

      data.forEach((r) => {
        agregarAlHistorial(r.texto, r.sentimiento, r.probabilidad, r.timestamp);
      });
    } catch (err) {
      console.error("Error al cargar historial de reseñas:", err);
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});
