// app.js

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("review-form");
  const textarea = document.getElementById("texto");
  const resultadoDiv = document.getElementById("resultado");
  const historyList = document.getElementById("history-list");
  const btnAnalizar = document.getElementById("btn-analizar");

  // 🔹 Al cargar la página, traemos las reseñas anteriores del backend
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
      // data = { sentimiento: "...", probabilidad: 0.87 }

      mostrarResultado(
        `Sentimiento: ${data.sentimiento} (confianza: ${(data.probabilidad * 100).toFixed(1)}%)`,
        data.sentimiento
      );

      // Agregar esta reseña al historial visual (sin timestamp, usamos la hora actual)
      agregarAlHistorial(texto, data.sentimiento, data.probabilidad);

      // Si quieres limpiar el textarea:
      // textarea.value = "";
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

  /**
   * Retorna un emoji según el sentimiento
   */
  function emojiPorSentimiento(sent) {
    if (sent === "positivo") return "😍";
    if (sent === "negativo") return "😠";
    return "😴"; // neutral
  }

  /**
   * Muestra el resultado actual del análisis
   * tipo puede ser: "positivo", "negativo", "neutral", "error"
   */
  function mostrarResultado(mensaje, tipo) {
    resultadoDiv.className = "resultado"; // resetea clases

    let prefix = "";
    if (tipo === "positivo") {
      resultadoDiv.classList.add("resultado-positivo");
      prefix = "😊 ";
    } else if (tipo === "negativo") {
      resultadoDiv.classList.add("resultado-negativo");
      prefix = "😞 ";
    } else if (tipo === "neutral") {
      resultadoDiv.classList.add("resultado-neutral");
      prefix = "😐 ";
    } else if (tipo === "error") {
      resultadoDiv.classList.add("resultado-error");
      prefix = "⚠️ ";
    }

    resultadoDiv.textContent = prefix + mensaje;
  }

  /**
   * Agrega una tarjeta al historial de reseñas
   * timestampIso es opcional:
   *  - si viene del backend, lo usamos
   *  - si es null, usamos la hora actual
   */
  function agregarAlHistorial(texto, sentimiento, probabilidad, timestampIso = null) {
    const item = document.createElement("article");
    item.classList.add("history-item");

    // Clase según sentimiento para colorear borde/fondo
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

    // Insertar arriba (última reseña primero)
    historyList.prepend(item);
  }

  /**
   * Cargar reseñas anteriores desde /resenas
   */
  async function cargarHistorialInicial() {
    try {
      const resp = await fetch("/resenas?limit=50");
      if (!resp.ok) {
        console.warn("No se pudo cargar historial de reseñas");
        return;
      }

      const data = await resp.json(); // [{timestamp, texto, sentimiento, probabilidad}, ...]

      // Limpiamos por si acaso
      historyList.innerHTML = "";

      data.forEach((r) => {
        agregarAlHistorial(r.texto, r.sentimiento, r.probabilidad, r.timestamp);
      });
    } catch (err) {
      console.error("Error al cargar historial de reseñas:", err);
    }
  }

  /**
   * Evitar que HTML del usuario se interprete como tags
   */
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});
