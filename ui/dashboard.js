let sentimentChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  cargarEstadisticas();
});

/**
 * Cargar /stats y actualizar la vista + gráfica
 */
async function cargarEstadisticas() {
  const elTotal = document.getElementById("stat-total");
  const elPos = document.getElementById("stat-pos");
  const elNeu = document.getElementById("stat-neu");
  const elNeg = document.getElementById("stat-neg");

  const elPosPct = document.getElementById("stat-pos-pct");
  const elNeuPct = document.getElementById("stat-neu-pct");
  const elNegPct = document.getElementById("stat-neg-pct");

  const elUpdated = document.getElementById("stats-updated");

  try {
    const resp = await fetch("/stats");
    if (!resp.ok) {
      console.warn("No se pudieron cargar estadísticas");
      return;
    }

    const stats = await resp.json();

    // Actualizar tarjetas numéricas
    elTotal.textContent = stats.total ?? 0;
    elPos.textContent = stats.positivos ?? 0;
    elNeu.textContent = stats.neutrales ?? 0;
    elNeg.textContent = stats.negativos ?? 0;

    elPosPct.textContent = ((stats.porc_positivos ?? 0)).toFixed(2) + "%";
    elNeuPct.textContent = ((stats.porc_neutrales ?? 0)).toFixed(2) + "%";
    elNegPct.textContent = ((stats.porc_negativos ?? 0)).toFixed(2) + "%";

    if (stats.ultima_actualizacion) {
      const fecha = new Date(stats.ultima_actualizacion).toLocaleString("es-MX", {
        dateStyle: "short",
        timeStyle: "short",
      });
      elUpdated.textContent = fecha;
    } else {
      elUpdated.textContent = "—";
    }

    // Actualizar / crear gráfica
    renderSentimentChart(stats);
  } catch (err) {
    console.error("Error al cargar estadísticas:", err);
  }
}

/**
 * Crea o actualiza la gráfica de dona con Chart.js
 */
function renderSentimentChart(stats) {
  const ctx = document.getElementById("sentimentChart");
  if (!ctx) return;

  const dataValues = [
    stats.positivos ?? 0,
    stats.neutrales ?? 0,
    stats.negativos ?? 0,
  ];

  const total = dataValues.reduce((a, b) => a + b, 0);
  if (total === 0) {
    // Si no hay datos, mejor no mostramos nada raro
    if (sentimentChartInstance) {
      sentimentChartInstance.destroy();
      sentimentChartInstance = null;
    }
    return;
  }

  const labels = ["Positivas", "Neutrales", "Negativas"];

  const backgroundColors = [
    "rgba(74, 222, 128, 0.7)",  // verde pastel
    "rgba(129, 140, 248, 0.7)", // lila pastel
    "rgba(248, 113, 113, 0.7)", // rojo pastel
  ];

  const borderColors = [
    "rgba(34, 197, 94, 1)",
    "rgba(79, 70, 229, 1)",
    "rgba(239, 68, 68, 1)",
  ];

  // Si ya existe, solo actualizamos datos
  if (sentimentChartInstance) {
    sentimentChartInstance.data.labels = labels;
    sentimentChartInstance.data.datasets[0].data = dataValues;
    sentimentChartInstance.update();
    return;
  }

  sentimentChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: dataValues,
          backgroundColor,
          borderColor: borderColors,
          borderWidth: 1,
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "60%", // dona
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            padding: 16,
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.raw;
              const pct = ((value / total) * 100).toFixed(1);
              return `${context.label}: ${value} reseñas (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

