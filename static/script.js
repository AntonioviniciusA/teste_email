document.addEventListener("DOMContentLoaded", function () {
  // Elementos da interface
  const uploadForm = document.getElementById("upload-form");
  const fileOption = document.getElementById("file-option");
  const textOption = document.getElementById("text-option");
  const fileInput = document.getElementById("file-input");
  const selectFileBtn = document.getElementById("select-file-btn");
  const fileInfo = document.getElementById("file-info");
  const textInput = document.getElementById("text-input");
  const processBtn = document.getElementById("process-btn");
  const loading = document.getElementById("loading");
  const errorMessage = document.getElementById("error-message");
  const resultContainer = document.getElementById("result-container");
  const categoryValue = document.getElementById("category-value");
  const scoreValue = document.getElementById("score-value");
  const reasonsList = document.getElementById("reasons-list");
  const responseContent = document.getElementById("response-content");
  const processedText = document.getElementById("processed-text");

  // Elementos do gerenciador de critérios
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");
  const criteriaForm = document.getElementById("criteria-form");
  const productiveKeywords = document.getElementById("productive-keywords");
  const unproductiveKeywords = document.getElementById("unproductive-keywords");
  const requiredPatterns = document.getElementById("required-patterns");
  const minLength = document.getElementById("min-length");
  const maxLength = document.getElementById("max-length");
  const resetCriteriaBtn = document.getElementById("reset-criteria-btn");
  const criteriaSuccessMessage = document.getElementById(
    "criteria-success-message"
  );
  const criteriaErrorMessage = document.getElementById(
    "criteria-error-message"
  );

  // Estado da aplicação
  let currentOption = "text";
  let selectedFile = null;

  // Inicialização
  loadCriteria();
  setupEventListeners();

  function setupEventListeners() {
    // Abas
    tabButtons.forEach((button) => {
      button.addEventListener("click", function () {
        const tabId = this.getAttribute("data-tab");
        switchTab(tabId);
      });
    });

    // Upload de arquivo
    selectFileBtn.addEventListener("click", function () {
      fileInput.click();
    });

    fileInput.addEventListener("change", function (e) {
      if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        fileInfo.textContent = `Arquivo selecionado: ${selectedFile.name}`;
        setActiveOption("file");
        validateInput();
      }
    });

    textInput.addEventListener("input", function () {
      setActiveOption("text");
      validateInput();
    });

    fileOption.addEventListener("click", function () {
      fileInput.click();
    });

    textOption.addEventListener("click", function () {
      setActiveOption("text");
      textInput.focus();
    });

    uploadForm.addEventListener("submit", function (e) {
      e.preventDefault();
      processEmail();
    });

    // Gerenciador de critérios
    criteriaForm.addEventListener("submit", function (e) {
      e.preventDefault();
      saveCriteria();
    });

    resetCriteriaBtn.addEventListener("click", function () {
      if (
        confirm(
          "Restaurar critérios padrão? Isso substituirá suas configurações atuais."
        )
      ) {
        resetCriteria();
      }
    });
  }

  function switchTab(tabId) {
    // Atualizar botões da aba
    tabButtons.forEach((button) => {
      button.classList.toggle(
        "active",
        button.getAttribute("data-tab") === tabId
      );
    });

    // Atualizar conteúdo da aba
    tabContents.forEach((content) => {
      content.classList.toggle("active", content.id === `${tabId}-tab`);
    });
  }

  function setActiveOption(option) {
    currentOption = option;

    if (option === "file") {
      fileOption.classList.add("active");
      textOption.classList.remove("active");
    } else {
      textOption.classList.add("active");
      fileOption.classList.remove("active");
    }

    validateInput();
  }

  function validateInput() {
    if (currentOption === "file") {
      processBtn.disabled = !selectedFile;
    } else {
      processBtn.disabled = textInput.value.trim() === "";
    }
  }

  function processEmail() {
    // Mostrar loading
    loading.style.display = "block";
    errorMessage.style.display = "none";
    resultContainer.style.display = "none";

    // Criar FormData
    const formData = new FormData();

    if (currentOption === "file" && selectedFile) {
      formData.append("file", selectedFile);
    } else {
      formData.append("text", textInput.value);
    }

    // Enviar para o backend
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        loading.style.display = "none";

        if (data.error) {
          showError(data.error);
        } else {
          displayResults(data);
        }
      })
      .catch((error) => {
        loading.style.display = "none";
        showError("Erro de conexão com o servidor");
        console.error("Error:", error);
      });
  }

  function displayResults(data) {
    // Atualizar categoria
    categoryValue.textContent = data.classification;
    categoryValue.className =
      "classification-value " +
      (data.classification === "Produtivo"
        ? "productive"
        : data.classification === "Improdutivo"
        ? "unproductive"
        : "neutral");

    // Atualizar score
    scoreValue.textContent = data.score;

    // Atualizar razões
    reasonsList.innerHTML = "";
    if (data.reasons && data.reasons.length > 0) {
      data.reasons.forEach((reason) => {
        const li = document.createElement("li");
        li.textContent = reason;
        reasonsList.appendChild(li);
      });
    }

    // Atualizar resposta
    responseContent.textContent = data.response;

    // Atualizar texto processado
    processedText.textContent =
      data.processed_text || "Texto processado não disponível";

    // Mostrar resultados
    resultContainer.style.display = "block";

    // Rolar para os resultados
    resultContainer.scrollIntoView({ behavior: "smooth" });
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = "block";
  }

  function loadCriteria() {
    fetch("/criteria")
      .then((response) => response.json())
      .then((criteria) => {
        // Preencher formulário com critérios atuais
        productiveKeywords.value = criteria.productive_keywords.join("\n");
        unproductiveKeywords.value = criteria.unproductive_keywords.join("\n");
        requiredPatterns.value = criteria.required_patterns.join("\n");
        minLength.value = criteria.min_length;
        maxLength.value = criteria.max_length;
      })
      .catch((error) => {
        console.error("Erro ao carregar critérios:", error);
      });
  }

  function saveCriteria() {
    const criteria = {
      productive_keywords: productiveKeywords.value
        .split("\n")
        .filter((k) => k.trim()),
      unproductive_keywords: unproductiveKeywords.value
        .split("\n")
        .filter((k) => k.trim()),
      required_patterns: requiredPatterns.value
        .split("\n")
        .filter((p) => p.trim()),
      min_length: parseInt(minLength.value),
      max_length: parseInt(maxLength.value),
    };

    fetch("/criteria", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(criteria),
    })
      .then((response) => response.json())
      .then((result) => {
        if (result.success) {
          showCriteriaSuccess("Critérios salvos com sucesso!");
        } else {
          showCriteriaError(result.error || "Erro ao salvar critérios");
        }
      })
      .catch((error) => {
        console.error("Erro:", error);
        showCriteriaError("Erro de conexão ao salvar critérios");
      });
  }

  function resetCriteria() {
    // Restaurar valores padrão
    productiveKeywords.value = [
      "reunião",
      "projeto",
      "trabalho",
      "relatório",
      "prazo",
      "cliente",
      "entrega",
      "solicitação",
      "orçamento",
      "contrato",
      "desenvolvimento",
      "apresentação",
      "análise",
      "negócio",
      "empresa",
    ].join("\n");

    unproductiveKeywords.value = [
      "spam",
      "promoção",
      "oferta",
      "desconto",
      "newsletter",
      "corrente",
      "loteria",
      "prêmio",
      "ganhador",
      "grátis",
      "urgente",
      "importante",
    ].join("\n");

    requiredPatterns.value = [
      "\\b(projeto|tarefa|atividade)\\b.*\\b(prazo|data|entrega)\\b",
      "\\b(reunião|encontro)\\b.*\\b(agenda|agendar|marcar)\\b",
      "\\b(relatório|documento|análise)\\b.*\\b(entregar|enviar|preparar)\\b",
      "\\b(solicitação|pedido|requisição)\\b.*\\b(resposta|retorno|feedback)\\b",
      "\\b(desenvolvimento|implementação)\\b.*\\b(código|sistema|módulo)\\b",
    ].join("\n");
    minLength.value = 50;
    maxLength.value = 5000;

    showCriteriaSuccess(
      'Critérios restaurados para os valores padrão. Clique em "Salvar Critérios" para aplicar.'
    );
  }

  function showCriteriaSuccess(message) {
    criteriaSuccessMessage.textContent = message;
    criteriaSuccessMessage.style.display = "block";
    criteriaErrorMessage.style.display = "none";

    setTimeout(() => {
      criteriaSuccessMessage.style.display = "none";
    }, 5000);
  }

  function showCriteriaError(message) {
    criteriaErrorMessage.textContent = message;
    criteriaErrorMessage.style.display = "block";
    criteriaSuccessMessage.style.display = "none";

    setTimeout(() => {
      criteriaErrorMessage.style.display = "none";
    }, 5000);
  }

  // Inicialização
  setActiveOption("text");
});

// Função para gerar padrões automaticamente das combinações
function generatePatternsFromCombinations() {
  const patterns = [];

  // Combinação 1: Projeto/Tarefa + Prazo/Entrega
  const combo1Words = document
    .getElementById("combo-words-1")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);
  const combo1Actions = document
    .getElementById("combo-actions-1")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);

  if (combo1Words.length > 0 && combo1Actions.length > 0) {
    const pattern = `\\b(${combo1Words.join("|")})\\b.*\\b(${combo1Actions.join(
      "|"
    )})\\b`;
    patterns.push(pattern);
  }

  // Combinação 2: Reunião/Encontro + Agenda/Marcar
  const combo2Words = document
    .getElementById("combo-words-2")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);
  const combo2Actions = document
    .getElementById("combo-actions-2")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);

  if (combo2Words.length > 0 && combo2Actions.length > 0) {
    const pattern = `\\b(${combo2Words.join("|")})\\b.*\\b(${combo2Actions.join(
      "|"
    )})\\b`;
    patterns.push(pattern);
  }

  // Combinação 3: Relatório/Documento + Entregar/Enviar
  const combo3Words = document
    .getElementById("combo-words-3")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);
  const combo3Actions = document
    .getElementById("combo-actions-3")
    .value.split(",")
    .map((w) => w.trim())
    .filter((w) => w);

  if (combo3Words.length > 0 && combo3Actions.length > 0) {
    const pattern = `\\b(${combo3Words.join("|")})\\b.*\\b(${combo3Actions.join(
      "|"
    )})\\b`;
    patterns.push(pattern);
  }

  return patterns;
}

// Função para salvar critérios (atualizada)
function saveCriteria() {
  const productiveKeywords = document
    .getElementById("productive-keywords")
    .value.split("\n")
    .filter((k) => k.trim());
  const unproductiveKeywords = document
    .getElementById("unproductive-keywords")
    .value.split("\n")
    .filter((k) => k.trim());
  const requiredPatterns = generatePatternsFromCombinations();

  const criteria = {
    productive_keywords: productiveKeywords,
    unproductive_keywords: unproductiveKeywords,
    required_patterns: requiredPatterns,
    min_length: parseInt(document.getElementById("min-length").value),
    max_length: parseInt(document.getElementById("max-length").value),
    productive_weight: parseInt(
      document.getElementById("productive-weight").value
    ),
    unproductive_weight: parseInt(
      document.getElementById("unproductive-weight").value
    ),
  };

  fetch("/criteria", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(criteria),
  })
    .then((response) => response.json())
    .then((result) => {
      if (result.success) {
        showCriteriaSuccess("Critérios salvos com sucesso!");
      } else {
        showCriteriaError(result.error || "Erro ao salvar critérios");
      }
    })
    .catch((error) => {
      console.error("Erro:", error);
      showCriteriaError("Erro de conexão ao salvar critérios");
    });
}

// Função para carregar critérios (atualizada)
function loadCriteria() {
  fetch("/criteria")
    .then((response) => response.json())
    .then((criteria) => {
      // Palavras simples
      document.getElementById("productive-keywords").value =
        criteria.productive_keywords.join("\n");
      document.getElementById("unproductive-keywords").value =
        criteria.unproductive_keywords.join("\n");

      // Configurações gerais
      document.getElementById("min-length").value = criteria.min_length;
      document.getElementById("max-length").value = criteria.max_length;
      document.getElementById("productive-weight").value =
        criteria.productive_weight || 2;
      document.getElementById("unproductive-weight").value =
        criteria.unproductive_weight || 3;
    })
    .catch((error) => {
      console.error("Erro ao carregar critérios:", error);
    });
}

// Função para resetar critérios (atualizada)
function resetCriteria() {
  if (
    confirm(
      "Restaurar critérios padrão? Isso substituirá suas configurações atuais."
    )
  ) {
    // Palavras produtivas
    document.getElementById("productive-keywords").value = [
      "reunião",
      "projeto",
      "trabalho",
      "relatório",
      "prazo",
      "cliente",
      "entrega",
      "solicitação",
      "orçamento",
      "contrato",
      "desenvolvimento",
      "apresentação",
      "análise",
      "negócio",
      "empresa",
      "equipe",
      "gestão",
    ].join("\n");

    // Palavras improdutivas
    document.getElementById("unproductive-keywords").value = [
      "spam",
      "promoção",
      "oferta",
      "desconto",
      "newsletter",
      "corrente",
      "loteria",
      "prêmio",
      "ganhador",
      "grátis",
      "urgente",
      "importante",
    ].join("\n");

    // Combinações
    document.getElementById("combo-words-1").value =
      "projeto, tarefa, atividade";
    document.getElementById("combo-actions-1").value = "prazo, data, entrega";

    document.getElementById("combo-words-2").value = "reunião, encontro";
    document.getElementById("combo-actions-2").value =
      "agenda, agendar, marcar";

    document.getElementById("combo-words-3").value =
      "relatório, documento, análise";
    document.getElementById("combo-actions-3").value =
      "entregar, enviar, preparar";

    // Configurações
    document.getElementById("min-length").value = 50;
    document.getElementById("max-length").value = 5000;
    document.getElementById("productive-weight").value = 2;
    document.getElementById("unproductive-weight").value = 3;

    showCriteriaSuccess(
      'Critérios restaurados para os valores padrão. Clique em "Salvar Critérios" para aplicar.'
    );
  }
}
