# 📸 Classificador de Imagens em Tempo Real com PyTorch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-orange?logo=pytorch)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.7.1%2Bcu118-orange?logo=pytorch)


Este projeto implementa uma Inteligência Artificial capaz de classificar imagens em tempo real utilizando a arquitetura **ResNet50** com a técnica de **Transfer Learning**. O modelo foi treinado para diferenciar **Cachorros, Gatos e Pessoas**, suportando inferência via Webcam, Câmeras IP ou arquivos estáticos.

---

## 📑 Índice
1. [Funcionalidades](#-funcionalidades)
2. [Tecnologias](#-tecnologias-utilizadas)
3. [Instalação](#-instalação)
4. [Configuração do Dataset (Kaggle)](#-configuração-do-dataset-kaggle)
5. [Como Usar](#-como-usar)
6. [Estrutura do Projeto](#-estrutura-do-projeto)

---

## 🚀 Funcionalidades

* **🧠 Transfer Learning:** Utiliza a ResNet50 pré-treinada na ImageNet, congelando camadas convolucionais e treinando apenas o classificador final (*fine-tuning*).
* **⚡ Alta Performance:** Detecção automática de GPU NVIDIA (CUDA) para aceleração de treino e inferência.
* **📹 Múltiplas Entradas:** Suporte nativo para Webcam local, Câmeras IP (celular via Wi-Fi) e imagens estáticas.
* **🔄 Data Augmentation:** Pipeline robusto com rotação, espelhamento e ajuste de cor para evitar *overfitting*.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Core AI:** PyTorch & Torchvision
* **Processamento de Imagem:** OpenCV & Pillow (PIL)
* **Modelo Base:** ResNet50

---

## 🐍 Configuração do Ambiente (Miniconda)

Recomendamos fortemente o uso do **Miniconda** para gerenciar as dependências e evitar conflitos com o Python do sistema.

### 1. Baixar e Instalar
Baixe o instalador para seu sistema operacional no site oficial:
* [Miniconda Download (Windows/Mac/Linux)](https://docs.conda.io/en/latest/miniconda.html)

**Dica de Instalação (Windows):**
Durante a instalação, marque a opção *"Add Miniconda3 to my PATH environment variable"* (embora o instalador diga que não é recomendado, facilita muito para iniciantes rodarem comandos direto no terminal).

### 2. Inicializar (Apenas Linux/Mac)
Se estiver no Linux ou Mac, abra o terminal após instalar e rode:

    conda init bash
    # Feche e abra o terminal novamente

### 3. Criar o Ambiente Virtual
No terminal (ou Anaconda Prompt no Windows), execute os comandos abaixo para criar um ambiente isolado com Python 3.10:

    # Cria o ambiente chamado 'torch-env'
    conda create -n torch-env python=3.10 -y

    # Ativa o ambiente (Obrigatório antes de rodar o projeto)
    conda activate torch-env

---

## 📦 Instalação do Projeto

Com o ambiente ativado (`conda activate torch-env`), instale as bibliotecas necessárias:

    pip install -r requirements.txt

---

## 📊 Configuração do Dataset (Kaggle)

Para treinar o modelo, é necessário baixar as imagens. O script espera uma estrutura de pastas específica que pode ser obtida via Kaggle.

### Passo 1: Obter Credenciais (`kaggle.json`)
1. Acesse sua conta no [Kaggle](https://www.kaggle.com/).
2. Vá em **Settings** > Seção **API** > Clique em **Create New Token**.
3. Um arquivo `kaggle.json` será baixado.

### Passo 2: Configurar Autenticação
Mova o arquivo `kaggle.json` para o local correto:

* **Linux/Mac:**
    
    mkdir -p ~/.kaggle
    mv ~/Downloads/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

* **Windows:** Mova para `C:\Users\<SEU_USUARIO>\.kaggle\kaggle.json`.

### Passo 3: Baixar os Dados
Substitua `usuario/dataset` pelo link do dataset desejado que contenha as classes (Gatos, Cachorros, Pessoas).

    # Instala o cliente API (se ainda não instalou)
    pip install kaggle

    # Baixa o dataset
    kaggle datasets download -d nome-do-usuario/nome-do-dataset

    # Descompacte e organize para que fique EXATAMENTE assim:
    # projeto/
    # ├── dataset/
    # │   ├── train/  (imagens de treino)
    # │   └── val/    (imagens de validação)

---

## 💻 Como Usar

Siga a ordem abaixo para garantir o funcionamento correto.

### 1. Treinamento (`treino.py`)
Obrigatório na primeira execução. O script lerá as imagens, treinará a IA e salvará o arquivo `.pth`.

    python src/treino.py

> **Nota:** O modelo final será salvo em `modelos/modelo_treinado.pth`.

### 2. Teste com Webcam (`webcam.py`)
Para classificação em tempo real usando a webcam do PC. Ideal para testar a IA diretamente no seu computador/notebook.

**Configuração:**
Por padrão, o script usa a câmera principal do sistema (índice `0`). Se for usar uma webcam externa (ou virtual), pode ser necessário alterar no arquivo `src/webcam.py`:
`cap = cv2.VideoCapture(1)` # ou 2, dependendo da porta USB

**Como executar:**

    python src/webcam.py

* **Interface Visual:** Uma barra preta no topo exibe a classe prevista. O texto ficará **Verde** se a confiança da IA for maior que 70%, e **Vermelho** se for menor ou igual.
* **Controles:** Pressione a tecla `q` enquanto a janela do vídeo estiver em foco para encerrar e liberar a câmera.

### 3. Teste com Câmera IP (`ipcam.py`)
Para usar a câmera do seu smartphone como uma webcam sem fio através do aplicativo **Iriun Webcam**. Diferente de webcams IP baseadas em URL, o Iriun cria uma câmera virtual no sistema operacional do seu PC.

**Passo a Passo (Iriun Webcam):**
1. Instale o app **Iriun Webcam** no seu celular (Android/iOS) e instale o programa correspondente no seu computador.
2. Inicie os dois aplicativos com o celular e o computador na mesma rede Wi-Fi. O computador reconhecerá o celular como uma webcam conectada.
3. Edite o arquivo `src/ipcam.py`. Como o Iriun simula uma webcam física, você deve usar um número inteiro para selecioná-la (geralmente `1`, `2` ou `3`, dependendo de quantas webcams normais você tem instaladas). Altere a linha: `cap = cv2.VideoCapture(1)`

    python src/webcam.py

* **Controles:** Pressione `q` na janela de vídeo para sair com segurança.

### 4. Classificar Foto (`classificar.py`)
Para testar uma imagem específica salva no disco.

    python src/classificar.py caminho/da/sua_foto.jpg

---

## 📂 Estrutura do Projeto

    projeto/
    ├── dataset/                  # Imagens (Baixadas/Organizadas)
    │   ├── train/                # ├── cachorros/ | gatos/ | pessoas/
    │   └── val/                  # └── cachorros/ | gatos/ | pessoas/
    ├── modelos/                  # Salva o arquivo .pth aqui
    ├── src/                      # Scripts Python
    │   ├── classificar.py        # Inferência estática
    │   ├── ipcam.py              # Inferência via Wi-Fi
    │   ├── treino.py             # Script de Fine-Tuning
    │   └── webcam.py             # Inferência local
    ├── requirements.txt          # Lista de libs necessárias
    └── README.md                 # Documentação

---

**Desenvolvido com PyTorch 2.7.1+cu118**
