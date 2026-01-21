# 📷 Classificador de Imagens em Tempo Real com PyTorch (ResNet50)

Este projeto consiste em uma Inteligência Artificial capaz de classificar imagens em tempo real utilizando a arquitetura **ResNet50** com a técnica de **Transfer Learning**.

O sistema foi treinado para identificar três classes específicas: **Cachorros, Gatos e Pessoas**, podendo realizar a inferência via Webcam, Câmeras IP (celular) ou imagens estáticas.

## 🚀 Funcionalidades

- **Treinamento Personalizado:** Script robusto de *fine-tuning* que congela as camadas convolucionais da ResNet50 e treina apenas a camada final.
- **Detecção em Tempo Real:** Suporte para inferência via Webcam e Câmeras IP (ex: DroidCam, IP Webcam).
- **Processamento Otimizado:** O código detecta automaticamente se há uma GPU NVIDIA (CUDA) disponível para acelerar tanto o treino quanto a inferência.
- **Data Augmentation:** O pipeline de treino inclui rotação, espelhamento e ajuste de cor para aumentar a generalização do modelo.

## 📦 Instalação

Recomendamos o uso do **Miniconda** ou **Anaconda** para gerenciar o ambiente.

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **PyTorch & Torchvision:** Framework de Deep Learning.
* **OpenCV:** Manipulação de vídeo e interface visual.
* **Pillow (PIL):** Processamento de imagens.
* **ResNet50:** Arquitetura de rede neural convolucional (CNN) pré-treinada na ImageNet.

## 📂 Estrutura do Projeto

A organização dos arquivos segue o padrão abaixo. Certifique-se de manter os scripts na pasta `src` e as imagens/modelos na raiz para que os caminhos funcionem corretamente.

```text
projeto/
├── dataset/                  # Pasta com as imagens
│   ├── train/                # Imagens de treinamento
│   │   ├── cachorros/
│   │   ├── gatos/
│   │   └── pessoas/
│   └── val/                  # Imagens de validação (mesma estrutura)
├── modelos/                  # Onde o arquivo .pth será salvo automaticamente
├── src/                      # Código-fonte do projeto
│   ├── classificar.py        # Teste em imagem estática
│   ├── ipcam.py              # Teste em câmera via IP/Wi-Fi
│   ├── treino.py             # Script de treinamento da IA
│   └── webcam.py             # Teste na webcam local
├── requirements.txt          # Dependências do projeto
└── README.md
