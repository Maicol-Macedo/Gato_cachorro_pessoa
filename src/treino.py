import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time

"""
SCRIPT DE TREINAMENTO: CLASSIFICAÇÃO DE IMAGENS COM RESNET50 (TRANSFER LEARNING)

Este script realiza o fine-tuning de uma rede neural ResNet50 pré-treinada para
classificação de imagens personalizadas.

HARDWARE ALVO:
    - GPU: Série NVIDIA RTX 40xx (Arquitetura Ada Lovelace)
    - CPU: Processador 6 Núcleos / 12 Threads (AMD/Intel)

ESTRUTURA DE DIRETÓRIOS ESPERADA:
    ./dataset/train/  -> Imagens de treino organizadas por classe (pastas)
    ./dataset/val/    -> Imagens de validação organizadas por classe (pastas)
    ./modelos/        -> Local onde o modelo treinado (.pth) será salvo
"""

# ==============================================================================
# 1. CONFIGURAÇÕES E HIPERPARÂMETROS
# ==============================================================================

# Configurações otimizadas para RTX 40xx (8GB VRAM+) e CPU 6/12
# O tamanho do lote (64) foi dimensionado para maximizar o uso da VRAM e Tensor Cores.
BATCH_SIZE = 64  

# Define o número de subprocessos para carregar dados. 
# Em um processador 6/12, usar metade das threads (6) equilibra carga e pre-fetch.
NUM_WORKERS = 6  

# Dimensão padrão de entrada para a ResNet (224x224 pixels).
TAMANHO_IMG = 224

# Número de passagens completas pelo dataset. 
# Definido em 25 para refinar os pesos da camada final, dado que a convergência é rápida.
EPOCHS = 25      

# Taxa de aprendizado para o otimizador Adam.
LEARNING_RATE = 0.001

# ==============================================================================
# 2. DEFINIÇÃO DE CAMINHOS
# ==============================================================================
CAMINHO_TRAIN = './dataset/train'
CAMINHO_VAL   = './dataset/val'
CAMINHO_SALVAR = './modelos'

def main():
    """
    Função principal que orquestra a configuração do hardware, preparação dos dados,
    carregamento do modelo e loop de treinamento.
    """
    
    # --------------------------------------------------------------------------
    # 3. DETECÇÃO E CONFIGURAÇÃO DE HARDWARE
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HARDWARE DETECTADO ---")
    print(f"Dispositivo de Processamento: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Ativa: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM Total Disponível: {props.total_memory / 1e9:.2f} GB")
        
        # Otimização para arquiteturas modernas (Ampere/Ada Lovelace).
        # Permite que o cuDNN encontre o melhor algoritmo de convolução para o hardware atual.
        torch.backends.cudnn.benchmark = True 
    else:
        print("AVISO: GPU não detectada! O treinamento será executado na CPU e será significativamente mais lento.")

    # --------------------------------------------------------------------------
    # 4. PRÉ-PROCESSAMENTO E DATA AUGMENTATION
    # --------------------------------------------------------------------------
    # Transformações para o conjunto de TREINO (inclui aumento de dados para generalização)
    transform_train = transforms.Compose([
        transforms.Resize((TAMANHO_IMG, TAMANHO_IMG)),
        transforms.RandomHorizontalFlip(),      # Espelhamento horizontal aleatório
        transforms.RandomRotation(10),          # Rotação leve (+/- 10 graus)
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Variação sutil de brilho/contraste
        transforms.ToTensor(),                  # Converte imagem PIL para Tensor PyTorch (0-1)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalização ImageNet
    ])

    # Transformações para o conjunto de VALIDAÇÃO (apenas redimensionamento e normalização)
    transform_val = transforms.Compose([
        transforms.Resize((TAMANHO_IMG, TAMANHO_IMG)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --------------------------------------------------------------------------
    # 5. CARREGAMENTO DOS DATASETS
    # --------------------------------------------------------------------------
    if not os.path.exists(CAMINHO_TRAIN):
        print(f"ERRO CRÍTICO: Diretório de treino '{CAMINHO_TRAIN}' não encontrado.")
        return

    train_dataset = datasets.ImageFolder(CAMINHO_TRAIN, transform=transform_train)
    val_dataset = datasets.ImageFolder(CAMINHO_VAL, transform=transform_val)

    print(f"\nImagens carregadas para Treino: {len(train_dataset)}")
    print(f"Classes identificadas: {train_dataset.classes}")

    # Criação dos DataLoaders
    # pin_memory=True: Reserva memória na RAM paginada para acelerar a transferência para a VRAM da GPU.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --------------------------------------------------------------------------
    # 6. CONFIGURAÇÃO DO MODELO (TRANSFER LEARNING)
    # --------------------------------------------------------------------------
    print("Carregando arquitetura ResNet50 pré-treinada...")
    # Carrega pesos padrão treinados na ImageNet (v2)
    weights = models.ResNet50_Weights.DEFAULT
    modelo = models.resnet50(weights=weights)
    
    # Congelamento das camadas convolucionais (Feature Extractor)
    # Isso impede que os pesos pré-treinados sejam destruídos no início do treino.
    for param in modelo.parameters():
        param.requires_grad = False
    
    # Substituição da camada totalmente conectada (Head)
    # A última camada original (1000 classes) é trocada para o número de classes do nosso dataset.
    # Apenas esta camada terá seus pesos atualizados (treinados).
    num_classes = len(train_dataset.classes)
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    
    # Envia o modelo para a GPU (ou CPU)
    modelo = modelo.to(device)

    # --------------------------------------------------------------------------
    # 7. DEFINIÇÃO DE PERDA E OTIMIZADOR
    # --------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss() # Função de perda padrão para classificação multiclasse
    
    # Otimizador Adam configurado apenas para os parâmetros da camada final (modelo.fc)
    optimizer = optim.Adam(modelo.fc.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------------------------
    # 8. LOOP DE TREINAMENTO E VALIDAÇÃO
    # --------------------------------------------------------------------------
    print(f"\n--- INICIANDO TREINAMENTO (RTX 40xx) ---")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- Fase de Treino ---
        modelo.train() # Coloca o modelo em modo de treino (ativa Dropout/BatchNorm se houver)
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Envia dados para GPU. non_blocking=True permite assincronia com a CPU.
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()           # Zera gradientes anteriores
            outputs = modelo(images)        # Forward pass
            loss = criterion(outputs, labels) # Cálculo da perda
            loss.backward()                 # Backpropagation
            optimizer.step()                # Atualização dos pesos
            
            running_loss += loss.item()
        
        # --- Fase de Validação ---
        modelo.eval() # Coloca o modelo em modo de avaliação (congela Dropout/BatchNorm)
        corretos = 0
        total = 0
        
        # Desativa o cálculo de gradientes para economizar VRAM e processamento na validação
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = modelo(images)
                _, predicted = torch.max(outputs.data, 1) # Obtém a classe com maior probabilidade
                total += labels.size(0)
                corretos += (predicted == labels).sum().item()
        
        # Métricas da Época
        acc = 100 * corretos / total
        avg_loss = running_loss / len(train_loader)
        print(f"Época {epoch+1}/{EPOCHS} | Loss Médio: {avg_loss:.4f} | Acurácia Validação: {acc:.2f}%")

    # --------------------------------------------------------------------------
    # 9. FINALIZAÇÃO E SALVAMENTO
    # --------------------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\nTempo Total de Execução: {total_time/60:.2f} minutos")

    if not os.path.exists(CAMINHO_SALVAR):
        os.makedirs(CAMINHO_SALVAR)
    
    caminho_final = os.path.join(CAMINHO_SALVAR, 'modelo_treinado.pth')
    torch.save(modelo.state_dict(), caminho_final)
    print(f"Sucesso! Modelo salvo em: {caminho_final}")

if __name__ == '__main__':
    main()
