import torch
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import sys

# --- CONFIGURAÇÕES GLOBAIS ---
# Caminho do arquivo onde os pesos do modelo treinado estão salvos.
CAMINHO_MODELO = './modelos/modelo_treinado.pth'

# Lista de classes que o modelo é capaz de identificar.
# NOTA: A ordem deve ser idêntica à usada no treino (alfabética se usou ImageFolder).
CLASSES = ['cachorros', 'gatos', 'pessoas'] 

# Configuração de dispositivo: Usa GPU (cuda) se disponível, senão usa CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def carregar_modelo() -> models.ResNet:
    """
    Carrega a arquitetura ResNet50 e os pesos treinados do disco.

    Esta função inicializa a rede, ajusta a camada final para o número de classes
    definido globalmente e carrega o estado do modelo para o dispositivo (CPU/GPU).

    Returns:
        models.ResNet: O modelo carregado em modo de avaliação (eval).

    Raises:
        SystemExit: Se o arquivo do modelo não for encontrado.
    """
    print("Carregando modelo... aguarde.")
    
    # 1. Recria a arquitetura base (ResNet50)
    # weights=None: Não baixamos pesos da ImageNet pois usaremos os nossos.
    modelo = models.resnet50(weights=None)
    
    # 2. Ajusta a camada totalmente conectada (fully connected)
    num_features = modelo.fc.in_features
    modelo.fc = torch.nn.Linear(num_features, len(CLASSES))
    
    # 3. Verifica e carrega os pesos salvos
    if not os.path.exists(CAMINHO_MODELO):
        print("ERRO CRÍTICO: Modelo não encontrado!")
        sys.exit()
        
    # map_location garante compatibilidade entre quem treinou (ex: GPU) e quem executa (ex: CPU)
    modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DEVICE))
    modelo.to(DEVICE)
    
    # 4. Modo de avaliação
    # Desativa comportamentos de treino como Dropout e atualiza estatísticas de BatchNormalization.
    modelo.eval()
    return modelo

def preprocessar_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Converte um frame do OpenCV para o formato de tensor esperado pela PyTorch.

    O OpenCV trabalha nativamente com BGR, enquanto modelos treinados com PIL/TorchVision
    esperam RGB. Esta função faz a conversão de cores, redimensionamento e normalização.

    Args:
        frame (np.ndarray): O frame de imagem capturado pela webcam (formato array numpy BGR).

    Returns:
        torch.Tensor: Tensor pronto para inferência com dimensões [1, 3, 224, 224].
    """
    # 1. Conversão de Espaço de Cor
    # OpenCV captura em BGR (Azul, Verde, Vermelho).
    # A IA foi treinada com imagens RGB (Vermelho, Verde, Azul).
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Conversão para PIL
    # Necessário para usar a pipeline de transformações do torchvision exatamente como no treino.
    imagem_pil = Image.fromarray(imagem_rgb)
    
    # 3. Definição das Transformações
    # Devem ser idênticas às usadas na validação/treino.
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensiona para entrada da ResNet
        transforms.ToTensor(),         # Converte para Tensor e escala [0, 255] -> [0, 1]
        transforms.Normalize(          # Normalização estatística padrão ImageNet
            [0.485, 0.456, 0.406],     # Médias
            [0.229, 0.224, 0.225]      # Desvios padrão
        )
    ])
    
    # 4. Criação do Batch
    # .unsqueeze(0) adiciona a dimensão do batch na posição 0.
    # Transforma [3, 224, 224] em [1, 3, 224, 224].
    return transform(imagem_pil).unsqueeze(0).to(DEVICE)

def main():
    """
    Função principal que gerencia o loop de captura da webcam e inferência.
    """
    # Inicializa o modelo
    modelo = carregar_modelo()
    
    # --- CONFIGURAÇÃO DA WEBCAM ---
    # Inicia a Webcam.
    # Nota: O índice '0' é geralmente a webcam integrada.
     cap = cv2.VideoCapture(0) 
    
    # Verifica se a câmera abriu corretamente
    if not cap.isOpened():
        print("ERRO: Não foi possível acessar a webcam.")
        return

    print("--- WEBCAM INICIADA ---")
    print("Pressione 'q' no teclado para sair.")

    # Loop de processamento em tempo real
    while True:
        # 1. Ler o frame da câmera
        # ret (bool): indica se o frame foi lido corretamente.
        # frame (np.array): a imagem em si.
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame. Encerrando.")
            break

        # 2. Preparar imagem e realizar predição
        input_tensor = preprocessar_frame(frame)
        
        with torch.no_grad(): # Desabilita gradientes para performance
            saida = modelo(input_tensor)
            
            # Calcula probabilidades
            probs = torch.nn.functional.softmax(saida, dim=1)[0] * 100
            
            # Obtém a classe com maior probabilidade
            confianca, indice = torch.max(probs, 0)
            
            classe_vencedora = CLASSES[indice.item()]
            confianca_val = confianca.item()

        # 3. Desenhar interface visual (Overlay)
        # Define a cor do texto: Verde se confiança > 70%, Vermelho caso contrário.
        cor = (0, 255, 0) if confianca_val > 70 else (0, 0, 255)
        
        texto = f"{classe_vencedora.upper()}: {confianca_val:.1f}%"
        
        # Desenha barra de fundo preta para legibilidade do texto
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        # Escreve o texto da predição
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

        # 4. Exibir o resultado
        cv2.imshow('IA Reconhecimento - PyTorch', frame)

        # 5. Controle de Saída
        # cv2.waitKey(1) espera 1ms por uma tecla.
        # 0xFF == ord('q') verifica se a tecla pressionada foi 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpeza de recursos ao encerrar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
