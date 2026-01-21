import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

# --- CONFIGURAÇÕES GLOBAIS ---
# Caminho do arquivo onde os pesos do modelo treinado estão salvos.
# IMPORTANTE: Deve corresponder ao caminho definido no script de treinamento.
CAMINHO_MODELO = './modelos/modelo_treinado.pth' 

# Lista de classes que o modelo é capaz de identificar.
# NOTA: A ordem alfabética é crucial e deve corresponder à ordem usada durante o treinamento
# (geralmente definida pelo ImageFolder ou similar).
CLASSES = ['cachorros', 'gatos', 'pessoas'] 

def carregar_modelo(device: torch.device) -> models.ResNet:
    """
    Reconstrói a arquitetura da rede neural e carrega os pesos treinados.

    Esta função inicializa uma ResNet50, modifica a última camada totalmente conectada (fc)
    para corresponder ao número de classes do nosso problema e carrega o estado (pesos)
    do arquivo especificado.

    Args:
        device (torch.device): O dispositivo onde o modelo será alocado ('cpu' ou 'cuda').

    Returns:
        models.ResNet: O modelo PyTorch carregado e colocado em modo de avaliação (eval).
    
    Raises:
        SystemExit: Se o arquivo do modelo não for encontrado no caminho especificado.
    """
    # 1. Recria a arquitetura exata usada no treino (ResNet50)
    # 'weights=None' indica que não queremos baixar os pesos padrão da ImageNet,
    # pois carregaremos nossos próprios pesos treinados logo abaixo.
    modelo = models.resnet50(weights=None) 
    
    # 2. Ajusta a camada final para o número correto de saídas (classes)
    num_features = modelo.fc.in_features
    modelo.fc = torch.nn.Linear(num_features, len(CLASSES))
    
    # 3. Carrega os pesos salvos do disco
    if os.path.exists(CAMINHO_MODELO):
        # map_location garante que o modelo seja carregado no dispositivo correto (CPU/GPU)
        modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=device))
        
        # Move o modelo para o dispositivo configurado
        modelo.to(device)
        
        # Coloca o modelo em modo de avaliação.
        # Isso é essencial para "congelar" camadas como Dropout e BatchNorm durante a inferência.
        modelo.eval() 
        return modelo
    else:
        print(f"ERRO CRÍTICO: Modelo não encontrado no caminho: {CAMINHO_MODELO}")
        sys.exit()

def predizer_imagem(caminho_imagem: str, modelo: models.ResNet, device: torch.device) -> None:
    """
    Processa uma imagem e realiza a inferência usando o modelo carregado.

    A função aplica as transformações necessárias na imagem, passa-a pelo modelo
    e imprime os resultados com as probabilidades de cada classe.

    Args:
        caminho_imagem (str): O caminho do arquivo de imagem a ser classificado.
        modelo (models.ResNet): O modelo carregado (retornado por carregar_modelo).
        device (torch.device): O dispositivo de processamento atual.
    """
    # Define as transformações de pré-processamento.
    # Devem ser IDÊNTICAS às usadas no conjunto de validação durante o treino.
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensiona para o padrão da ResNet
        transforms.ToTensor(),         # Converte para Tensor PyTorch e normaliza pixels para [0,1]
        transforms.Normalize(          # Normaliza usando média e desvio padrão padrão da ImageNet
            [0.485, 0.456, 0.406],     # Médias (RGB)
            [0.229, 0.224, 0.225]      # Desvios Padrão (RGB)
        )
    ])
    
    # Tenta carregar a imagem do disco
    try:
        # .convert('RGB') garante que imagens com 4 canais (PNG transparente) ou 1 canal (Grayscale)
        # sejam convertidas para o padrão de 3 canais de cor esperado pelo modelo.
        imagem = Image.open(caminho_imagem).convert('RGB')
    except Exception as e:
        print(f"Erro ao abrir imagem '{caminho_imagem}': {e}")
        print("Verifique se o caminho está correto e se o arquivo é uma imagem válida.")
        return

    # Prepara a imagem para a rede neural
    # transform(imagem): Aplica as transformações definidas acima (Retorna Tensor [3, 224, 224])
    # .unsqueeze(0): Adiciona uma dimensão extra para o batch (Batch Size).
    #                A rede espera entrada [Batch, Canais, Altura, Largura], ou seja [1, 3, 224, 224].
    imagem_tensor = transform(imagem).unsqueeze(0).to(device)
    
    # Realiza a inferência
    # torch.no_grad() desativa o cálculo de gradientes, economizando memória e processamento,
    # já que não estamos treinando a rede agora.
    with torch.no_grad():
        saida = modelo(imagem_tensor) # Passa a imagem pela rede (Forward pass)
        
        # Pós-processamento da saída
        # Softmax converte os números brutos (logits) em probabilidades (0 a 1)
        porcentagens = torch.nn.functional.softmax(saida, dim=1)[0] * 100
        
        # Identifica o índice com o maior valor na saída
        _, indice_vencedor = torch.max(saida, 1)
        
        # Recupera o nome da classe e a confiança baseada no índice
        vencedor = CLASSES[indice_vencedor.item()]
        confianca = porcentagens[indice_vencedor.item()].item()
        
        # --- EXIBIÇÃO DOS RESULTADOS ---
        print(f"\n--- RESULTADO DA ANÁLISE ---")
        print(f"Arquivo analisado: {caminho_imagem}")
        print(f"Predição Principal: {vencedor.upper()}")
        print(f"Nível de Confiança: {confianca:.2f}%")
        
        print("\nDetalhes das Probabilidades:")
        for i, classe in enumerate(CLASSES):
            print(f"  - {classe.ljust(10)}: {porcentagens[i]:.2f}%")

if __name__ == '__main__':
    # Bloco principal de execução
    
    # 1. Configuração de Hardware
    # Verifica se há uma GPU NVIDIA disponível para acelerar o processo, caso contrário usa CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de processamento: {device}")
    
    # 2. Carregamento do Modelo
    print("Carregando modelo...")
    modelo = carregar_modelo(device)
    print("Modelo carregado com sucesso.")
    
    # 3. Definição da Imagem de Entrada
    # Se o usuário passou o caminho como argumento no terminal (ex: python identificador.py foto.jpg)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Caso contrário, solicita interativamente
        img_path = input("Digite o caminho da imagem para testar (ex: teste.jpg): ")
    
    # 4. Execução da Predição
    predizer_imagem(img_path, modelo, device)
