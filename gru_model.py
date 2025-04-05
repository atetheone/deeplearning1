import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simple_rnn import generate_sine_wave

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couche GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # Initialisation de l'état caché si non fourni
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Propagation à travers la couche GRU
        out, hidden = self.gru(x, hidden)
        
        # Appliquer la couche fully connected à chaque pas de temps
        out = self.fc(out)
        
        return out, hidden
    
    def get_last_output(self, x):
        out, _ = self.forward(x)
        return out[:, -1, :]


def train_gru_model():
    """Entraîne un modèle GRU sur des données sinusoïdales"""
    # Paramètres
    input_size = 1
    hidden_size = 32
    output_size = 1
    num_layers = 2
    seq_length = 20
    
    # Générer des données
    num_samples = 200
    inputs, targets = generate_sine_wave(seq_length, num_samples, freq=0.2, noise=0.1)
    
    # Créer le modèle
    model = GRUModel(input_size, hidden_size, output_size, num_layers=num_layers, dropout=0.2)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Historique des pertes
    train_losses = []
    
    # Entraînement
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # Enregistrer la perte
        train_losses.append(loss.item())
        
        # Backward et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Évaluer le modèle
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = generate_sine_wave(seq_length, 20, freq=0.2, noise=0.1)
        test_outputs, _ = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Courbe d'apprentissage
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Courbe d\'apprentissage GRU')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    
    # Prédictions vs réalité
    plt.subplot(2, 1, 2)
    with torch.no_grad():
        sample_idx = 5
        plt.plot(test_targets[sample_idx].squeeze().numpy(), 'r-', label='Cible')
        plt.plot(test_outputs[sample_idx].squeeze().numpy(), 'b--', label='Prédiction')
        plt.legend()
        plt.title('Prédictions vs Réalité (GRU)')
    
    plt.tight_layout()
    plt.savefig('gru_results.png')
    plt.show()
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'gru_model.pth')
    
    return model


if __name__ == "__main__":
    print("Entraînement du modèle GRU...")
    trained_model = train_gru_model()
    print("Entraînement terminé!")