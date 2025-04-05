import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Couche récurrente
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,  # Format des entrées: (batch, seq_len, input_size)
            nonlinearity='tanh'  # Fonction d'activation
        )
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # Initialisation de l'état caché si non fourni
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        
        # Propagation à travers la couche RNN
        # out shape: (batch, seq_len, hidden_size)
        # hidden shape: (1, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        
        # Pour une tâche many-to-many, appliquer la couche fully connected à chaque pas de temps
        # Reshape out pour appliquer la couche linéaire à tous les pas de temps
        out = self.fc(out)
        
        return out, hidden
    
    # Pour extraire la prédiction finale (utile pour les tâches many-to-one)
    def get_last_output(self, x):
        out, _ = self.forward(x)
        return out[:, -1, :]  # Prendre la sortie du dernier pas de temps


def generate_sine_wave(seq_length, num_samples, freq=0.1, noise=0.05):
    """Génère des données de type onde sinusoïdale pour tester le RNN"""
    x = np.linspace(0, 2*np.pi, num_samples*seq_length)
    y = np.sin(freq * x) + np.random.normal(0, noise, x.shape)
    
    # Créer des séquences d'entrée et cibles
    input_seqs = []
    target_seqs = []
    
    for i in range(num_samples):
        start_idx = i * seq_length
        end_idx = start_idx + seq_length
        
        # Séquence d'entrée : t à t+seq_length-1
        # Séquence cible : t+1 à t+seq_length
        input_seq = y[start_idx:end_idx-1]
        target_seq = y[start_idx+1:end_idx]
        
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    # Convertir en tensors PyTorch
    input_seqs = torch.FloatTensor(input_seqs).unsqueeze(-1)  # (num_samples, seq_length-1, 1)
    target_seqs = torch.FloatTensor(target_seqs).unsqueeze(-1)  # (num_samples, seq_length-1, 1)
    
    return input_seqs, target_seqs


def train_simple_rnn():
    """Entraîne le modèle RNN simple sur des données sinusoïdales"""
    # Paramètres
    input_size = 1      # Dimension de chaque point d'entrée
    hidden_size = 16    # Dimension de l'état caché
    output_size = 1     # Dimension de chaque point de sortie
    seq_length = 10     # Longueur de la séquence
    
    # Générer les données
    num_samples = 100
    inputs, targets = generate_sine_wave(seq_length, num_samples)
    
    # Créer le modèle
    model = SimpleRNN(input_size, hidden_size, output_size)
    
    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Historique des pertes pour visualisation
    train_losses = []
    
    # Entraînement
    num_epochs = 50
    
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
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Évaluer le modèle
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = generate_sine_wave(seq_length, 10)
        test_outputs, _ = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    # Visualiser les résultats
    plt.figure(figsize=(10, 6))
    
    # Courbe d'apprentissage
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    
    # Prédictions vs réalité
    plt.subplot(2, 1, 2)
    with torch.no_grad():
        sample_idx = 0
        plt.plot(test_targets[sample_idx].squeeze().numpy(), 'r-', label='Cible')
        plt.plot(test_outputs[sample_idx].squeeze().numpy(), 'b--', label='Prédiction')
        plt.legend()
        plt.title('Prédictions vs Réalité')
    
    plt.tight_layout()
    plt.savefig('rnn_results.png')
    plt.show()
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'simple_rnn_model.pth')
    
    return model


if __name__ == "__main__":
    print("Entraînement du RNN simple...")
    trained_model = train_simple_rnn()
    print("Entraînement terminé!")