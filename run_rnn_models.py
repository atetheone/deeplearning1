"""
Script principal pour exécuter tous les modèles RNN
"""
from simple_rnn import train_simple_rnn
from lstm_model import train_lstm_model
from gru_model import train_gru_model

def main():
    print("=== Entraînement des modèles RNN ===")
    
    print("\n1. Entraînement du RNN simple")
    simple_rnn_model = train_simple_rnn()
    
    print("\n2. Entraînement du modèle LSTM")
    lstm_model = train_lstm_model()
    
    print("\n3. Entraînement du modèle GRU")
    gru_model = train_gru_model()
    
    print("\nTous les modèles ont été entraînés avec succès et sauvegardés.")
    print("Fichiers créés:")
    print("- simple_rnn_model.pth")
    print("- lstm_model.pth")
    print("- gru_model.pth")
    print("- rnn_results.png")
    print("- lstm_results.png")
    print("- gru_results.png")

if __name__ == "__main__":
    main()