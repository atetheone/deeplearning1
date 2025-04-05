import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from alexnet_model import AlexNet
import time

def train_alexnet(model, epochs=5, batch_size=64, lr=0.001, max_minutes=15, patience=3):
    print(f"Entraînement avec {epochs} époques (max {max_minutes} minutes)")
    
    # Transformations pour les images
    transform = transforms.Compose([
        transforms.Resize(227),  # AlexNet nécessite 227x227
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Télécharger et charger CIFAR-10
    print("Téléchargement et préparation du jeu de données CIFAR-10...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, 
                                   download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil: {device}")
    model = model.to(device)
    
    start_time = time.time()
    best_accuracy = 0
    patience_counter = 0
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Réinitialisation des gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}')
                running_loss = 0.0
            
            # Vérifier le temps écoulé
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes > max_minutes:
                print(f"Entraînement arrêté après {elapsed_minutes:.1f} minutes (limite: {max_minutes})")
                break
        
        # Vérifier si on a dépassé le temps maximal
        if elapsed_minutes > max_minutes:
            break
            
        # Ajustement du taux d'apprentissage
        scheduler.step()
        
        # Mode évaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy on validation set: {accuracy:.2f}%')
        
        # Arrêt précoce
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_alexnet_cifar10.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Arrêt précoce à l\'époque {epoch+1}, aucune amélioration depuis {patience} époques')
            break
        
        # Afficher le temps d'époque
        epoch_time = time.time() - epoch_start
        print(f"Temps pour l'époque {epoch+1}: {epoch_time:.2f} secondes")
    
    # Sauvegarde du modèle final
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print(f'Entraînement terminé en {(time.time() - start_time)/60:.2f} minutes')
    print(f'Meilleure précision: {best_accuracy:.2f}%')
    
    return model

if __name__ == "__main__":
    # Création du modèle (avec 10 classes pour CIFAR-10 au lieu de 1000)
    model = AlexNet(num_classes=10)
    
    print("Début de l'entraînement d'AlexNet...")
    trained_model = train_alexnet(
        model, 
        epochs=5,       
        batch_size=64,
        lr=0.001,
        max_minutes=600, # Limite de temps
        patience=2      # Arrêt précoce
    )
    print("Entraînement terminé !")