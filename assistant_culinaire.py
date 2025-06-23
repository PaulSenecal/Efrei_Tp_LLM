import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple
import re
import random


class TextProcessor:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocabulary_size = 0
        self.max_sequence_length = 50
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s\-àâäéèêëïîôùûüÿæœç]', '', text)
        return text.strip()
    
    def build_vocabulary(self, texts: List[str]):
        all_words = []
        for text in texts:
            words = self.clean_text(text).split()
            all_words.extend(words)
        
        unique_words = sorted(set(all_words))
        self.word_to_index = {word: idx + 1 for idx, word in enumerate(unique_words)}
        self.word_to_index['<PAD>'] = 0
        self.word_to_index['<UNK>'] = len(self.word_to_index)
        self.word_to_index['<START>'] = len(self.word_to_index)
        self.word_to_index['<END>'] = len(self.word_to_index)
        
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.vocabulary_size = len(self.word_to_index)
    
    def text_to_indices(self, text: str) -> List[int]:
        words = self.clean_text(text).split()
        return [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in words]
    
    def indices_to_text(self, indices: List[int]) -> str:
        words = [self.index_to_word.get(idx, '<UNK>') for idx in indices 
                if idx != 0 and idx not in [self.word_to_index['<START>'], self.word_to_index['<END>']]]
        return ' '.join(words)
    
    def create_sequences(self, texts: List[str]) -> List[Tuple[List[int], int]]:
        sequences = []
        for text in texts:
            indices = [self.word_to_index['<START>']] + self.text_to_indices(text) + [self.word_to_index['<END>']]
            
            for i in range(1, len(indices)):
                input_seq = indices[:i]
                target = indices[i]
                
                if len(input_seq) <= self.max_sequence_length:
                    sequences.append((input_seq, target))
        
        return sequences


class CulinaryDataset(Dataset):
    def __init__(self, sequences: List[Tuple[List[int], int]], max_length: int):
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        
        padding_length = self.max_length - len(input_seq)
        padded_input = [0] * padding_length + input_seq
        
        return torch.tensor(padded_input, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class CulinaryLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2):
        super(CulinaryLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocabulary_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        
        dropped = self.dropout(last_hidden)
        fc1_out = self.relu(self.fc1(dropped))
        output = self.fc2(fc1_out)
        
        return output


class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, dataloader: DataLoader, epochs: int = 50):
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Époque {epoch + 1}/{epochs}, Perte: {avg_loss:.4f}")


class TextGenerator:
    def __init__(self, model: nn.Module, processor: TextProcessor, device: str = 'cpu'):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        
    def generate(self, seed_text: str, max_length: int = 50, temperature: float = 0.8) -> str:
        self.model.eval()
        
        current_indices = [self.processor.word_to_index['<START>']] + self.processor.text_to_indices(seed_text)
        generated_text = seed_text
        
        with torch.no_grad():
            for _ in range(max_length):
                if len(current_indices) > self.processor.max_sequence_length:
                    input_indices = current_indices[-self.processor.max_sequence_length:]
                else:
                    padding_length = self.processor.max_sequence_length - len(current_indices)
                    input_indices = [0] * padding_length + current_indices
                
                input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
                output = self.model(input_tensor)
                probabilities = torch.softmax(output[0] / temperature, dim=0)
                
                next_index = torch.multinomial(probabilities, 1).item()
                
                if next_index == self.processor.word_to_index['<END>']:
                    break
                
                next_word = self.processor.index_to_word.get(next_index, '<UNK>')
                generated_text += ' ' + next_word
                current_indices.append(next_index)
                
                if next_word in ['.', '!', '?']:
                    break
        
        return generated_text


class CulinaryChatbot:
    def __init__(self, model: nn.Module, processor: TextProcessor, device: str = 'cpu'):
        self.generator = TextGenerator(model, processor, device)
        self.processor = processor
        self.intent_keywords = {
            'recette': ['recette', 'préparer', 'cuisiner', 'faire'],
            'conseil': ['conseil', 'astuce', 'comment', 'technique'],
            'ingredient': ['ingrédient', 'produit', 'aliment'],
            'cuisson': ['cuisson', 'cuire', 'température', 'temps'],
            'dessert': ['dessert', 'gâteau', 'tarte', 'sucré'],
            'plat': ['plat', 'viande', 'poisson', 'légume']
        }
        
        self.intent_starters = {
            'recette': "Pour préparer ce plat",
            'conseil': "Mon conseil culinaire",
            'ingredient': "Cet ingrédient",
            'cuisson': "Pour la cuisson",
            'dessert': "Ce dessert délicieux",
            'plat': "Ce plat savoureux",
            'general': "En cuisine"
        }
    
    def detect_intent(self, user_input: str) -> str:
        user_input_lower = user_input.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def generate_response(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        starter = self.intent_starters[intent]
        
        response = self.generator.generate(starter, max_length=40)
        
        return self.format_response(response)
    
    def format_response(self, response: str) -> str:
        response = response.strip()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        return response[0].upper() + response[1:] if response else response
    
    def chat(self):
        print("Assistant Culinaire: Bonjour! Je suis votre assistant culinaire virtuel.")
        print("Demandez-moi des recettes, conseils ou informations sur la cuisine.")
        print("Tapez 'quit' pour terminer.\n")
        
        while True:
            user_input = input("Vous: ").strip()
            
            if user_input.lower() == 'quit':
                print("Assistant Culinaire: Au revoir et bon appétit!")
                break
            
            if not user_input:
                continue
            
            response = self.generate_response(user_input)
            print(f"Assistant Culinaire: {response}\n")


def create_culinary_dataset() -> List[str]:
    base_texts = [
        "Pour préparer une délicieuse ratatouille, découpez les légumes en rondelles régulières",
        "La blanquette de veau traditionnelle nécessite une cuisson douce et longue",
        "Le secret d'un risotto crémeux est l'ajout progressif du bouillon chaud",
        "Une pâte brisée réussie demande de travailler rapidement le beurre froid",
        "Les macarons parfaits nécessitent une précision dans les mesures",
        "La sauce béarnaise se prépare délicatement au bain-marie",
        "Le bœuf bourguignon mijote lentement dans le vin rouge",
        "Un soufflé au fromage demande des blancs montés en neige ferme",
        "La crème anglaise est prête quand elle nappe la cuillère",
        "Les légumes rôtis au four développent des saveurs caramélisées",
        "Le pain maison nécessite un bon pétrissage et patience",
        "La mousse au chocolat légère incorpore délicatement les blancs",
        "Une quiche lorraine authentique utilise crème et lardons",
        "Le confit de canard se conserve dans sa propre graisse",
        "La vinaigrette équilibrée respecte les proportions huile vinaigre",
        "Les pâtes fraîches cuisent rapidement dans l'eau bouillante salée",
        "Le fondant au chocolat a un cœur coulant après cuisson précise",
        "Caraméliser les oignons demande patience et feu doux",
        "La mayonnaise maison se monte en ajoutant l'huile doucement",
        "Un bon pot-au-feu mijote avec des légumes de saison",
        "Blanchir les légumes préserve leur couleur éclatante",
        "Déglacer une poêle récupère les sucs savoureux",
        "Mariner la viande attendrit et parfume délicatement",
        "Le beurre clarifié supporte les hautes températures",
        "Réduire une sauce concentre intensément les saveurs",
        "Les œufs pochés demandent une eau frémissante vinaigrée",
        "Braiser combine saisir et mijoter pour attendrir",
        "Une émulsion réussie lie harmonieusement les ingrédients",
        "Flamber fait évaporer l'alcool en gardant l'arôme",
        "Sauter à feu vif saisit et conserve les textures"
    ]
    
    expanded_dataset = []
    for text in base_texts:
        expanded_dataset.append(text)
        words = text.split()
        if len(words) > 8:
            expanded_dataset.append(' '.join(words[:len(words)//2]))
            expanded_dataset.append(' '.join(words[len(words)//2:]))
    
    return expanded_dataset


def save_model_and_processor(model: nn.Module, processor: TextProcessor, 
                           model_path: str = "culinary_model.pth",
                           processor_path: str = "processor.pkl"):
    torch.save(model.state_dict(), model_path)
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"Modèle sauvegardé: {model_path}")
    print(f"Processeur sauvegardé: {processor_path}")


def load_model_and_processor(vocabulary_size: int,
                           model_path: str = "culinary_model.pth",
                           processor_path: str = "processor.pkl"):
    model = CulinaryLanguageModel(vocabulary_size)
    model.load_state_dict(torch.load(model_path))
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    return model, processor


def main():
    print("=== Assistant Culinaire LLM - PyTorch ===\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositif utilisé: {device}\n")
    
    print("Création du dataset...")
    texts = create_culinary_dataset()
    
    print("Préparation du processeur de texte...")
    processor = TextProcessor()
    processor.build_vocabulary(texts)
    print(f"Taille du vocabulaire: {processor.vocabulary_size} mots")
    
    print("\nCréation des séquences d'entraînement...")
    sequences = processor.create_sequences(texts)
    print(f"Nombre de séquences: {len(sequences)}")
    
    dataset = CulinaryDataset(sequences, processor.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("\nInitialisation du modèle...")
    model = CulinaryLanguageModel(processor.vocabulary_size)
    
    print("\nEntraînement du modèle...")
    trainer = ModelTrainer(model, device)
    trainer.train(dataloader, epochs=50)
    
    print("\nSauvegarde du modèle...")
    save_model_and_processor(model, processor)
    
    print("\n=== Tests de génération ===\n")
    generator = TextGenerator(model, processor, device)
    
    test_phrases = [
        "Pour préparer",
        "Mon conseil",
        "La cuisson",
        "Ce dessert"
    ]
    
    for phrase in test_phrases:
        generated = generator.generate(phrase, max_length=30)
        print(f"Début: '{phrase}'")
        print(f"Génération: '{generated}'\n")
    
    print("\n=== Lancement du Chatbot ===\n")
    chatbot = CulinaryChatbot(model, processor, device)
    chatbot.chat()


if __name__ == "__main__":
    main()